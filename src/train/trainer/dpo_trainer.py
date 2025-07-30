from typing import Optional, Any, List, Tuple

import os
import shutil
import torch
from torch import nn
from typing import Union
import torch.nn.functional as F


from transformers.trainer import (
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from trl import DPOTrainer
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from src.train.utils import get_base_model_state_dict

# Constant definitions
MB_TO_BYTES = 2 ** 20  # 1MB in bytes
BASE_MODEL_STATE_DICT_NAME = "base_model_state_dict.bin"
CONFIG_FILE_NAME = "config.json"


class CustomDPOTrainer(DPOTrainer):
    
    def __init__(self, processing_class, *args, **kwargs):
        super(CustomDPOTrainer, self).__init__(processing_class=processing_class, *args, **kwargs)
        
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name
    ):
        return dataset
    
    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        concatenated_batch = {}

        concatenated_batch['prompt_input_ids'] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        concatenated_batch['prompt_attention_mask'] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
        
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        concatenated_batch['completion_input_ids'] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )

        concatenated_batch['completion_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )
        return concatenated_batch

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]], **kwargs):

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True
        
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
    
    def _save_checkpoint(self, model: torch.nn.Module, trial: Optional[Any] = None) -> None:
        """
        Save checkpoint. Apply custom saving logic when using LoRA.

        Args:
            model: The model to save
            trial: Hyperparameter tuning trial object (optional)

        Raises:
            IOError: If checkpoint saving fails
        """
        if not getattr(self.args, 'use_lora', False):
            # Use default saving logic when not using LoRA
            return super()._save_checkpoint(model, trial)

        try:
            output_dir = self._prepare_checkpoint_directory(trial)

            # Save model and LoRA weights
            self._save_lora_checkpoint(output_dir)

            # Track best checkpoint
            self._track_best_checkpoint(trial)

            # Save additional components
            if not self.args.save_only_model:
                self._save_training_components(output_dir)

            # Save callback states and config
            if self.args.should_save:
                self._save_callback_states(output_dir)
                self._save_model_config(output_dir)

            # Push to hub if requested
            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            logger.info(f"Checkpoint saved to {output_dir}")

        except Exception as e:
            # Clean up partially saved checkpoint
            if 'output_dir' in locals() and os.path.exists(output_dir):
                logger.error(f"Failed to save checkpoint. Cleaning up partial checkpoint at {output_dir}")
                try:
                    shutil.rmtree(output_dir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up partial checkpoint: {cleanup_error}")

            raise IOError(f"Failed to save checkpoint: {str(e)}") from e

    def _prepare_checkpoint_directory(self, trial: Optional[Any] = None) -> str:
        """
        Prepare checkpoint directory.

        Args:
            trial: Hyperparameter tuning trial object

        Returns:
            str: Path for saving the checkpoint
        """
        ckpt_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        # Save FLOPs (when not in hyperparameter search)
        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, ckpt_dir)

        # Create directory
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def _save_lora_checkpoint(self, output_dir: str) -> None:
        """
        Save LoRA model and non-LoRA weights.

        Args:
            output_dir: The directory where to save
        """
        # Save LoRA adapters
        self.save_model(output_dir, _internal_call=True)

        # Save non-LoRA weights (base model weights)
        try:
            base_model_weights = get_base_model_state_dict(
                self.model.named_parameters(),
                require_grad_only=False
            )
            base_model_save_path = os.path.join(output_dir, BASE_MODEL_STATE_DICT_NAME)
            torch.save(base_model_weights, base_model_save_path)
            logger.debug(f"Base Model (non-LoRA part) weights saved to {base_model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save Base Model (non-LoRA part) weights: {e}")
            raise

    def _track_best_checkpoint(self, trial: Optional[Any] = None) -> None:
        """
        Track and update the best checkpoint.

        Args:
            trial: Hyperparameter tuning trial object
        """
        if (
            self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]
            and hasattr(self.state, 'best_global_step')
            and self.state.best_global_step
        ):
            best_ckpt_name = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            run_dir = self._get_output_dir(trial=trial)
            best_ckpt_dir = os.path.join(run_dir, best_ckpt_name)

            if os.path.exists(best_ckpt_dir):
                self.state.best_ckpt = best_ckpt_dir
                logger.debug(f"Best checkpoint tracked: {best_ckpt_dir}")

    def _save_training_components(self, output_dir: str) -> None:
        """
        Save optimizer, scheduler, scaler, and RNG state.

        Args:
            output_dir: The directory where to save
        """
        try:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            self._save_rng_state(output_dir)
            logger.debug("Training components saved successfully")
        except Exception as e:
            logger.error(f"Failed to save training components: {e}")
            raise

    def _save_callback_states(self, output_dir: str) -> None:
        """
        Save callback states and trainer state.

        Args:
            output_dir: The directory where to save
        """
        # Collect callback states
        for callback in self.callback_handler.callbacks + [self.control]:
            if isinstance(callback, ExportableState):
                callback_name = callback.__class__.__name__
                try:
                    callback_state = callback.state()

                    # Save state entries (handle list types)
                    if callback_name in self.state.stateful_callbacks and isinstance(
                        self.state.stateful_callbacks[callback_name], list
                    ):
                        self.state.stateful_callbacks[callback_name].append(callback_state)
                    else:
                        self.state.stateful_callbacks[callback_name] = callback_state

                except Exception as e:
                    logger.warning(f"Failed to save callback state for {callback_name}: {e}")

        # Save trainer state
        try:
            state_path = os.path.join(output_dir, TRAINER_STATE_NAME)
            self.state.save_to_json(state_path)
            logger.debug(f"Trainer state saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save trainer state: {e}")
            raise

    def _save_model_config(self, output_dir: str) -> None:
        """
        Save model configuration file.

        Args:
            output_dir: The directory where to save
        """
        try:
            # Check for base_model.config attribute
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
                config_path = os.path.join(output_dir, CONFIG_FILE_NAME)
                self.model.base_model.config.to_json_file(config_path)
                logger.debug(f"Model config saved to {config_path}")
            else:
                logger.warning("Model does not have base_model.config attribute. Skipping config save.")
        except Exception as e:
            logger.error(f"Failed to save model config: {e}")