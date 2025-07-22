import os
import shutil
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

try:
    import bitsandbytes
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from src.train.utils import get_base_model_state_dict

# 상수 정의
MB_TO_BYTES = 2 ** 20  # 1MB in bytes
BASE_MODEL_STATE_DICT_NAME = "base_model_state_dict.bin"
CONFIG_FILE_NAME = "config.json"


class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning Trainer extending HuggingFace Trainer.
    
    특징:
    - LayerNorm과 bias 파라미터에 weight decay 미적용
    - BitsAndBytes Adam8bit 사용 시 Embedding 레이어 fp32 유지
    """
    
    def __init__(self, *args, **kwargs):
        """SFTTrainer 초기화"""
        super().__init__(*args, **kwargs)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        커스텀 optimizer 생성.
        
        Returns:
            torch.optim.Optimizer: 생성된 optimizer
            
        Raises:
            ValueError: optimizer 생성 실패 시
        """
        if self.optimizer is not None:
            logger.warning("Optimizer already exists. Returning existing optimizer.")
            return self.optimizer
            
        try:
            # Weight decay를 적용할 파라미터와 적용하지 않을 파라미터 분리
            decay_params, no_decay_params = self._get_grouped_parameters()
            
            optimizer_grouped_parameters = [
                {
                    "params": decay_params,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ]

            # Optimizer 생성
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
            # BitsAndBytes Adam8bit 특별 처리
            if HAS_BITSANDBYTES and optimizer_cls.__name__ == "Adam8bit":
                self._configure_bitsandbytes_optimizer()
            elif not HAS_BITSANDBYTES and optimizer_cls.__name__ == "Adam8bit":
                logger.warning("BitsAndBytes not installed but Adam8bit optimizer requested. "
                             "Embedding layers will not be optimized in fp32.")
                        
        except Exception as e:
            raise ValueError(f"Failed to create optimizer: {str(e)}") from e
            
        return self.optimizer
    
    def _get_grouped_parameters(self) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        """
        Weight decay 적용 여부에 따라 파라미터 그룹화.
        
        Returns:
            Tuple[List, List]: (decay_params, no_decay_params)
        """
        # LayerNorm 레이어 이름 가져오기
        try:
            decay_parameter_names = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        except Exception as e:
            logger.warning(f"Failed to get LayerNorm parameter names: {e}. "
                         "Using empty list.")
            decay_parameter_names = []
            
        # bias 파라미터는 weight decay에서 제외
        decay_parameter_names = [name for name in decay_parameter_names if "bias" not in name]
        decay_parameter_names_set = set(decay_parameter_names)
        
        # 파라미터를 한 번만 순회하여 그룹화
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if name in decay_parameter_names_set:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
                
        logger.info(f"Optimizer parameter groups - "
                   f"with weight decay: {len(decay_params)}, "
                   f"without weight decay: {len(no_decay_params)}")
        
        return decay_params, no_decay_params
    
    def _configure_bitsandbytes_optimizer(self) -> None:
        """
        BitsAndBytes Adam8bit optimizer 설정.
        Embedding 레이어를 fp32로 유지하여 정확도 향상.
        """
        try:
            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
            
            total_skipped_params = 0
            embedding_modules = []
            
            for module in self.model.modules():
                if isinstance(module, nn.Embedding):
                    # 각 Embedding 모듈의 파라미터 수 계산
                    module_params = sum(p.numel() for p in module.parameters())
                    total_skipped_params += module_params
                    
                    # fp32로 유지할 모듈 등록
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    embedding_modules.append((module.__class__.__name__, module_params))
                    
            # 로깅
            if embedding_modules:
                logger.info(f"BitsAndBytes: Optimizing {len(embedding_modules)} embedding layer(s) in fp32")
                logger.info(f"Total embedding parameters kept in fp32: {total_skipped_params / MB_TO_BYTES:.2f}M")
                
                for module_name, param_count in embedding_modules:
                    logger.debug(f"  - {module_name}: {param_count / MB_TO_BYTES:.2f}M params")
                    
        except Exception as e:
            logger.error(f"Failed to configure BitsAndBytes optimizer: {e}")
            logger.warning("Proceeding without fp32 embedding optimization.")
    
    def _save_checkpoint(self, model: torch.nn.Module, trial: Optional[Any] = None) -> None:
        """
        체크포인트 저장. LoRA 사용 시 커스텀 저장 로직 적용.
        
        Args:
            model: 저장할 모델
            trial: 하이퍼파라미터 튜닝 trial 객체 (optional)
            
        Raises:
            IOError: 체크포인트 저장 실패 시
        """
        if not getattr(self.args, 'use_lora', False):
            # LoRA를 사용하지 않는 경우 기본 저장 로직 사용
            return super()._save_checkpoint(model, trial)
            
        try:
            output_dir = self._prepare_checkpoint_directory(trial)
            
            # 모델 및 LoRA 가중치 저장
            self._save_lora_checkpoint(output_dir)
            
            # Best 체크포인트 추적
            self._track_best_checkpoint(trial)
            
            # 추가 컴포넌트 저장
            if not self.args.save_only_model:
                self._save_training_components(output_dir)
                
            # 콜백 상태 및 설정 저장
            if self.args.should_save:
                self._save_callback_states(output_dir)
                self._save_model_config(output_dir)
                
            # Hub에 푸시
            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
                
            logger.info(f"Checkpoint saved to {output_dir}")
            
        except Exception as e:
            # 부분적으로 저장된 체크포인트 정리
            if 'output_dir' in locals() and os.path.exists(output_dir):
                logger.error(f"Failed to save checkpoint. Cleaning up partial checkpoint at {output_dir}")
                try:
                    shutil.rmtree(output_dir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up partial checkpoint: {cleanup_error}")
                    
            raise IOError(f"Failed to save checkpoint: {str(e)}") from e
    
    def _prepare_checkpoint_directory(self, trial: Optional[Any] = None) -> str:
        """
        체크포인트 디렉토리 준비.
        
        Args:
            trial: 하이퍼파라미터 튜닝 trial 객체
            
        Returns:
            str: 체크포인트 저장 경로
        """
        ckpt_dir = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        
        # FLOPs 저장 (하이퍼파라미터 서치가 아닌 경우)
        if self.hp_search_backend is None and trial is None:
            self.store_flos()
            
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, ckpt_dir)
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    
    def _save_lora_checkpoint(self, output_dir: str) -> None:
        """
        LoRA 모델 및 non-LoRA 가중치 저장.
        
        Args:
            output_dir: 저장 경로
        """
        # LoRA 어댑터 저장
        self.save_model(output_dir, _internal_call=True)
        
        # Non-LoRA 가중치 저장 (base model weights)
        try:
            base_model_weights = get_base_model_state_dict(
                self.model.named_parameters(), 
                require_grad_only=False
            )
            base_model_save_path = os.path.join(output_dir, BASE_MODEL_STATE_DICT_NAME)
            torch.save(base_model_weights, base_model_save_path)
            logger.debug(f"Base Model(Non-LoRA part) weights saved to {base_model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save Base Model(Non-LoRA part) weights: {e}")
            raise
    
    def _track_best_checkpoint(self, trial: Optional[Any] = None) -> None:
        """
        Best 체크포인트 추적 및 업데이트.
        
        Args:
            trial: 하이퍼파라미터 튜닝 trial 객체
        """
        if (self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] 
            and hasattr(self.state, 'best_global_step') 
            and self.state.best_global_step):
            
            best_ckpt_name = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            run_dir = self._get_output_dir(trial=trial)
            best_ckpt_dir = os.path.join(run_dir, best_ckpt_name)

            if os.path.exists(best_ckpt_dir):
                self.state.best_ckpt = best_ckpt_dir
                logger.debug(f"Best checkpoint tracked: {best_ckpt_dir}")
    
    def _save_training_components(self, output_dir: str) -> None:
        """
        Optimizer, Scheduler, Scaler, RNG 상태 저장.
        
        Args:
            output_dir: 저장 경로
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
        콜백 상태 및 trainer state 저장.
        
        Args:
            output_dir: 저장 경로
        """
        # 콜백 상태 수집
        for callback in self.callback_handler.callbacks + [self.control]:
            if isinstance(callback, ExportableState):
                callback_name = callback.__class__.__name__
                try:
                    callback_state = callback.state()
                    
                    # 상태 저장 (list 타입 처리)
                    if callback_name in self.state.stateful_callbacks and \
                       isinstance(self.state.stateful_callbacks[callback_name], list):
                        self.state.stateful_callbacks[callback_name].append(callback_state)
                    else:
                        self.state.stateful_callbacks[callback_name] = callback_state
                        
                except Exception as e:
                    logger.warning(f"Failed to save callback state for {callback_name}: {e}")
                    
        # Trainer state 저장
        try:
            state_path = os.path.join(output_dir, TRAINER_STATE_NAME)
            self.state.save_to_json(state_path)
            logger.debug(f"Trainer state saved to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save trainer state: {e}")
            raise
    
    def _save_model_config(self, output_dir: str) -> None:
        """
        모델 설정 파일 저장.
        
        Args:
            output_dir: 저장 경로
        """
        try:
            # base_model 속성 확인
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
                config_path = os.path.join(output_dir, CONFIG_FILE_NAME)
                self.model.base_model.config.to_json_file(config_path)
                logger.debug(f"Model config saved to {config_path}")
            else:
                logger.warning("Model does not have base_model.config attribute. Skipping config save.")
        except Exception as e:
            logger.error(f"Failed to save model config: {e}")