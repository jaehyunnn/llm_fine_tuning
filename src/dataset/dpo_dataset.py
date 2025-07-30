from typing import List, Dict, Union, Any

import os
import torch
import transformers
import ujson as json

from torch.utils.data import Dataset

from src.params import DataArguments
from src.const import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)

from .utils import pad_sequence

class DPODataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, List[Dict[str, Any]]],
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        padding: bool = True,
    ) -> None:
        super().__init__()
        self.list_data_dict = self._load_data(data_path)
        self.model_id = model_id
        self.processor = processor
        self.tokenizer = self._get_tokenizer(processor)
        self.data_args = data_args
        self.padding = padding
        
    def _load_data(self, data_path: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Load data from file path or return data directly if already loaded."""
        if isinstance(data_path, str):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Failed to load data from {data_path}: {e}")
        return data_path
    
    def _get_tokenizer(self, processor: transformers.ProcessorMixin) -> transformers.PreTrainedTokenizerBase:
        """Extract tokenizer from processor or return processor if it is the tokenizer."""
        return processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        all_input_ids, all_rejected, all_chosen = self._process_sources(sources)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        chosen = torch.cat(all_chosen, dim=0).to(torch.long)
        rejected = torch.cat(all_rejected, dim=0).to(torch.long)
        
        data_dict = dict(
            prompt_input_ids=input_ids,
            chosen_input_ids=chosen,
            rejected_input_ids=rejected,
        )
        
        return data_dict
    
    def _process_sources(self, sources: List[Dict[str, str]]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Process sources into input_ids and labels."""
        all_input_ids = [] 
        all_rejected = []
        all_chosen =[]
        
        if SYSTEM_MESSAGE:
            system_ids = self._process_system_message()
            all_input_ids.append(system_ids)
        
        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{sources['prompt']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        chosen_response = f"{sources['chosen']}{DEFAULT_IM_END_TOKEN}\n"
        rejected_response = f"{sources['rejected']}{DEFAULT_IM_END_TOKEN}\n"
        
        prompt_input_ids = self.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
        
        input_ids = prompt_input_ids.squeeze(0)
        chosen_input_ids = self.tokenizer(chosen_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        rejected_input_ids = self.tokenizer(rejected_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        all_input_ids.append(input_ids)
        all_chosen.append(chosen_input_ids)
        all_rejected.append(rejected_input_ids)
            
        return all_input_ids, all_rejected, all_chosen

    def _process_system_message(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Process system message into tokens."""
        system_text = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_ids = self._tokenize_text(system_text)
        return system_ids
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text and return as 1D tensor."""
        return self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

class DataCollatorForDPODataset:
    """Data collator for DPO datasets."""
    
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch with proper padding."""
        
        batch_input_ids = []
        batch_chosen_ids = []
        batch_rejected_ids = []
        
        for example in examples:
            keys = example.keys()
        
        batch_input_ids.append(example["prompt_input_ids"])
        batch_chosen_ids.append(example["chosen_input_ids"])
        batch_rejected_ids.append(example["rejected_input_ids"])
        
        prompt_input_ids = pad_sequence(batch_input_ids, padding_side='right', padding_value=self.pad_token_id)
        chosen = pad_sequence(batch_chosen_ids, padding_side='right', padding_value=self.pad_token_id)
        rejected = pad_sequence(batch_rejected_ids, padding_side='right', padding_value=self.pad_token_id)

        prompt_attention_mask = (prompt_input_ids != self.pad_token_id)
        chosen_attention_mask = (chosen != self.pad_token_id)
        rejected_attention_mask = (rejected != self.pad_token_id)
        
        return {
            'prompt_input_ids': prompt_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'chosen_input_ids': chosen,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected,
            'rejected_attention_mask': rejected_attention_mask,
        }
    
def get_dpo_dataset(
    model_id: str, 
    processor: transformers.ProcessorMixin, 
    data_args: DataArguments
) -> Dict[str, Any]:
    dpo_dataset = DPODataset(
        data_path=data_args.data_path, 
        processor=processor, 
        data_args=data_args, 
        model_id=model_id
    )
    
    pad_token_id = (
        processor.tokenizer.pad_token_id 
        if hasattr(processor, 'tokenizer') 
        else processor.pad_token_id
    )
    
    data_collator = DataCollatorForDPODataset(pad_token_id=pad_token_id)
    
    return {
        'train_dataset': dpo_dataset,
        'eval_dataset': None,
        'data_collator': data_collator
    }
