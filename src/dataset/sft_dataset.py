from typing import Dict, List, Union, Any
import copy
import torch
import transformers

try:
    import ujson as json
except ImportError:
    import json
    
from torch.utils.data import Dataset

from src.params import DataArguments
from src.const import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)

from .utils import pad_sequence

__all__ = ['get_sft_dataset']


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with conversation format."""
    
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
        conversations = self._validate_and_extract_conversations(self.list_data_dict[i])
        
        all_input_ids, all_labels = self._process_conversations(conversations)
        
        input_ids = torch.cat(all_input_ids, dim=0)
        labels = torch.cat(all_labels, dim=0)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
    def _validate_and_extract_conversations(self, data_item: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate conversation format and extract conversations."""
        conversations = data_item.get('conversations', [])
        
        if not conversations or not isinstance(conversations[0], dict):
            raise ValueError("Invalid conversations format")
            
        first_conv = conversations[0]
        if 'role' in first_conv and 'content' in first_conv:
            return copy.deepcopy(conversations)
        else:
            raise ValueError(
                f"Unknown conversation format. Expected OpenAI format (role/content). "
                f"Got: {list(first_conv.keys())}"
            )
            
    def _process_conversations(self, conversations: List[Dict[str, str]]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Process conversations into input_ids and labels."""
        all_input_ids = []
        all_labels = []
        
        if SYSTEM_MESSAGE:
            system_ids, system_labels = self._process_system_message()
            all_input_ids.append(system_ids)
            all_labels.append(system_labels)
            
        for i in range(0, len(conversations), 2):
            if i + 1 >= len(conversations):
                break
                
            user_msg = conversations[i]
            assistant_msg = conversations[i + 1]
            
            input_ids, labels = self._process_conversation_pair(user_msg, assistant_msg)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
        return all_input_ids, all_labels
        
    def _process_system_message(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Process system message into tokens."""
        system_text = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_ids = self._tokenize_text(system_text)
        system_labels = torch.full_like(system_ids, IGNORE_INDEX)
        return system_ids, system_labels
        
    def _process_conversation_pair(self, user_msg: Dict[str, str], assistant_msg: Dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a user-assistant conversation pair."""
        user_text = f"{DEFAULT_IM_START_TOKEN}{user_msg['role']}\n{user_msg['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{assistant_msg['role']}\n"
        response_text = f"{assistant_msg['content']}{DEFAULT_IM_END_TOKEN}\n"
        
        user_ids = self._tokenize_text(user_text)
        response_ids = self._tokenize_text(response_text)
        
        input_ids = torch.cat([user_ids, response_ids], dim=0)
        labels = torch.cat([
            torch.full_like(user_ids, IGNORE_INDEX),
            response_ids
        ], dim=0)
        
        return input_ids, labels
        
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text and return as 1D tensor."""
        return self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)

class DataCollatorForSupervisedDataset:
    """Data collator for supervised fine-tuning datasets."""
    
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate examples into a batch with proper padding."""
        if not examples:
            raise ValueError("Cannot collate empty list of examples")
            
        batch_input_ids = [example["input_ids"] for example in examples]
        batch_labels = [example["labels"] for example in examples]
            
        input_ids = pad_sequence(
            batch_input_ids, 
            padding_side='right', 
            padding_value=self.pad_token_id
        )
        
        labels = pad_sequence(
            batch_labels, 
            padding_side='right', 
            padding_value=IGNORE_INDEX
        )
        
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
def get_sft_dataset(
    model_id: str, 
    processor: transformers.ProcessorMixin, 
    data_args: DataArguments
) -> Dict[str, Any]:
    sft_dataset = SupervisedDataset(
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
    
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=pad_token_id)

    return {
        'train_dataset': sft_dataset,
        'eval_dataset': None,
        'data_collator': data_collator
    }