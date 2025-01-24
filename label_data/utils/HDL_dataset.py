from torch.utils.data import Dataset
import torch
from typing import Dict, List, Any

class HDLDataset(Dataset):
    """Dataset class for VHDL code segments during training.
    
    This class handles the batching and indexing of tokenized code segments
    and their corresponding labels, supporting both single-label and multi-label cases."""
    
    def __init__(self, encodings: Dict[str, List], labels: List, is_multilabel: bool = False):
        self.encodings = encodings  # Tokenized input texts
        self.labels = labels        # Corresponding labels
        self.is_multilabel = is_multilabel

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.is_multilabel:
            item['labels'] = torch.FloatTensor(self.labels[idx])
        else:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)
