from torch.utils.data import Dataset
import torch
class HDLDataset(Dataset):
    def __init__(self, encodings, labels, is_multilabel=False):  # Added is_multilabel parameter
        self.encodings = encodings
        self.labels = labels
        self.is_multilabel = is_multilabel

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.is_multilabel:
            item['labels'] = torch.FloatTensor(self.labels[idx])  # Convert to float for multi-label
        else:
            item['labels'] = self.labels[idx]  # Keep as long/int for single-label
        return item

    def __len__(self):
        return len(self.labels)
