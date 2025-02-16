import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class MCQADataset(Dataset):
    def __init__(self, csv_path: Path):
        self.dataset = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset.loc[idx, 'question']
        options = self.dataset.loc[idx, ['option1', 'option2', 'option3', 'option4']].values
        label = int(self.dataset.loc[idx, 'correct_option_number']) - 1
        return (question, options, label)