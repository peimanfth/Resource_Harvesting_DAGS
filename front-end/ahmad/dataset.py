import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, csv_file, mode='train', test_size=1/3, random_state=None):
        # Load the dataset
        data = pd.read_csv(csv_file)
        
        # Split the data into training and testing
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        
        # Use the appropriate data based on the mode
        self.data = train_data if mode == 'train' else test_data
        
        # Inputs: num_of_iterations, num_of_processes, length_of_message
        self.X = self.data[['num_of_iterations', 'num_of_processes', 'length_of_message']].values
        
        # Outputs: max_memory_usage, max_cpu_usage
        self.y = self.data[['max_memory_usage', 'max_cpu_usage']].values
        # self.y = self.data['max_cpu_usage'].values
    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve inputs and outputs by index
        inputs = torch.tensor(self.X[idx], dtype=torch.float32)
        outputs = torch.tensor(self.y[idx], dtype=torch.float32)
        return inputs, outputs
