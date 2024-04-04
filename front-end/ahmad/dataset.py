import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, csv_file):
        # Load the dataset
        self.data = pd.read_csv(csv_file)
        
        # Inputs: num_of_processes, num_of_iterations, length_of_message
        self.X = self.data[['num_of_iterations', 'num_of_processes', 'length_of_message']].values
        
        # Outputs: max_memory_usage, max_cpu_usage
        self.y = self.data[['max_memory_usage', 'max_cpu_usage']].values

        # Normalization ranges
        self.iterations_range = (5000, 120000)
        self.processes_range = (1, 8)
        self.length_of_message_range = (50, 500)
        
        # Normalize the inputs
        # self.X[:, 0] = (self.X[:, 0] - self.iterations_range[0]) / (self.iterations_range[1] - self.processes_range[0]) # num_of_processes
        # self.X[:, 1] = (self.X[:, 1] - self.processes_range[0]) / (self.processes_range[1] - self.iterations_range[0]) # num_of_iterations
        # self.X[:, 2] = (self.X[:, 2] - self.length_of_message_range[0]) / (self.length_of_message_range[1] - self.length_of_message_range[0]) # length_of_message


    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve inputs and outputs by index
        inputs = torch.tensor(self.X[idx], dtype=torch.float32)
        outputs = torch.tensor(self.y[idx], dtype=torch.float32)
        return inputs, outputs