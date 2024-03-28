import subprocess
import pandas as pd
import numpy as np
from params import MEMORY_CAP_PER_FUNCTION, MEM_UNIT, CPU_UNIT
#
# Function utilities
# 

def run_cmd(cmd):
    pong = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    result = pong.stdout.read().decode().replace('\n', '')

    return result
def decode_action(index):
    if index is not None:
        action = {}
        action["cpu"] = int(index / MEMORY_CAP_PER_FUNCTION) + 1
        action["memory"] = int(index % MEMORY_CAP_PER_FUNCTION) + 1
    else:
        action = None

    return action

def encode_action(cpu, memory):
    return (cpu - 1) * MEMORY_CAP_PER_FUNCTION + memory

def decodeCpu(index):
    return int((index - 1) / MEMORY_CAP_PER_FUNCTION) * CPU_UNIT + CPU_UNIT

def decodeMemory(index):
    return (index - 1) % MEMORY_CAP_PER_FUNCTION * MEM_UNIT + MEM_UNIT


# def cleanCouchDB(DB_NAME):
def run_shell_command(command):
    """
    Runs a shell command and captures its output and errors.

    Parameters:
    - command: A string representing the shell command to run.

    Returns:
    A tuple containing the command's output and errors.
    """
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Extract output and errors
        output = result.stdout
        errors = result.stderr
        
        return (output, errors)
    except subprocess.CalledProcessError as e:
        # Handle errors in case of command execution failure
        return (e.stdout, e.stderr)
    
def ensure_1d_array(data):
        """
        Ensure the data is a 1-dimensional numpy array.
        
        This function checks if the input data is a pandas DataFrame or Series, 
        or a numpy ndarray, and then converts it to a 1-dimensional numpy array.
        """
        if isinstance(data, pd.DataFrame):
            # If the DataFrame has more than one column, raise an error
            if data.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column.")
            data = data.iloc[:, 0].values  # Convert the single column to a numpy array
        elif isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()
        
        # Check if the data is still not 1-dimensional
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Data could not be converted to a 1-dimensional array.")
        
        return data
    