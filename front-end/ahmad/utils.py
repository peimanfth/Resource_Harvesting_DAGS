import pandas as pd
import json

def create_dataset():
    csv = pd.read_csv("runs.csv")
    columns = ["num_of_processes", "num_of_iterations", "length_of_message", "max_memory_usage", "max_cpu_usage"]
    df = pd.DataFrame(columns=columns)

    for row in csv.iterrows():
        function_name = row[1]["Function Name"]
        functions = ["AES1", "AES2", "AES3"]
        if function_name in functions:
            input_addr = row[1]["Input File"]
            with open(input_addr, "r") as f:
                data = json.load(f)
                num_of_processes = data['data'][function_name]["num_of_processes"]
                num_of_iterations = data['data'][function_name]["num_of_iterations"]
                length_of_message = data['data'][function_name]["length_of_message"]
                max_memory_usage = row[1]["Max Memory Usage"]
                max_cpu_usage = row[1]["Max CPU Usage"]
                row = {"num_of_processes": num_of_processes, "num_of_iterations": num_of_iterations, "length_of_message": length_of_message, "max_memory_usage": max_memory_usage, "max_cpu_usage": max_cpu_usage}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    df.to_csv("dataset.csv", index=False)