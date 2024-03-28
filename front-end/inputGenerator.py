import json
import os

def generate_dag_input(start_length=50, end_length=500, length_increment=50,
                       aes1_process_range=(1, 4), aes2_process_range=(2, 6), aes3_process_range=(4, 8),
                       aes1_iteration_base=5000):
    configs = []

    for length_of_message in range(start_length, end_length + 1, length_increment):
        # Fixed work per process for AES1, use as base for others
        work_per_process = aes1_iteration_base // aes1_process_range[0]

        # Generate configurations with constant work per process
        for aes1_processes in range(aes1_process_range[0], aes1_process_range[1] + 1):
            aes1_iterations = work_per_process * aes1_processes
            
            # Adjust AES2 and AES3 based on work_per_process, within their process ranges
            for aes2_processes in range(aes2_process_range[0], aes2_process_range[1] + 1):
                aes2_iterations = work_per_process * aes2_processes
                
                for aes3_processes in range(aes3_process_range[0], aes3_process_range[1] + 1):
                    aes3_iterations = work_per_process * aes3_processes

                    config = {
                        "$composer": {
                            "openwhisk": {
                                "ignore_certs": True
                            },
                            "redis": {
                                "uri": "redis://174.64.28.38"
                            }
                        },
                        "data": {
                            "AES1": {
                                "length_of_message": length_of_message,
                                "num_of_iterations": aes1_iterations,
                                "num_of_processes": aes1_processes
                            },
                            "AES2": {
                                "length_of_message": length_of_message,
                                "num_of_iterations": aes2_iterations,
                                "num_of_processes": aes2_processes
                            },
                            "AES3": {
                                "length_of_message": length_of_message,
                                "num_of_iterations": aes3_iterations,
                                "num_of_processes": aes3_processes
                            }
                        }
                    }
                    
                    configs.append(config)

    return configs

def save_configs_to_files(configs, base_dir="inputss"):
    os.makedirs(base_dir, exist_ok=True)
    files = []
    
    for i, config in enumerate(configs):
        filename = f"aes_run{i+1}.json"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved: {filepath}")
        files.append(filepath)
    return files

# Generate the DAG configurations
dag_configs = generate_dag_input()

# Save the configurations to files
save_configs_to_files(dag_configs)
