import json
import os

def generate_aes_dag_input(message_length=50, start_base_iterations=2000, end_base_iterations=12000, increment_iterations=200,
                           aes1_processes=2, aes2_processes=4, aes3_processes=6):
    configs = []

    for base_iterations in range(start_base_iterations, end_base_iterations + 1, increment_iterations):
        aes1_iterations = base_iterations * aes1_processes
        aes2_iterations = base_iterations * aes2_processes * 2
        aes3_iterations = base_iterations * aes3_processes * 3

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
                    "length_of_message": message_length,
                    "num_of_iterations": aes1_iterations,
                    "num_of_processes": aes1_processes
                },
                "AES2": {
                    "length_of_message": message_length,
                    "num_of_iterations": aes2_iterations,
                    "num_of_processes": aes2_processes
                },
                "AES3": {
                    "length_of_message": message_length,
                    "num_of_iterations": aes3_iterations,
                    "num_of_processes": aes3_processes
                }
            }
        }
        configs.append(config)

    return configs

def generate_video_dag_input(start_frames=2, end_frames=22, video_name="car"):
    configs = []

    for num_frames in range(start_frames, end_frames + 1):
        config = {
            "$composer": {
                "openwhisk": {
                    "ignore_certs": True
                },
                "redis": {
                    "uri": "redis://96.125.119.232"
                }
            },
            "video": video_name,
            "num_frames": num_frames
        }
        configs.append(config)

    return configs

def save_configs_to_files(configs, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    files = []
    
    for i, config in enumerate(configs):
        filename = f"run{i+1}.json"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved: {filepath}")
        files.append(filepath)
    return files

# Generate the AES and Video DAG configurations
aes_dag_configs = generate_aes_dag_input()
video_dag_configs = generate_video_dag_input()

# Save the configurations to files in specific directories
save_configs_to_files(aes_dag_configs, "peiman/aes_inputs")
save_configs_to_files(video_dag_configs, "peiman/video_inputs")
