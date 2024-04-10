

from params import *
import json
from utils import run_cmd, encode_action, run_shell_command, decodeCpu, decodeMemory
from params import *
import json
from Couch import Couch
import csv
from datetime import datetime
import time
import uuid
import os
def invokeDAG(dag_ID, input,current_time):

    # Load the input JSON to pass to encode_dag_input
    with open(input) as f:
        input_json = json.load(f)
    if dag_ID == "AS":
        data = input_json.get("data", {})
    elif dag_ID == "vid":
        data = input_json
        print(data)
    
    # Encode the whole input of the DAG
    if dag_ID == "AS":
        encoded_dag_input = encode_dag_input(data)
    elif dag_ID == "vid":
        encoded_dag_input = VA_input_extractor(data)

    couch = Couch(COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD)
    couch.delete_guest_docs(ACTIVATIONS_DB_NAME)

    container_ID, error = run_shell_command(f"{WSK_CLI} {WSK_ACTION} invoke {dag_ID} -P {input} | awk '{{print $6}}'")
    # Omit tabs, newlines, and spaces
    container_ID = container_ID.replace('\n', '').replace('\t', '').replace(' ', '')
    print(container_ID)
    if error:
        print("Errors:", error)
    
    result = couch.poll_couchdb_for_results(ACTIVATIONS_DB_NAME, f'guest/{str(container_ID)}')

    # Generate a unique identifier for this DAG invocation
    unique_dag_id = f"{dag_ID}-{uuid.uuid4()}"

    # Define CSV file name
    csv_file_name = 'test.csv'

    # Get current time and format it as a string
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Use current time in directory name
    csv_dir = './logs/' + current_time
    missing_logs = 'missing_functions.log'

    os.makedirs(csv_dir, exist_ok=True)
    # with open(csv_file_name, 'a', newline='') as csvfile:
    with open(os.path.join(csv_dir, csv_file_name), 'a', newline='') as csvfile:
        # Define the CSV column names
        fieldnames = ['Unique DAG ID','DAG Input', 'Function Name', "Input Feature", 'Duration','Parallel Duration', 'Memory_Allocated', 'CPU_Allocated','Max Memory Usage', 'Max CPU Usage','Avg CPU Usage', 'Start Time', 'End Time', 'Timeout Status', 'Input File']
        # Create a CSV DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Check if file is empty to write header
        csvfile.seek(0, 2) # Move to the end of file
        if csvfile.tell() == 0: # If file is empty, write header
            writer.writeheader()
        
        # Get current time for 'Invocation Time'
        # invocation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        input_dict={}
        max_parallel_duration = 0
        for key, details in USER_CONFIG[dag_ID]["Functions"].items():
            if details['stage'] == 'parallel':
                doc = couch.get_doc_with_name(ACTIVATIONS_DB_NAME, key)
                if doc is None:
                    with open(os.path.join(csv_dir, missing_logs), "a") as f:
                        f.write(f"Parallel Function {key} not found in the database for input {input}\n")
                    continue
                duration = doc.get('duration', 0)  # Default to 0 if duration is not found
                max_parallel_duration = max(max_parallel_duration, duration)
            if details['stage'] == 'pre':
                parallel_stage_input = get_parallel_stage_input(key, couch)
                input_dict.update(parallel_stage_input)

        # print(input_dict)
        for key in USER_CONFIG[dag_ID]["Functions"]:
            doc = couch.get_doc_with_name(ACTIVATIONS_DB_NAME, key)
            # if doc does not exist, skip
            if doc is None:
                #record the missing function in a log file
                with open(os.path.join(csv_dir, missing_logs), "a") as f:
                        f.write(f"Function {key} not found in the database for input {input}\n")
                continue

            # Extract data from doc
            duration = doc.get('duration')
            # if USER_CONFIG[dag_ID]["Functions"][key]['stage'] == 'parallel':
            #     max_parallel_duration = max(max_parallel_duration, duration)
            response = doc.get('response', {})
            data = response.get('result', {})
            mem_usage = max([float(x) for x in data.get('mem_usage', ['0'])])  
            cpu_usage = max([float(x) for x in data.get('cpu_usage', ['0'])])
            avg_cpu_usage = sum([float(x) for x in data.get('cpu_usage', ['0'])])/len(data.get('cpu_usage', ['0']))  
            anotations = doc.get('annotations')
            timeout = anotations[4]['value']
            hyperparams = anotations[5]['value']
            encoded_allocation = hyperparams.get('memory')
            memory_allocated = decodeMemory(int(encoded_allocation))
            cpu_allocated = decodeCpu(int(encoded_allocation))
            start_time = doc.get('start')
            end_time = doc.get('end')
            
            # input_feature  = feature_encoder(input_dict[key]) if input_dict[key] else None
            # print(input_dict.
            if dag_ID == "AS":
                if input_dict.get(key) is not None:
                    input_feature = feature_encoder(input_dict[key])
                else:
                    input_feature = None
            elif dag_ID == "vid":
                input_feature = VA_ImgRec_input_extractor(input_dict, USER_CONFIG[dag_ID]["Functions"][key]['idx'])



            parallel_duration = None  # Initialize as None
            # Set parallel_duration for functions in the parallel stage
            if USER_CONFIG[dag_ID]["Functions"][key]['stage'] == 'parallel':
                parallel_duration = max_parallel_duration
            
            # Write data to the CSV file
            writer.writerow({
                'Unique DAG ID': unique_dag_id,
                'DAG Input':  encoded_dag_input,
                'Function Name': key,
                # 'Parallelism': key['num_of_processes'],
                "Input Feature": input_feature,
                'Duration': duration,
                'Parallel Duration': parallel_duration,
                'Memory_Allocated': memory_allocated,
                'CPU_Allocated': cpu_allocated,
                'Max Memory Usage': mem_usage,
                'Max CPU Usage': cpu_usage,
                'Avg CPU Usage': avg_cpu_usage,
                'Start Time': start_time,
                'End Time': end_time,
                'Timeout Status': timeout,
                'Input File': input
            })

    return result


def update_Function(function_ID, memory, cpu):
    """
    Update the function with the given memory and cpu
    """
    out,err = run_shell_command(f"{WSK_CLI} {WSK_ACTION} update {function_ID} --memory {encode_action(cpu, memory)}")
    print(out)


def get_parallel_stage_input(previous_stage, couch):
    """
    Get the input for the current stage based on the output of the previous stage.
    The return function should be changed based on the DAG.
    """
    # return couch.get_doc_with_name(ACTIVATIONS_DB_NAME, previous_stage)["response"]["result"]["params"]
    return couch.get_doc_with_name(ACTIVATIONS_DB_NAME, previous_stage)["response"]["result"]

def feature_encoder(input_features):
    """
    Encodes input features {'length_of_message', 'num_of_iterations', 'num_of_processes'}
    into a single numeric value by scaling each feature to a range between 0 and 1 based on predefined
    maximum values, then combining these scaled values with specific multipliers to ensure
    uniqueness and reversibility of the encoding.

    Args:
    - input_features (dict): A dictionary containing the input features to be encoded. Expected keys are
      'length_of_message', 'num_of_iterations', 'num_of_processes', with their respective numeric values.

    Returns:
    - float: A single float value representing the encoded input features.
    """

    max_length_of_message = 1000
    max_num_of_iterations = 10000
    max_num_of_processes = 8
    
    length_of_message = input_features['length_of_message']
    num_of_iterations = input_features['num_of_iterations']
    num_of_processes = input_features['num_of_processes']
    
    scaled_length = (length_of_message - 50) / (max_length_of_message - 50)
    scaled_iterations = (num_of_iterations - 500) / (max_num_of_iterations - 500)
    scaled_processes = (num_of_processes - 1) / (max_num_of_processes - 1)
 
    encoded_value = scaled_length * 0.0001 + scaled_iterations * 0.001 + scaled_processes * 0.01
    
    return encoded_value

def VA_ImgRec_input_extractor(input_features,idx):
    """
    Extracts the input features from the function input for the Image Recognition function.
    Args:
    - input_features (dict): A dictionary containing the input features to be encoded.

    Returns:
    - int: for now the function returns the number of images for this function
    """
    if idx == None:
        return None

    return len(input_features['images'][idx])

def VA_input_extractor(input_features):
    """
    Extracts the input features from the input JSON for the Video Analytics DAG.
    Args:
    - input_features (dict): A dictionary containing the input features to be encoded. Expected keys are
      'video' which is a text and "num_frames" which is an integer.

    Returns:
    - int: for now the function returns the number of frames in the video
    """
    
    return input_features["num_frames"]

def encode_dag_input(data):
    """
    Encodes the whole input of the DAG based on the contents of the 'data' key in the JSON file.

    Args:
    - data (dict): A dictionary representing the 'data' key from the DAG's input JSON.

    Returns:
    - float: A single float value representing the encoded DAG input.
    """
    encoded_values = []
    for _, function_input in data.items():
        length_of_message = function_input['length_of_message']
        num_of_iterations = function_input['num_of_iterations']
        num_of_processes = function_input['num_of_processes']
        
        encoded_value = feature_encoder({
            'length_of_message': length_of_message,
            'num_of_iterations': num_of_iterations,
            'num_of_processes': num_of_processes
        })
        encoded_values.append(encoded_value)
    
    # Combine encoded values for all functions in 'data'
    combined_encoded_value = sum(encoded_values) / len(encoded_values)
    return combined_encoded_value


def feature_decoder(encoded_value):
    """
    Decodes the encoded value back into the original input features by reversing the encoding
    process, which involves extracting each scaled feature value from the combined encoded value
    and rescaling them to their original ranges.

    Args:
    - encoded_value (float): The encoded value representing the scaled and combined input features.

    Returns:
    - dict: A dictionary of the decoded input features, with keys 'length_of_message', 'num_of_iterations',
      and 'num_of_processes' mapping to their respective original numeric values.
    """
    max_length_of_message = 1000
    max_num_of_iterations = 10000
    max_num_of_processes = 8

    # Extracting the scaled values from the encoded value
    scaled_processes = int((encoded_value / 0.01) % 100)
    scaled_iterations = int((encoded_value / 0.001) % 100)
    scaled_length = int((encoded_value / 0.0001) % 100)

    # Rescaling to original ranges
    length_of_message = scaled_length * (max_length_of_message - 50) / 99 + 50
    num_of_iterations = scaled_iterations * (max_num_of_iterations - 500) / 99 + 500
    num_of_processes = scaled_processes * (max_num_of_processes - 1) / 99 + 1

    return {
        'length_of_message': round(length_of_message),
        'num_of_iterations': round(num_of_iterations),
        'num_of_processes': round(num_of_processes)
    }




if __name__ == "__main__":
    dag_ID = "vid"
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # update_Function("AES1", 2,8)
    # update_Function("AES2", 2,8)
    # update_Function("AES3", 2,8)
  
    # inputs_dir = "./newinput"
    
    # # Iterate through each input file in the inputs directory
    # interations = 0
    # for input_file in os.listdir(inputs_dir):
    #     if input_file.endswith(".json"):
    #         input_path = os.path.join(inputs_dir, input_file)
            
    #         # Update function configurations if necessary
    #         # Example: update_Function("AES1", 2, 1)
            
    #         # Invoke the DAG with the input and capture its profiling data
    #         result = invokeDAG(dag_ID, input_path)

    #         print(f"intraction {interations} completed.")

    #         # You might want to log the result or ID from the invocation
    #         print(f"Invocation for {input_file} completed. Result: {result}")
    #         interations += 1

    
    input = "./inputs/vid.json"
    # update_Function("streaming", 1, 1)
    # update_Function("decoder", 5, 1)
    # update_Function("recognition1", 20, 3)
    # update_Function("recognition2", 20, 6)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("streaming", 1, 1)
    # update_Function("decoder", 5, 1)
    # update_Function("recognition1", 20, 4)
    # update_Function("recognition2", 20, 8)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("streaming", 1, 1)
    # update_Function("decoder", 5, 1)
    # update_Function("recognition1", 20, 6)
    # update_Function("recognition2", 20, 6)
    # result = invokeDAG(dag_ID, input, current_time)


    # update_Function("streaming", 1, 1)
    # update_Function("decoder", 5, 1)
    # update_Function("recognition1", 20, 4)
    # update_Function("recognition2", 20, 4)
    # result = invokeDAG(dag_ID, input, current_time)

    update_Function("streaming", 1, 1)
    update_Function("decoder", 5, 1)
    update_Function("recognition1", 20, 2)
    update_Function("recognition2", 20, 2)
    result = invokeDAG(dag_ID, input, current_time)

    update_Function("streaming", 1, 1)
    update_Function("decoder", 5, 1)
    update_Function("recognition1", 20, 2)
    update_Function("recognition2", 20, 2)
    result = invokeDAG(dag_ID, input, current_time)

    update_Function("streaming", 1, 1)
    update_Function("decoder", 5, 1)
    update_Function("recognition1", 20, 2)
    update_Function("recognition2", 20, 2)
    result = invokeDAG(dag_ID, input, current_time)


    # time.sleep(60)
    # dag_ID = "AS"
    # input = "./inputs/AS.json"
    # update_Function("wait1", 1, 1)
    # update_Function("AES1", 1, 6)
    # update_Function("AES2", 1, 6)
    # update_Function("AES3", 1, 6)
    # update_Function("Stats", 1, 1)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("wait1", 1, 1)
    # update_Function("AES1", 1, 2)
    # update_Function("AES2", 1, 2)
    # update_Function("AES3", 1, 2)
    # update_Function("Stats", 1, 1)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("wait1", 1, 1)
    # update_Function("AES1", 1, 2)
    # update_Function("AES2", 1, 4)
    # update_Function("AES3", 1, 6)
    # update_Function("Stats", 1, 1)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("wait1", 1, 1)
    # update_Function("AES1", 1, 3)
    # update_Function("AES2", 1, 3)
    # update_Function("AES3", 1, 3)
    # update_Function("Stats", 1, 1)
    # result = invokeDAG(dag_ID, input, current_time)

    # update_Function("wait1", 1, 1)
    # update_Function("AES1", 1, 5)
    # update_Function("AES2", 1, 5)
    # update_Function("AES3", 1, 5)
    # update_Function("Stats", 1, 1)
    # result = invokeDAG(dag_ID, input, current_time)




    # dag_ID = "ml"
    # input = "./inputs/ml.json"
    # # result = invokeDAG(dag_ID, input)
    # # update_Function("pca", 4,2)
    # update_Function("paramtune1", 4,8)
    # update_Function("paramtune2", 4,8)
    # update_Function("paramtune3", 4,8)
    # update_Function("paramtune4", 4,8)
    # result = invokeDAG(dag_ID, input)
    # update_Function("paramtune1", 4,1)
    # update_Function("paramtune2", 4,2)
    # update_Function("paramtune3", 4,2)
    # update_Function("paramtune4", 4,4)
    # result = invokeDAG(dag_ID, input)
    # update_Function("combine", 4,2)
    # print(result)

    # #convert result to json format
    # # result = json.dumps(result)
    # print(result["_id"])
