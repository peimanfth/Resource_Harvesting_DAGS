from utils import decode_action, encode_action, decodeCpu, decodeMemory, run_cmd, run_shell_command
from params import *
from Couch import Couch
from pilotRun import update_Function


# for i in range(1, 8*32):
#     print(i, decode_action(i), encode_action(decode_action(i)['cpu'], decode_action(i)['memory']))
#     print(decodeCpu(i), decodeMemory(i))

# run_cmd("../ansible/wipe.sh")
# print("Wiped CouchDB")

# dag_ID = "hi"
# # input = "./inputs/AS.json"
# output, errors = run_shell_command(f"{WSK_CLI} {WSK_ACTION} invoke {dag_ID} | awk '{{print $6}}'")
# print("Output:", output)
# if errors:
#     print("Errors:", errors)
# print(len([33, 85, 58, 54, 60678, 60656, 64, 82, 66, 70, 50, 54, 56, 52, 74, 72, 47, 54, 46, 44, 49, 42, 90, 91, 76, 79, 77, 57, 51, 46]))

# couch = Couch(COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD)
# docs_to_keep = ["train_digits", "3196c40c-91e2-4986-b4da-2ab0d5d601d5"]
# couch.delete_docs_except_provided(ML_DB_NAME, docs_to_keep)





# from multiprocessing import Process
# from itertools import product

# # Assuming num_of_trees is consistent across scenarios and threads indicates the number of parallel processes

# # Generate all combinations of depths and feature_fractions


# depths = [2,4, 6, 8]
# feature_fractions = [0.5, 0.6, 0.7, 0.8]
# threads = 2
# num_of_trees = 10
# all_combinations = list(product(depths, feature_fractions))

# # Function to split scenarios into chunks for processing
# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# # Distribute combinations across available threads
# scenarios_per_thread = list(chunks(all_combinations, len(all_combinations) // threads + (len(all_combinations) % threads > 0)))

# for i, scenarios in enumerate(scenarios_per_thread):
#     processes = []
#     print(f"Batch {i+1} with {len(scenarios)} scenarios.")
#     for scenario in scenarios:
#         depth, feature_fraction = scenario
#         # proc = Process(target=train_tree, args=(X_train, y_train, event, num_of_trees, depth, feature_fraction, COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD, str(request_id), return_dict, process_dict, upload_dict))
#         # processes.append(proc)
#         # proc.start()
#         print(f"Training tree with depth {depth} and feature fraction {feature_fraction} and number of trees{num_of_trees}.")
#     print("Done iteration number", i)


# print("Completed all scenarios.")

# print(decodeMemory(42), decodeCpu(42))
# couch = Couch(COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD)
# print(couch.get_doc_with_name(ACTIVATIONS_DB_NAME, "wait1"))


# def normalize_input(input_data):
#     # Extract all values for each feature across all functions
#     lengths = [details['length_of_message'] for details in input_data.values()]
#     iterations = [details['num_of_iterations'] for details in input_data.values()]
#     processes = [details['num_of_processes'] for details in input_data.values()]

#     # Compute min and max for each feature
#     min_length, max_length = min(lengths), max(lengths)
#     min_iterations, max_iterations = min(iterations), max(iterations)
#     min_processes, max_processes = min(processes), max(processes)

#     # Normalize each feature of each function
#     normalized_data = {}
#     for function, details in input_data.items():
#         normalized_data[function] = {
#             'length_of_message': (details['length_of_message'] - min_length) / (max_length - min_length),
#             'num_of_iterations': (details['num_of_iterations'] - min_iterations) / (max_iterations - min_iterations),
#             'num_of_processes': (details['num_of_processes'] - min_processes) / (max_processes - min_processes),
#         }

#     return normalized_data

# input_data = {
#     'AES1': {'length_of_message': 50, 'num_of_iterations': 5000, 'num_of_processes': 1},
#     'AES2': {'length_of_message': 50, 'num_of_iterations': 20000, 'num_of_processes': 2},
#     'AES3': {'length_of_message': 50, 'num_of_iterations': 100000, 'num_of_processes': 4}
# }

# print(normalize_input(input_data))
# from invoker import feature_encoder
# input = {'length_of_message': 50, 'num_of_iterations': 5000, 'num_of_processes': 1}
# print(feature_encoder(input))
# from invoker import update_Function

# update_Function("AES2", 2,2)

# print(encode_action(2, 40))
# update_Function("streaming", 2,1)
# update_Function("decoder", 2,5)
# update_Function("recognition1", 20,3)
# update_Function("recognition2", 6,3)
# print(encode_action(3, 1))

# run_shell_command("wsk -i action invoke hi")

# run_shell_command("wsk -i action invoke bye")
# run_shell_command("wsk -i action invoke hi")
# run_shell_command("wsk -i action invoke bye")
# run_shell_command("wsk -i action invoke bye")
# run_shell_command("wsk -i action invoke hi")
# run_shell_command("wsk -i action invoke bye")

# run_shell_command("wsk -i action invoke hi")


# run_shell_command("wsk -i action invoke hi")


# run_shell_command("wsk -i action invoke hi")


# run_shell_command("wsk -i action invoke hi")
update_Function("wait1", 8, 1)
update_Function("AES1", 8, 2)
update_Function("AES2", 8, 2)
update_Function("AES3", 8, 2)
update_Function("Stats", 8, 1)
