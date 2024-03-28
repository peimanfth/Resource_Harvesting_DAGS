import lightgbm as lgb
# import boto3
import couchdb
import numpy as np
from boto3.s3.transfer import TransferConfig
import json
import random
import time
import os
from multiprocessing import Process, Manager
from params import COUCHDB_URL, COUCHDB_PASSWORD, COUCHDB_USERNAME, ML_DB_NAME
from threading import Thread, Event
from queue import Queue
from Monitor import monitor_peak

interval = 0.02

def handler(event, context=None):

    #monitor daemon part 1

    stop_signal = Event()
    q_cpu = Queue()
    q_mem = Queue()
    tmon = Thread(
        target=monitor_peak,
        args=(interval, q_cpu, q_mem, stop_signal),
        daemon=True
    )
    tmon.start()

    index = event['index']
    request_id = event['request_id']
    event = event['data'][event['index']]
    #context = "0,15"
    #os.system("taskset -p --cpu-list " + context  + " %d" % os.getpid())
    start_time = int(round(time.time() * 1000))

    # s3_client = boto3.client(
    # 's3',
    # aws_access_key_id=accessKeyId,
    # aws_secret_access_key=accessKey
    # )
    # bucket_name = bucketName
    # config = TransferConfig(use_threads=False)

    couch = couchdb.Server(COUCHDB_URL)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    if ML_DB_NAME in couch:
        db = couch[ML_DB_NAME]
    else:
        db = couch.create(ML_DB_NAME)


    filename = "/tmp/Digits_Train_Transform.txt"
    f = open(filename, "wb")
    start_download = int(round(time.time() * 1000))
    for i in range(1):
        # s3_client.download_fileobj(bucket_name, "ML_Pipeline/train_pca_transform_2.txt" , f, Config=config)
        #get the file from couchdb
        doc = db[request_id]
        result = db.get_attachment(doc['_id'], 'train_pca_transform.txt')
        f.write(result.read())
    end_download = int(round(time.time() * 1000))

    start_process = int(round(time.time() * 1000))
    f.close()
    end_time = int(round(time.time() * 1000))
    train_data = np.genfromtxt('/tmp/Digits_Train_Transform.txt', delimiter='\t')

    print("train data shape")
    print(train_data.shape)


    y_train = train_data[0:5000,0]
    X_train = train_data[0:5000,1:train_data.shape[1]]
    #lgb_train = lgb.Dataset(X_train, y_train)

    print(type(y_train))
    #print(type(lgb_train))
    manager = Manager() 
    return_dict = manager.dict()  
    process_dict = manager.dict()
    upload_dict = manager.dict()
    num_of_trees=event['num_of_trees']
    depthes=event['max_depth'] 
    feature_fractions=event['feature_fraction']
    # for runs in range(len(num_of_trees)): 
    #     # Use multiple processes to train trees in parallel
    #     threads=event['threads'][str(index)]
    #     print("tree depth: " + str(depthes[runs]) + " feature fraction: " + str(feature_fractions[runs]) + " num of trees: " + str(num_of_trees[runs]) + " threads: " + str(threads))
    #     ths = []
    #     for t in range(threads):
    #         ths.append(Process(target=train_tree, args=(t, runs, index, X_train, y_train, event, num_of_trees[runs], depthes[runs], feature_fractions[runs], COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD, str(request_id), return_dict, process_dict, upload_dict)))
    #     for t in range(threads):
    #         ths[t].start()
    #     for t in range(threads):
    #         ths[t].join()
    threads = event['threads']
    iterations = len(num_of_trees) / threads
    for runs in range(int(iterations)):
    # Use multiple processes to train trees in parallel
        
        # Initialize a list to hold the Process objects
        processes = []
        for t in range(threads):
            # Create a Process for each thread required for the current 'runs' iteration
            print(f" num of trees: {num_of_trees[int(runs*threads + t)]} tree depth: {depthes[int(runs*threads + t)]} feature fraction: {feature_fractions[int(runs*threads + t)]} thread: {t}")
            process = Process(target=train_tree, args=(t, runs, index, X_train, y_train, event, num_of_trees[int(runs*threads + t)], depthes[int(runs*threads + t)], feature_fractions[int(runs*threads + t)], COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD, str(request_id), return_dict, process_dict, upload_dict))
            processes.append(process)
            
        # Start all processes
        for process in processes:
            process.start()
            
        # Wait for all processes to complete
        for process in processes:
            process.join()

        # There is a possibility that the number of trees is not a multiple of the number of threads
        # In such a case, the last iteration will have fewer threads than the rest
        # This will cause an index out of range error
        # To avoid this, we need to check if the current iteration is the last one
        # If it is, we need to break out of the loop
        if (runs + 1) * threads >= len(num_of_trees):
            break        
    end_process = int(round(time.time() * 1000))		 
    print("download duration: " + str(end_time-start_time))
    end_time = int(round(time.time() * 1000))
    e2e=end_time-start_time
    print("E2E duration: " + str(e2e))
    # j= {
    #     'statusCode': 200,
    #     'body': json.dumps('Done Training Threads = ' + str(threads)),
    #     'key1': event['key1'],
    #     'duration': e2e,		
    #     'trees_max_depthes': return_dict.keys(),		
    #     'accuracies': return_dict.values(),
    #     'process_times': process_dict.values(),
    #     'upload_times': upload_dict.values(),
    #     'download_times': (end_download - start_download),
    #     'PCA_Download':  event['PCA_Download'],
    #     'PCA_Process': event['PCA_Process'],
    #     'PCA_Upload': event['PCA_Upload']
    # }
    # print(j)

    stop_signal.set()  # Signal the monitor thread to stop
    tmon.join()

    # montor daemon part 2
    cpu_timestamp = []
    cpu_usage = []
    while q_cpu.empty() is False:
        (timestamp, cpu) = q_cpu.get()
        cpu_timestamp.append(timestamp)
        cpu_usage.append(cpu)

    mem_timestamp = []
    mem_usage = []
    while q_mem.empty() is False:
        (timestamp, mem) = q_mem.get()
        mem_timestamp.append(timestamp)
        mem_usage.append(mem)


    return {
        'statusCode': 200,
        'body': json.dumps('Done Training Threads = ' + str(threads)),
        'key1': event['key1'],
        'duration': e2e,		
        'trees_max_depthes': return_dict.keys(),		
        'accuracies': return_dict.values(),
        'process_times': process_dict.values(),
        'upload_times': upload_dict.values(),
        'download_times': (end_download - start_download),
        'PCA_Download':  event['PCA_Download'],
        'PCA_Process': event['PCA_Process'],
        'PCA_Upload': event['PCA_Upload'],
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

    # return {
    #     'statusCode': 200,
    #     'body': json.dumps('Done Training Threads = ' + str(threads)),
    #     'key1': event['key1'],
    #     'duration': e2e,		
    #     'trees_max_depthes': return_dict.keys(),		
    #     'accuracies': return_dict.values(),
    #     'process_times': process_dict.values(),
    #     'upload_times': upload_dict.values(),
    #     'download_times': (end_download - start_download),
    #     'PCA_Download':  event['PCA_Download'],
    #     'PCA_Process': event['PCA_Process'],
    #     'PCA_Upload': event['PCA_Upload']
    # }

def train_tree(t_index, run_index, f_index, X_train, y_train, event, num_of_trees, max_depth, feature_fraction, couch_url, couch_user, couch_pass, request_id, return_dict, process_dict, upload_dict): 
    lgb_train = lgb.Dataset(X_train, y_train)
    _id=str(event['mod_index']) + "_" + str(t_index) + "_" + str(run_index)
    # _id=str(f_index) + "_" + str(t_index)
    chance = 0.8  #round(random.random()/2 + 0.5,1)
    params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_classes' : 10,
    'metric': {'multi_logloss'},
    'num_leaves': 50,
    'learning_rate': 0.05,
    'feature_fraction': feature_fraction,
    'bagging_fraction': chance, # If model indexes are 1->20, this makes feature_fraction: 0.7->0.9
    'bagging_freq': 5,
    'max_depth': max_depth,
    'verbose': -1,
    'num_threads': 1
    }
    print('Starting training...')
    start_process = int(round(time.time() * 1000))
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_of_trees, # number of trees
                valid_sets=lgb_train,
                callbacks=[lgb.early_stopping(5)])

    y_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    count_match=0
    for i in range(len(y_pred)):
        #print(y_pred[i])
        #print(y_train[i])
        result = np.where(y_pred[i] == np.amax(y_pred[i]))[0]
        #print(result)
        #print(y_train[i])
        if result == y_train[i]:
            count_match +=1
    acc = count_match/len(y_pred)
    end_process = int(round(time.time() * 1000))


    model_name="lightGBM_model_" + str(_id) + ".txt"
    gbm.save_model("/tmp/" + model_name)
    print("Ready to uploaded " + model_name )
    start_upload = int(round(time.time() * 1000))
    couch = couchdb.Server(couch_url)
    couch.resource.credentials = (couch_user, couch_pass)
    if ML_DB_NAME in couch:
        db = couch[ML_DB_NAME]
    else:
        db = couch.create(ML_DB_NAME)
    # s3_client.upload_file("/tmp/" + model_name, bucket_name, "ML_Pipeline/"+model_name, Config=config)
    #upload the model to couchdb
    doc = {'_id': f'{request_id}_func_{f_index}_thread_{t_index}_run_{run_index}'}
    db.save(doc)
    with open("/tmp/" + model_name, 'rb') as f:
        db.put_attachment(doc, f, filename=model_name, content_type='application/octet-stream')

    end_upload = int(round(time.time() * 1000))

    print("model uploaded " + model_name )

    #end_time = int(round(time.time() * 1000))
    #print("duration: " + str(end_time-start_time))
    #subfilename = "Train_" + event['key1'] + "_id_" + str(_id) + "_start_" + str(start_time) + "_end_"+ str(end_time)
    #filename = "/tmp/" + subfilename
    #f = open(filename, "w")
    #f.write(filename)
    #f.close()
    #s3_client.upload_file(filename, bucket_name, "LightGBM_Times/" + subfilename, Config=config)

    return_dict[str(_id) + "_" + str(max_depth) + "_" + str(feature_fraction)] = acc
    process_dict[str(_id) + "_" + str(max_depth) + "_" + str(feature_fraction)] = (end_process - start_process)
    upload_dict[str(_id) + "_" + str(max_depth) + "_" + str(feature_fraction)] = (end_upload - start_upload)
    return {
        'statusCode': 200,
        'body': json.dumps('Done Training With Accuracy = ' + str(acc)),
        '_id': _id,
        'key1': event['key1']
    }
 

# if __name__ == "__main__":
#     ev = {'data': [{'mod_index': 0, 'PCA_Download': 60, 'PCA_Process': 2530, 'PCA_Upload': 709, 'key1': 'inv_300', 'num_of_trees': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 'max_depth': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40], 'feature_fraction': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.95], 'threads': 2}, {'mod_index': 1, 'PCA_Download': 60, 'PCA_Process': 2530, 'PCA_Upload': 709, 'key1': 'inv_300', 'num_of_trees': [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25], 'max_depth': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40], 'feature_fraction': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.95], 'threads': 4}, {'mod_index': 2, 'PCA_Download': 60, 'PCA_Process': 2530, 'PCA_Upload': 709, 'key1': 'inv_300', 'num_of_trees': [125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125], 'max_depth': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40], 'feature_fraction': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.95], 'threads': 6}, {'mod_index': 3, 'PCA_Download': 60, 'PCA_Process': 2530, 'PCA_Upload': 709, 'key1': 'inv_300', 'num_of_trees': [625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625, 625], 'max_depth': [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40], 'feature_fraction': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.95], 'threads': 8}], 'request_id': 'efe8de79-47fd-47fb-8e68-3f6ce6a7b21b',
#           "index": 0
#     }
#     # handler({'mod_index': 1, 'PCA_Download': 3551, 'PCA_Process': 2487, 'PCA_Upload': 23488, 'key1': 'inv_300', 'num_of_trees': [5, 10, 15, 20], 'max_depth': [10, 10, 10, 10], 'feature_fraction': [0.25, 0.25, 0.25, 0.25], 'threads': 1})
#     print(handler(ev))