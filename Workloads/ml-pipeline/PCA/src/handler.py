from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from numpy import genfromtxt
from numpy import concatenate
from numpy import savetxt
import numpy as np

import json
import random
import time
import io
import uuid

import boto3
import couchdb
from boto3.s3.transfer import TransferConfig
from params import COUCHDB_URL, COUCHDB_PASSWORD, COUCHDB_USERNAME, ML_DB_NAME, COUCHDB_DOC
from threading import Thread, Event
from queue import Queue
from Monitor import monitor_peak


interval = 0.02
# s3_client = boto3.client(
#   's3',
#   aws_access_key_id=accessKeyId,
#   aws_secret_access_key=accessKey
#  )
# bucket_name = bucketName
# config = TransferConfig(use_threads=False)
# filename = "/tmp/Digits_Test.txt"
# f = open(filename, "wb")
# s3_client.download_fileobj(bucket_name, "ML_Pipeline/Digits_Test.txt" , f, Config=config)
# f.close()
# print("file downloaded")

def handler(event, context=None):

    # monitor daemon part 1

    stop_signal = Event()
    q_cpu = Queue()
    q_mem = Queue()
    t = Thread(
        target=monitor_peak,
        args=(interval, q_cpu, q_mem, stop_signal),
        daemon=True
    )
    t.start()


    # s3_client = boto3.client(
    # 's3',
    # aws_access_key_id=accessKeyId,
    # aws_secret_access_key=accessKey
    # )
    # bucket_name = bucketName
    # config = TransferConfig(use_threads=False)


    # start_time = int(round(time.time() * 1000))
    event = event.get('data')

    couch = couchdb.Server(COUCHDB_URL)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    if ML_DB_NAME in couch:
        db = couch[ML_DB_NAME]
    else:
        db = couch.create(ML_DB_NAME)

    doc = db[COUCHDB_DOC]

    request_id = str(uuid.uuid4())

    start_download = int(round(time.time() * 1000))

    filename = "/tmp/Digits_Train_Org.txt"
    f = open(filename, "wb")
    # s3_client.download_fileobj(bucket_name, "ML_Pipeline/Digits_Train.txt" , f, Config=config)
    #download the file from couchdb
    response = db.get_attachment(doc['_id'], "digits.txt")
    print(response)
    f.write(response.read())
    f.close()
    end_download = int(round(time.time() * 1000))

    start_process = int(round(time.time() * 1000))
    #filename = "/tmp/Digits_Test.txt"
    #f = open(filename, "wb")
    #s3_client.download_fileobj(bucket_name, "LightGBM_Data_Input/Digits_Test_Small.txt" , f, Config=config)
    #f.close()

    train_data = genfromtxt('/tmp/Digits_Train_Org.txt', delimiter='\t')
    #test_data = genfromtxt('/tmp/Digits_Test.txt', delimiter='\t')

    train_labels = train_data[:,0]
    #test_labels = test_data[:,0]

    A = train_data[:,1:train_data.shape[1]]
    #B = test_data[:,1:test_data.shape[1]]

    # calculate the mean of each column
    MA = mean(A.T, axis=1)
    #MB = mean(B.T, axis=1)

    # center columns by subtracting column means
    CA = A - MA
    #CB = B - MB

    # calculate covariance matrix of centered matrix
    VA = cov(CA.T)

    # eigendecomposition of covariance matrix
    values, vectors = eig(VA)

    # project data
    PA = vectors.T.dot(CA.T)
    #PB = vectors.T.dot(CB.T)

    np.save("/tmp/vectors_pca.txt", vectors)

    #savetxt("/tmp/vectors_pca.txt", vectors, delimiter="\t")
    #vectors.tofile("/tmp/vectors_pca.txt")

    #print("vectors shape:")
    #print(vectors.shape)


    first_n_A = PA.T[:,0:100].real
    #first_n_B = PB.T[:,0:10].real
    train_labels =  train_labels.reshape(train_labels.shape[0],1)
    #test_labels = test_labels.reshape(test_labels.shape[0],1)

    first_n_A_label = concatenate((train_labels, first_n_A), axis=1)
    #first_n_B_label = concatenate((test_labels, first_n_B), axis=1)

    savetxt("/tmp/Digits_Train_Transform.txt", first_n_A_label, delimiter="\t")
    #savetxt("/tmp/Digits_Test_Transform.txt", first_n_B_label, delimiter="\t")

    end_process = int(round(time.time() * 1000))

    start_upload = int(round(time.time() * 1000))
   
    doc = {'_id': request_id}
    db.save(doc)


    # s3_client.upload_file("/tmp/vectors_pca.txt.npy", bucket_name, "ML_Pipeline/vectors_pca.txt", Config=config)
    # s3_client.upload_file("/tmp/Digits_Train_Transform.txt", bucket_name, "ML_Pipeline/train_pca_transform_2.txt", Config=config)

    #save vectors and transformed data to couchdb
    with open("/tmp/vectors_pca.txt.npy", 'rb') as f:
        db.put_attachment(doc, f, filename='vectors_pca.txt.npy', content_type='application/octet-stream')

    with open("/tmp/Digits_Train_Transform.txt", 'rb') as f:
        db.put_attachment(doc, f, filename='train_pca_transform.txt', content_type='application/octet-stream')

    #s3_client.upload_file("/tmp/Digits_Test_Transform.txt", bucket_name, "LightGBM_Data/test_pca_transform.txt", Config=config)

    end_upload = int(round(time.time() * 1000)) 
    end_time = int(round(time.time() * 1000))

    # subfilename = "PCA_" + event['key1'] + "_start_" + str(start_time) + "_end_"+ str(end_time)
    # filename = "/tmp/" + subfilename
    # f = open(filename, "w")
    # f.write(filename)
    # f.close()
    # s3_client.upload_file(filename, bucket_name, "ML_Pipeline/LightGBM_Times/" + subfilename, Config=config)

    # bundle_size= event['bundle_size']
    bundle_size = 4
    results = []
    list_hyper_params=[]
    all_runs = 4*4*4
    size_per_function = all_runs / int(bundle_size)
    # for i in range(bundle_size):
    for num_of_trees in  event['num_of_trees']:
        for feature_fraction in event['feature_fraction']:
            for max_depth in event['max_depth']:
                list_hyper_params.append((num_of_trees, max_depth, feature_fraction))
    #random.shuffle(list_hyper_params)
    # for list in list_all_hyper_params:
    #     print(list)
        
    for item in list_hyper_params:
        print(item)
    print("length of list_hyper_params")
    print(len(list_hyper_params))

    list_all_hyper_params = [list_hyper_params[i:i + int(size_per_function)] for i in range(0, len(list_hyper_params), int(size_per_function))]

    for item in list_all_hyper_params:
        print(item)
    print("length of list_all_hyper_params")
    print(len(list_all_hyper_params))


    # for i, tri in enumerate(list_hyper_params):
    #     if i % bundle_size == 0:
    #         pass
    #     elif i % bundle_size == 1:
    #         pass
    #     elif i % bundle_size == 2:
    #         pass
    #     elif i % bundle_size == 3:
    #         pass

    for i in range(bundle_size):
        max_depth=[]
        feature_fraction = []
        num_of_trees=[]
        for tri in list_all_hyper_params[i]:
            feature_fraction.append(tri[2])
            max_depth.append(tri[1])
            num_of_trees.append(tri[0])
        j={ "mod_index": i, "PCA_Download": (end_download - start_download), "PCA_Process": (end_process - start_process), "PCA_Upload": (end_upload - start_upload) , "key1": "inv_300", "num_of_trees": num_of_trees, "max_depth": max_depth, "feature_fraction": feature_fraction, "threads": event[f'paramtune{i+1}']['num_of_processes']}
        
        results.append(j)




        # num_of_trees=[]
        # max_depth=[]
        # feature_fraction = []
        # num_bundles=0
        # count=0
        # for tri in list_all_hyper_params[i]:
        #     feature_fraction.append(tri[2])
        #     max_depth.append(tri[1])
        #     num_of_trees.append(tri[0])
        # j={ "mod_index": i, "PCA_Download": (end_download - start_download), "PCA_Process": (end_process - start_process), "PCA_Upload": (end_upload - start_upload) , "key1": "inv_300", "num_of_trees": num_of_trees, "max_depth": max_depth, "feature_fraction": feature_fraction, "threads": event['threads']}
        # num_of_trees=[]
        # max_depth=[]
        # feature_fraction = []
        # results.append(j)

# print(results)

    stop_signal.set()  # Signal the monitor thread to stop
    t.join()

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

    # return {
    #     "data": results,
    #     'request_id': request_id,
    # }

    return {
        "data": results,
        'request_id': request_id,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
            }

if __name__ == "__main__":
    print(
        handler({"data": {
    "key1": "300",
    "num_of_trees": [5, 25, 125, 625],
    "max_depth": [10, 20, 30, 40],
    "feature_fraction": [0.25, 0.5, 0.75, 0.95],
    "paramtune1": {
      "num_of_processes": 2
    },
    "paramtune2": {
      "num_of_processes": 4
    },
    "paramtune3": {
      "num_of_processes": 6
    },
    "paramtune4": {
      "num_of_processes": 8
    }
  }})
    )




#     {
#   "$composer": {
#     "openwhisk": {
#       "ignore_certs": "true"
#     },
#     "redis": {
#       "uri": "redis://174.64.28.38"
#     }
#   },
#   "data": {
#     "key1": "300",
#     "bundle_size": 4,
#     "train1": {
#       "num_of_trees": [5, 10, 20, 40],

#       "num_of_processes": 2
#     },
#     "train2": {
#       "num_of_trees": [50, 100, 200, 400],
#       "num_of_processes": 4
#     },
#     "train3": {
#       "num_of_trees": [50, 1000, 2000, 4000],
#       "num_of_processes": 6
#     },
#     "train4": {
#       "num_of_trees": [5000, 10000, 20000, 40000],
#       "num_of_processes": 8
#     }
#   }
# }
    