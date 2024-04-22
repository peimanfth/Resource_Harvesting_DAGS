MEMORY_CAP_PER_FUNCTION = 32
CPU_CAP_PER_FUNCTION = 8
MEM_UNIT = 128
CPU_UNIT = 1
# MAX_MEM = 32
MAX_ALLOCATION = MEMORY_CAP_PER_FUNCTION * CPU_CAP_PER_FUNCTION
USER_CONFIG = {
    "AS": {
        "Functions": {
            "wait1": {
                "Memory": 128,
                "CPU": 1,
                "stage" : "pre"
            },
            "AES1":{
                "Memory": 128,
                "CPU": 1,
                "stage" : "parallel"
            },
            "AES2":{
                "Memory": 128,
                "CPU": 1,
                "stage" : "parallel"
            },
            "AES3":{
                "Memory": 128,
                "CPU": 1,
                "stage" : "parallel"
            },
            "Stats": {
                "Memory": 128,
                "CPU": 1,
                "stage" : None
            }
        },
        "Memory": 64,
        "CPU": 1,
        "Params" : "./inputs/AS.json"
    },
    "vid": {
        "Functions": {
            "streaming": {
                "Memory": 128,
                "CPU": 1,
                "stage" : None,
                "idx": None
            },
            "decoder": {
                "Memory": 1024,
                "CPU": 1,
                "stage" : "pre",
                "idx": None
            },
            "recognition1": {
                "Memory": 1024,
                "CPU": 2,
                "stage" : "parallel",
                "idx": 0
            },
            "recognition2": {
                "Memory": 1024,
                "CPU": 4,
                "stage" : "parallel",
                "idx": 1
            }
        },
        "Memory": 64,
        "CPU": 1,
        "Params" : "./inputs/VA.json"
    },
    "MR": {
        "Functions": {
            "map": {
                "Memory": 1024,
                "CPU": 1,
                "Params" : None
            },
            "reduce": {
                "Memory": 1024,
                "CPU": 1,
                "Params" : None
            },
            "combine": {
                "Memory": 1024,
                "CPU": 1,
                "Params" : None
            }
        },
        "Memory": 64,
        "CPU": 1,
        "Params" : "./inputs/MR.json"
    },
    "ml": {
        "Functions": {
            "pca": {
                "Memory": 1024,
                "CPU": 1,
                "Params" : None
            },
            "paramtune1":{
                "Memory": 512,
                "CPU": 2,
                "Params" : None
            },
            "paramtune2":{
                "Memory": 512,
                "CPU": 2,
                "Params" : None
            },
            "paramtune3":{
                "Memory": 512,
                "CPU": 2,
                "Params" : None
            },
            "paramtune4": {
                "Memory": 512,
                "CPU": 2,
                "Params" : None
            },
            "combine": {
                "Memory": 512,
                "CPU": 2,
                "Params" : None
            }
        },
        "Memory": 64,
        "CPU": 1,
        "Params" : "./inputs/ml.json"
    }


}
WSK_CLI = "wsk -i"
WSK_ACTION = "action"
AWS_ACCESS_KEY_ID = 'AKIA26EO4UIX52Y2OUVZ'
AWS_SECRET_ACCESS_KEY = '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'
COUCHDB_URL = 'http://172.17.0.1:5984/'
VA_DB_NAME = 'video-bench'
MR_DB_NAME = 'mr-wikipedia'
ML_DB_NAME = 'ml-pipeline'
ACTIVATIONS_DB_NAME = 'whisk_local_activations'
COUCHDB_PASSWORD = 'some_passw0rd'
COUCHDB_USERNAME = 'whisk_admin'
MODELS_DIR = 'models/remote'


