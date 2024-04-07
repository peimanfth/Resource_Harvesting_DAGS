import boto3
import os
import uuid

# def setEnvVar(envVar, value):
#     if envVar in os.environ:
#         print("Warning: environment variable %s is already set to %s. Overwriting it with %s" % (envVar, os.environ[envVar], value))
#     os.environ[envVar] = value


# def downloadFileFromS3(bucket_name, file_name):
#     setEnvVar('AWS_ACCESS_KEY_ID', 'AKIA26EO4UIX52Y2OUVZ')
#     setEnvVar('AWS_SECRET_ACCESS_KEY','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' )
#     s3 = boto3.client('s3')
#     s3.download_file(bucket_name, file_name, '/tmp/' + file_name)
#     return '/tmp/' + file_name

def handler(event, context=None):
    request_id = str(uuid.uuid4())
    if event['video'] == 'car':
        file_name = 'car'
        doc_name = 'videos'
    if event['num_frames'] is not None:
        num_frames = event['num_frames']
    return {
        'video_name': file_name,
        'num_frames': num_frames,
        'db_name': 'video-bench',
        'doc_name': doc_name,
        'request_ids': [f'{request_id}-recog1', f'{request_id}-recog2'],

    }

# if __name__ == "__main__":
#     event = {
#         "video": "car",
#         "num_frames": 2
#     }
#     print(handler(event))