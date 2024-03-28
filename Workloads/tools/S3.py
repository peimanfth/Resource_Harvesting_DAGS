import boto3
import os
from utils import setEnvVar

def checkBucketExists(bucket_name):
    setEnvVar('AWS_ACCESS_KEY_ID', 'AKIA26EO4UIX52Y2OUVZ')
    setEnvVar('AWS_SECRET_ACCESS_KEY','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' )

    s3 = boto3.resource('s3')
    return s3.Bucket(bucket_name) in s3.buckets.all()

def uploadVideoToS3(bucket_name, video_path):
    s3 = boto3.client('s3')
    s3.upload_file(video_path, bucket_name, 'car.mp4')
def checkBucketContents(bucket_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        print(obj.key, obj.last_modified)


if __name__ == "__main__":
    bucket_name = 'video-bench'
    print(checkBucketExists(bucket_name))
    video_path = '/home/peiman/Downloads/video.mp4'
    uploadVideoToS3(bucket_name, video_path)
    checkBucketContents(bucket_name)
