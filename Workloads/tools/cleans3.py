import boto3

def clean_s3_bucket(bucket_name):
    # Create an S3 client
    s3 = boto3.client('s3')

    # List all objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    objects = response.get('Contents', [])

    # Delete each object in the bucket
    for obj in objects:
        key = obj['Key']
        print(f"Deleting object: {key}")
        s3.delete_object(Bucket=bucket_name, Key=key)

    print(f"All objects deleted from bucket: {bucket_name}")

if __name__ == "__main__":
    # Replace 'your-bucket-name' with your actual S3 bucket name
    bucket_name = 'mr-results'

    # Clean the S3 bucket
    clean_s3_bucket(bucket_name)
