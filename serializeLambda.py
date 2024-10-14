import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Log the entire event for debugging
    print("Received event:", json.dumps(event, indent=2))
    
    # Get the s3 address from the Step Function event input
    key = event.get('s3_key')  # Use get to avoid KeyError
    bucket = event.get('s3_bucket')  # Use get to avoid KeyError
    
    # Check if key and bucket are present
    if not key or not bucket:
        print(f"Key: {key}, Bucket: {bucket}")  # Log the values for debugging
        raise ValueError("Missing 's3_key' or 's3_bucket' in event")

    # Download the data from s3 to /tmp/image.png
    try:
        s3.download_file(bucket, key, '/tmp/image.png')  # Download the image to the /tmp directory
    except Exception as e:
        raise RuntimeError(f"Failed to download file from S3: {str(e)}")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')  # Encode and convert bytes to string

    # Pass the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "s3_bucket": bucket,
            "s3_key": key,
            "image_data": image_data,
            "inferences": []
        }
    }
