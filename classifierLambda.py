import json
import boto3
import base64

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('runtime.sagemaker')

# Retrieve the SageMaker endpoint name from environment variables
ENDPOINT = "image-classification-endpoint"  # Replace with your actual endpoint name

def lambda_handler(event, context):
    # Validate input
    if "body" not in event or "image_data" not in event["body"]:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Invalid input, 'image_data' not found"})
        }
    
    # Decode the image data
    try:
        image = base64.b64decode(event["body"]["image_data"])
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": f"Failed to decode image data: {str(e)}"})
        }

    # Invoke SageMaker endpoint
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="image/png",  # This is specific to your model
            Body=image
        )
        # Parse the response
        inferences = json.loads(response['Body'].read().decode())
        
        # Add inferences to the event body
        event["body"]["inferences"] = inferences
        return {
            'statusCode': 200,
            'body': event["body"]
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": f"Prediction failed: {str(e)}"})
        }
