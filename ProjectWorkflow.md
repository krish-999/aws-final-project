# Image Classification Workflow with AWS SageMaker and Step Functions

This project implements an end-to-end image classification solution using **AWS SageMaker** for model inference, **Lambda functions** for automation, and **Step Functions** to orchestrate the entire workflow. The solution processes images, classifies them using a pre-trained model, and filters low-confidence predictions before passing them to downstream systems.

## Project Overview

The project builds a scalable and automated pipeline that integrates various AWS services to classify images stored in S3. The workflow consists of multiple steps:
1. **Image Serialization Lambda**: Encodes image data to base64.
2. **Inference Lambda**: Uses AWS SageMaker to classify the image.
3. **Threshold Filtering Lambda**: Filters out low-confidence predictions to ensure only valid inferences are passed to downstream systems.

### Key Features
- **Step Functions Orchestration**: The workflow is managed using AWS Step Functions, which allows easy visualization and tracking of the entire pipeline.
- **AWS SageMaker Integration**: The model deployed on SageMaker predicts image labels, which are then used to filter out predictions based on a set confidence threshold.
- **Lambda Functions**: Three Lambda functions are used to serialize images, make predictions, and handle filtering based on confidence scores.
- **Model Monitoring**: SageMaker Model Monitor captures and stores inference data in S3 for future analysis.

## Step Functions Workflow

The workflow is constructed with the following steps:

1. **Image Serialization Lambda**: 
    - This function reads an image from an S3 bucket, serializes it, and passes it to the next step in the Step Function.

    - [Image Serializer Lambda Function](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/serializeLambda.py)



2. **Image Classification Lambda**: 
    - This function invokes the SageMaker model deployed at an endpoint to classify the image. The predictions are returned and passed to the next step.

    - [Classifier Lambda Function](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/classifierLambda.py)



3. **Threshold Filter Lambda**: 
    - Filters out predictions that don’t meet the confidence threshold of 73%. If the highest confidence prediction is below the threshold, the workflow throws an error. If the threshold is met, the inference is passed along.
  
    - [Threshold Filter Lambda Function](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/thresholdLambda.py)



### AWS Lambda Implementation

1. **Image Serializer Lambda**:
    ```python
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
    ```

2. **Classifier Lambda**:
    ```python
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
    ```

3. **Threshold Filter Lambda**:
    ```python
    import json
    
    THRESHOLD = 0.73
    
    def lambda_handler(event, context):
        
        # Grab the inferences from the event
        inferences = event["body"].get("inferences", [])
        
        # Check if any values in our inferences are above THRESHOLD
        meets_threshold = any(prob >= THRESHOLD for prob in inferences)
    
        # If our threshold is met, pass our data back out of the
        # Step Function, else, end the Step Function with an error
        if meets_threshold:
            return {
                'statusCode': 200,
                'body': event["body"]
            }
        else:
            raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    ```



## Visualization of Model Performance

After running multiple Step Function executions, the captured inference data from the SageMaker Model Monitor can be visualized. This helps track the confidence levels of the predictions over time and monitor system performance.
**Step Functions**: For orchestrating the entire ML workflow, from inference to monitoring.

1. **Step Functions Graph Architecture**
![Step Functions Graph Architecture](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/stepfunctions_graph.png)



2. **Showing a Step Function execution that successfully passes the threshold**
![Showing a Step Function execution that successfully passes the threshold](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/state_definition_that_succeeds.png)



3. **Showing a Step Function execution that fails to pass the threshold**
![Showing a Step Function execution that fails to pass the threshold](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/state_definition_that_shows_a_fail.png)



### Unique Visualization Example
We processed the inference data from SageMaker Model Monitor, grouping them into time bins and visualizing the mean confidence per bin. The resulting graph helps assess whether the model is consistently meeting the confidence threshold over time.

```python
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines

# Load and parse captured data
json_data = []
for jsonl in file_handles:
    with jsonlines.open(jsonl) as f:
        for dict_line in f.iter():
            json_data.append(dict_line)

# Define how to extract inferences and timestamps
def simple_getter(obj):
    inferences = obj["captureData"]["endpointOutput"]["data"]
    timestamp = obj["eventMetadata"]["inferenceTime"]
    return json.loads(inferences), timestamp

# Extract confidence values and timestamps
data_points = [(obj["eventMetadata"]["inferenceTime"], max(simple_getter(obj)[0])) for obj in json_data]

# Convert to DataFrame for visualization
df = pd.DataFrame(data_points, columns=["timestamp", "confidence"])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='T')).confidence.mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_grouped.index, df_grouped.values, label='Mean Confidence per Time Bin')
plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
plt.title('Model Confidence Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Mean Confidence')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("model_performance.png")
```

## Stretch Goals Achieved
- **Lambda Layers for Dependency Management**: To optimize the Lambda functions, dependencies like `sagemaker` were packaged into Lambda Layers.
- **Detailed Visualization**: We grouped inference data into time bins and visualized mean confidence to monitor the model’s performance.
- **Model Monitor Integration**: By capturing inferences with SageMaker Model Monitor, we were able to visualize and analyze the model's performance over time.

## Next Steps
- **Monitoring Dashboard**: Expand on the current visualization to include real-time monitoring of model confidence levels and alerting when the threshold is not met.
- **Automated Retraining**: Implement automated model retraining if the confidence scores consistently fall below a certain threshold.

## Conclusion
This project demonstrates how AWS services like Lambda, SageMaker, and Step Functions can be integrated to build an automated, scalable image classification pipeline. The pipeline filters low-confidence inferences, ensuring that only high-confidence predictions are passed downstream for further action. The system's performance is continuously monitored using SageMaker Model Monitor, and the data can be visualized for further analysis.
