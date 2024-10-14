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
