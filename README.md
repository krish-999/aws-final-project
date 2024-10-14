# ML-Workflow-On-Amazon-SageMaker

This project demonstrates how to build a **Machine Learning (ML) workflow** on **Amazon SageMaker** to automate the process of training, deploying, and monitoring models for a fictional company called **Scones Unlimited**.

## Project Overview

You have been hired as a **Machine Learning Engineer** by Scones Unlimited, a logistics company focused on scone delivery. Your task is to build and deploy an **image classification model** that can help route delivery vehicles more efficiently by detecting which kind of vehicle delivery drivers have, in order to route them to the correct loading bay and orders. This model will allow Scones Unlimited to optimize their delivery operations by assigning delivery professionals who have a bicycle to nearby orders and giving motorcyclists orders that are farther can help Scones Unlimited optimize their operations.


The project includes:

- Building a scalable, safe image classification model.
- Deploying the model on **AWS SageMaker**.
- Integrating the model with **AWS Lambda** and **Step Functions**.
- Monitoring the deployed model using **SageMaker Model Monitor** to detect drift or degraded performance over time.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Model Deployment](#model-deployment)
- [Model Monitoring](#model-monitoring)
- [Visualization](#visualization)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

## Project Architecture

The architecture of this project includes the following components:

- **Amazon SageMaker Studio**: Used for creating and managing Jupyter notebooks for training and deployment.
- **SageMaker Model Monitor**: To track and evaluate model performance over time and detect data drift.
- **AWS Lambda**: Used to implement custom inference logic and integrate with AWS Step Functions.
- **Step Functions**: For orchestrating the entire ML workflow from inference to monitoring and decision making.
- **S3 Bucket**: For storing training data, test images, and captured inference data.

![Architecture](https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker/blob/main/stepfunctions_graph.png)

## Getting Started

### 1. Clone the repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/Elomunait/ML-Workflow-On-Amazon-SageMaker.git
```

### 2. Install required packages

Ensure that you have the required Python dependencies installed. You can use the `requirements.txt` file to set up the environment:

```bash
pip install -r requirements.txt
```

### 3. Set up AWS Environment

Ensure that you have your **AWS credentials** configured and that the following services are enabled:

- Amazon SageMaker
- AWS Step Functions
- AWS Lambda
- S3 Bucket (for data storage)

You will also need to create an **IAM Role** with appropriate permissions for SageMaker, Lambda, and Step Functions.

## Prerequisites

To replicate this project, you’ll need:

- An **AWS account** with permissions for SageMaker, Lambda, Step Functions, and S3.
- Basic understanding of **Python (3.x)** and **Jupyter Notebooks**.
- **AWS CLI** and **Boto3** configured with credentials.
- Installed Python libraries:
  - `boto3`
  - `sagemaker`
  - `tensorflow` or `PyTorch` (depending on your chosen framework).

## Model Deployment

### 1. Data Preprocessing

Prepare and upload your dataset (e.g., CIFAR-10 for vehicle classification) to an S3 bucket. This data will be used to train the image classification model.

### 2. Model Training

Use Amazon SageMaker to train the image classification model. The model will be trained to differentiate between delivery vehicles, such as bicycles and motorcycles. Use the `train_model.ipynb` notebook to start this process.

### 3. Endpoint Deployment

After training, deploy the model to a **SageMaker Endpoint** to enable real-time inference.

```python
deployment = img_classifier_model.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
    endpoint_name="vehicle-classification-endpoint",
    data_capture_config=data_capture_config
)

endpoint = deployment.endpoint_name
print("Endpoint Name:", endpoint)
```

## Model Monitoring

Once deployed, **SageMaker Model Monitor** is used to capture data from the inference requests. This data is analyzed for performance degradation or drift. If the model's predictions start to deviate, you can take action such as retraining the model.

Key steps include:

- **Data Capture**: Configure the SageMaker endpoint to capture data for monitoring.
- **Data Drift Detection**: Set up Model Monitor to track and detect data drift or anomalies over time.

## Visualization

Visualization is an important step to evaluate the performance of your model. You can build custom visualizations to analyze the results, such as **box plots** and **histograms** to display prediction confidence levels.

Example visualization:

```python
import plotly.express as px
box_fig = px.box(df, y="Confidence", points="all", title="Confidence Level Distribution")
box_fig.show()
```

This allows you to monitor the confidence levels of predictions and spot any trends or anomalies.

## Results

- **Model Confidence**: The model achieved high confidence scores, with values ranging from **69.6% to 99.8%** for various test images.
- **Performance**: Visualizations, such as box plots and histograms, indicate stable performance with minor outliers that warrant further analysis.

## Troubleshooting

- **SageMaker Permissions**: Ensure that your **SageMaker role** has the necessary permissions for accessing S3, Lambda, and Step Functions.
- **Endpoint Issues**: If the SageMaker endpoint fails, check the **CloudWatch Logs** for detailed error messages and troubleshooting information.
- **Data Drift**: Use SageMaker Model Monitor to regularly check for data drift and retrain the model as necessary.

## References

- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [AWS Lambda Documentation](https://aws.amazon.com/lambda/)
- [Step Functions Developer Guide](https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html)
