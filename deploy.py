from sagemaker.huggingface.model import HuggingFaceModel
import boto3
import sagemaker

role = sagemaker.get_execution_role()
model_data = "s3://your-bucket/finbert/model.tar.gz"  # Path to model artifact

# Deploy endpoint
hf_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version="4.39.0",
    pytorch_version="2.2.0",
    py_version="py310"
)

predictor = hf_model.deploy(
    initial_instance_count=2,
    instance_type="ml.m5.large",
    endpoint_name="finbert-inference"
)

# Enable autoscaling
client = boto3.client("application-autoscaling")

client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/finbert-inference/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=2,
    MaxCapacity=8
)
client.put_scaling_policy(
    PolicyName="InvocationsScaling",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/finbert-inference/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 50.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 60,
        "ScaleOutCooldown": 60
    }
)
print("Deployment and autoscaling set up complete.")
