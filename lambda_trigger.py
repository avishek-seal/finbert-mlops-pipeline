import boto3

def lambda_handler(event, context):
    sagemaker = boto3.client("sagemaker")
    response = sagemaker.start_pipeline_execution(
        PipelineName="FinBertDailyPipeline",
        PipelineParameters=[{
            "Name": "InputData",
            "Value": "s3://your-bucket/finbert/delta/"
        }]
    )
    return {"status": "Started", "pipeline_execution": response["PipelineExecutionArn"]}
