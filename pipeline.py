import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.huggingface import HuggingFace
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

role = sagemaker.get_execution_role()
region = sagemaker.Session().boto_region_name
sess = sagemaker.Session()

input_data = ParameterString(name="InputData", default_value="s3://your-bucket/finbert/delta/")
epochs = ParameterInteger(name="Epochs", default_value=2)
lr = ParameterFloat(name="LearningRate", default_value=2e-5)

processor = ScriptProcessor(
    image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.2.0-transformers4.39.0-cpu-py310",
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)
step_process = ProcessingStep(
    name="PreprocessStep",
    processor=processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination="s3://your-bucket/finbert/preprocessed/")],
    code="src/preprocess.py"
)

hyperparameters = {
    "epochs": epochs,
    "learning_rate": lr,
    "model_id": "ProsusAI/finbert"
}
estimator = HuggingFace(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.39.0",
    pytorch_version="2.2.0",
    py_version="py310",
    hyperparameters=hyperparameters
)
step_train = TrainingStep(
    name="TrainStep",
    estimator=estimator,
    inputs={
        "train": step_process.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri
    }
)

eval_processor = ScriptProcessor(
    image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:2.2.0-transformers4.39.0-cpu-py310",
    command=["python3"],
    role=role,
    instance_count=1,
    instance_type="ml.m5.large"
)
step_eval = ProcessingStep(
    name="EvaluateStep",
    processor=eval_processor,
    inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=step_process.outputs[0].destination, destination="/opt/ml/processing/data")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/eval", destination="s3://your-bucket/finbert/eval/")
    ],
    code="src/evaluate.py"
)

condition_step = ConditionStep(
    name="CheckEvalMetric",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=step_eval.outputs[0].destination + "/evaluation.json",
            right=0.85
        )
    ],
    if_steps=[step_train],  # For demo; replace with ModelStep for deployment
    else_steps=[]
)

pipeline = Pipeline(
    name="FinBertDailyPipeline",
    parameters=[input_data, epochs, lr],
    steps=[step_process, step_train, step_eval, condition_step]
)
if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
