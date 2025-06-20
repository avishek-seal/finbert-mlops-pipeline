# FinBERT MLOps Pipeline on SageMaker

A reproducible, modular, end-to-end pipeline for daily fine-tuning and deployment of FinBERT using Amazon SageMaker Pipelines, sourcing fresh data from Amazon Redshift.

## Structure

- `src/preprocess.py` – Tokenizes/preps data for model training.
- `src/train.py`      – Fine-tunes FinBERT on your data.
- `src/evaluate.py`   – Evaluates the model, outputs metrics.
- `pipeline.py`       – SageMaker pipeline definition.
- `redshift_to_s3.py` – Redshift delta export utility.
- `deploy.py`         – Deploys trained model to a SageMaker endpoint (with autoscaling).
- `lambda_trigger.py` – Lambda function to kick off the pipeline.
- `requirements.txt`  – Python dependencies.

## Usage

1. **Clone repo:**  
   `git clone https://github.com/YOUR_ORG/finbert-mlops-pipeline.git`

2. **Install dependencies:**  
   `pip install -r requirements.txt`

3. **Edit S3 paths, IAM role ARNs, Redshift details.**

4. **Upload to S3 or run in SageMaker Studio.**

5. **Run `pipeline.py` to register/update your SageMaker pipeline.**

6. **Automate with Lambda/EventBridge (see `lambda_trigger.py`).**

7. **Deploy to a managed endpoint (`deploy.py`).**

---

**Author:**  
Your Name / Organization
