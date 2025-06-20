import boto3
import time

redshift = boto3.client('redshift-data', region_name='ap-south-1')

def extract_delta_to_s3(sql, cluster_id, db, user):
    response = redshift.execute_statement(
        ClusterIdentifier=cluster_id,
        Database=db,
        DbUser=user,
        Sql=sql
    )
    while True:
        desc = redshift.describe_statement(Id=response['Id'])
        if desc['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
            break
        time.sleep(10)
    print("Query Status:", desc['Status'])

if __name__ == "__main__":
    unload_sql = '''
    UNLOAD ('SELECT * FROM financial_data WHERE ingestion_ts > dateadd(day, -1, current_date)')
    TO 's3://your-bucket/finbert/delta/'
    IAM_ROLE 'arn:aws:iam::1234567890:role/RedshiftUnloadRole'
    PARQUET;
    '''
    extract_delta_to_s3(unload_sql, 'redshift-cluster-1', 'analytics', 'rsadmin')
