from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Zuhthi Sahan',
    'depends_on_past' : False,
    'start_date': datetime(2024, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'smart_portfolio_pipeline',
    default_args = default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:
    
    # 1: Bronze to Silver (transactions and User data)
    clean_transactions = BashOperator(
        task_id = 'clean_transactions',
        bash_command = 'python /opt/airflow/src/pipeline/bronze_to_silver_transactions.py'
    )
    
    clean_users = BashOperator(
        task_id = 'clean_users',
        bash_command = 'python /opt/airflow/src/pipeline/bronze_to_silver_users.py'
    )
    
    # 2: Silver to Gold (Feature Engineering)
    silver_to_gold = BashOperator(
        task_id = 'silver_to_gold',
        bash_command = 'python /opt/airflow/src/pipeline/silver_to_gold.py'
    )
    
    # 3: run the optimizer
    run_optimizer = BashOperator(
        task_id='run_portfolio_optimizer',
        bash_command='python /opt/airflow/src/optimizer/portfolio_optimizer.py'
    )
    
    # 4. Gold timee-series user tables
    gold_user_ts = BashOperator(
        task_id = 'gold_user_time_series',
        bash_command = 'python /opt/airflow/src/pipeline/silver_to_gold_timeseries.py'
    )
    
    # Set dependencies
    [clean_transactions, clean_users] >> silver_to_gold >> [run_optimizer, gold_user_ts]