from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

# Import your ETL functions
from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database
)

# Load environment variables
GENEWS_API = os.getenv("GNews_API")
GEMINI_API = os.getenv("Gemini_API")
SUPABASE_DB_URL = os.getenv("PostgreSQL_API")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="ETL_2",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule="0 0 * * *",  # Every day at 12 AM
    catchup=False
) as dag:

    def extract_task(**kwargs):
        news_list = find_latest_headline_and_url(GENEWS_API)
        kwargs['ti'].xcom_push(key='news_list', value=news_list)

    def transform_task(**kwargs):
        news_list = kwargs['ti'].xcom_pull(key='news_list')
        detailed_news = get_body(news_list)
        transformed_news = get_metadata(detailed_news, GEMINI_API)
        kwargs['ti'].xcom_push(key='transformed_news', value=transformed_news)

    def load_task(**kwargs):
        transformed_news = kwargs['ti'].xcom_pull(key='transformed_news')
        add_to_database(transformed_news, SUPABASE_DB_URL)

    extract = PythonOperator(
        task_id="extract_news",
        python_callable=extract_task
        #provide_context=True
    )

    transform = PythonOperator(
        task_id="transform_news",
        python_callable=transform_task
        #provide_context=True
    )

    load = PythonOperator(
        task_id="load_news",
        python_callable=load_task
        #provide_context=True
    )

    extract >> transform >> load
