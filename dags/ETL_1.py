from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database
)

API_KEY = os.getenv("GNEWS_API_KEY")
GEMINI_API = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# with DAG(
#     dag_id="daily_news_etl",
#     schedule="0 0 * * *",  # Daily at midnight
#     start_date=datetime(2025, 1, 1),
#     catchup=False
# ) as dag:

with DAG(
    dag_id="daily_news_etl",
    schedule_interval="0 0 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    def extract_news():
        return find_latest_headline_and_url(API_KEY)

    def extract_body(ti):
        news = ti.xcom_pull(task_ids="extract_news")
        return get_body(news)

    def enrich(ti):
        body = ti.xcom_pull(task_ids="extract_body")
        return get_metadata(body, GEMINI_API)

    def load(ti):
        enriched = ti.xcom_pull(task_ids="enrich")
        add_to_database(enriched, DATABASE_URL)

    t1 = PythonOperator(task_id="extract_news", python_callable=extract_news)
    t2 = PythonOperator(task_id="extract_body", python_callable=extract_body)
    t3 = PythonOperator(task_id="enrich", python_callable=enrich)
    t4 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3 >> t4
