from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task

from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database,
)

with DAG(
    dag_id="etl_pipeline_3",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule="0 0 * * *",
    catchup=False,
    tags=["ETL", "news"],
) as dag:
    
    @task
    def extract_and_transform():
        """
        Extracts news headlines and URLs, then adds body content and metadata.
        This single task combines multiple steps for better performance and data passing.
        """
        from airflow.models.variable import Variable
        
        gnews_api = Variable.get("GNews_API")
        gemini_api = Variable.get("Gemini_API")
        
        news_list = find_latest_headline_and_url(gnews_api)
        news_list = get_body(news_list)
        news_list = get_metadata(news_list, gemini_api)
        
        return news_list
        
    @task
    def load_to_database(news_list):
        """
        Loads the processed news data into the database.
        """
        from airflow.models.variable import Variable
        
        postgres_api = Variable.get("PostgreSQL_API")
        add_to_database(news_list, postgres_api)
    
    # Task dependencies
    processed_data = extract_and_transform()
    load_to_database(processed_data)