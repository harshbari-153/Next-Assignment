from __future__ import annotations

import pendulum
import os # Import the os module

# Use the modern Airflow decorators
from airflow.decorators import dag, task

from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database,
)

# Use the @dag decorator to define your DAG
@dag(
    dag_id="etl_pipeline_3",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule="0 0 * * *",
    catchup=False,
    tags=["ETL", "news"],
)
def etl_pipeline_3():
    
    @task
    def extract_and_transform():
        """
        Extracts news headlines and URLs, then adds body content and metadata.
        """
        # Read the environment variables using os.environ.get()
        gnews_api = os.environ.get("GNews_API")
        gemini_api = os.environ.get("Gemini_API")
        
        # Pass the API keys to your functions
        news_list = find_latest_headline_and_url(gnews_api)
        news_list = get_body(news_list)
        news_list = get_metadata(news_list, gemini_api)
        
        return news_list
        
    @task
    def load_to_database(news_list):
        """
        Loads the processed news data into the database.
        """
        # Read the environment variable for the database connection URI
        postgres_api = os.environ.get("PostgreSQL_API")
        add_to_database(news_list, postgres_api)
    
    # Define dependencies by calling the functions like a normal Python pipeline
    processed_data = extract_and_transform()
    load_to_database(processed_data)

# Call the DAG function to instantiate it
etl_pipeline_3()
