from __future__ import annotations

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import psycopg2
from psycopg2.extras import execute_values

import pendulum
import os # Import the os module
from dotenv import load_dotenv

from airflow.models.dag import DAG
from airflow.decorators import task, dag


from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database,
)


@dag(
    dag_id="fetch_news",   # FIXED (no spaces)
    start_date=pendulum.datetime(2025, 4, 22),
    schedule="0 0 * * *",
    default_args={"owner": "Harsh Bari", "retries": 3},
    catchup=False,
    tags=["ETL", "news"],
)
def start_etl():
    """ETL pipeline to fetch latest news, extract body, enrich metadata, and store in DB."""

    @task
    def task_1(api_key: str):
        return find_latest_headline_and_url(api_key)

    @task
    def task_2(news_list: list):
        return get_body(news_list)

    @task
    def task_3(news_list: list, api_key: str):
        return get_metadata(news_list, api_key)

    @task
    def task_4(news_list: list, db_url: str):
        add_to_database(news_list, db_url)

    # ✅ Use consistent env variable names
    gnews_api = os.getenv("GNEWS_API")
    gemini_api = os.getenv("GEMINI_API")
    postgres_url = os.getenv("POSTGRESQL_API")

    # ✅ Chain tasks properly
    news_list = task_1(gnews_api)
    news_list = task_2(news_list)
    news_list = task_3(news_list, gemini_api)
    task_4(news_list, postgres_url)

# ✅ Instantiate DAG so Airflow can discover it
dag = start_etl()

