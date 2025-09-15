import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import psycopg2
import os
from psycopg2.extras import execute_values
from __future__ import annotations

import pendulum
import os # Import the os module
from dotenv import load_dotenv

from airflow.models.dag import DAG
from airflow.decorators import task
from airflow.sdk import Asset, dag, task


from include.my_etl_module import (
    find_latest_headline_and_url,
    get_body,
    get_metadata,
    add_to_database,
)

# Define the basic parameters of the DAG, like schedule and start_date
@dag(
    dag_id="Fetch News",
    start_date=datetime(2025, 4, 22),
    schedule="0 0 * * *",
    default_args={"owner": "Harsh Bari", "retries": 3},
    catchup=False,
    tags=["ETL", "news"],
)

def start_etl():

  @task
  def task_1(api):
    return find_latest_headline_and_url(api)

  @task
  def task_2(news_list):
    return get_body(news_list)

  @task
  def task_3(news_list, api):
    return get_metadata(news_list, api)

  @task
  def task_4(news_list, api):
    add_to_database(news_list, api)


  # fetch variables
  gnews_api = os.environ.get("GNEWS_API")
  gemini_api = os.environ.get("GEMINI_API")
  postgres_api = os.environ.get("POSTGRESQL_API")

  news_list = task_1(gnews_api)
  news_list_2 = task_2(news_list)
  news_list_3 = task_3(news_list_2, gemini_api)
  task_4(news_list_3, postgres_api)


# Initiate ETL
start_etl()

