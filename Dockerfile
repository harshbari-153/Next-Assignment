# Start with a base Airflow image from Astronomer
FROM quay.io/astronomer/astro-runtime:latest

# Install any Python dependencies from requirements.txt
# This ensures that your DAGs and modules have the necessary libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt