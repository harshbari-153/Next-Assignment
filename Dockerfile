# Dockerfile for Astronomer / Airflow image
FROM astrocrpublic.azurecr.io/runtime:3.0-10

USER root

# Ensure bash is present (your previous comment mentioned WSL issues)
RUN apt-get update && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy dags & package code into the image
# Astronomer / Airflow standard: DAGs usually in /usr/local/airflow/dags
COPY dags /usr/local/airflow/dags
COPY include /usr/local/airflow/include
COPY plugins /usr/local/airflow/plugins

# Make sure ownership/permissions are okay for the runtime user
RUN chown -R  50000:50000 /usr/local/airflow/dags /usr/local/airflow/plugins /usr/local/airflow/include || true

# Switch back to the astro runtime user (lowercase 'astro' is common)
USER astro

# (Optional) set a label
LABEL org.opencontainers.image.source="https://github.com/harshbari-153/Next-Assignment"
