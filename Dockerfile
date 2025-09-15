FROM astrocrpublic.azurecr.io/runtime:3.0-10

# Install dependencies via pip
RUN pip install \
    requests \
    beautifulsoup4 \
    google-generativeai \
    psycopg2-binary \
    newsapi-python

# ✅ Install bash to avoid WSL/bash issues during deploy
USER root
RUN apt-get update && apt-get install -y bash
USER astro