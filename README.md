# Next Assignment 🚀

[![GitHub Actions CI](https://github.com/harshbari-153/Next-Assignment/actions/workflows/run-tests.yaml/badge.svg)](https://github.com/harshbari-153/Next-Assignment/actions/workflows/run-tests.yaml)
[![GitHub Actions ETL](https://github.com/harshbari-153/Next-Assignment/actions/workflows/etl_daily.yaml/badge.svg)](https://github.com/harshbari-153/Next-Assignment/actions/workflows/etl_daily.yaml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://next-assignment.streamlit.app/)

Next Assignment transforms trending Indian news into practical, real-world coding assignments. It's designed for learners and professionals who want to apply their technical skills to current events, moving beyond generic textbook problems.

---

## 🎯 Core Features

* **Automated News Pipeline**: A daily ETL process fetches, transforms, and stores the top 5 trending Indian news articles.
* **AI-Powered Assignment Generation**: Uses Google's Gemini model to create unique, relevant assignment prompts based on news content.
* **Semantic Search**: Employs sentence transformers to find the most contextually relevant news articles for a user's specified skills.
* **CI/CD Automation**: Integrated with GitHub Actions for daily data refreshes and continuous integration testing on every commit.
* **Interactive UI**: A clean and simple user interface built with Streamlit.

---

## 🛠️ How It Works & Tech Stack

The project is split into two main components: a daily automated ETL pipeline and the user-facing assignment generation application.



### 1. The ETL (Extract, Transform, Load) Pipeline

This automated process runs every day at 12 AM via GitHub Actions to keep our news database fresh.

1.  **Extract**: Fetches the top 5 trending news articles for India from the **GNews API**.
2.  **Transform**: The raw content of each article is processed by the **`gemini-2.5-flash`** model to summarize and structure the data.
3.  **Load**: The transformed data is stored in a **PostgreSQL** database hosted on **Neon**.
4.  **Maintain**: To keep the dataset relevant, the script automatically deletes the 5 oldest news articles if the total number of rows exceeds 10.

### 2. Assignment Generation

This is the interactive part of the application where users get their custom assignments.

1.  **User Input**: The user enters a subject (e.g., *Machine Learning*) and three relevant skills (e.g., *Python, Scikit-learn, Classification*) via the **Streamlit** interface.
2.  **Embedding**: The user's input is combined and converted into a vector embedding using the **`all-MiniLM-L6-v2`** sentence transformer model.
3.  **Similarity Search**: The application performs a cosine similarity search between the user's input embedding and the pre-computed embeddings of the news articles stored in the PostgreSQL database.
4.  **AI Generation**: The content of the top 3 most relevant news articles is passed to the **`gemini-2.5-flash`** model with a specific prompt to generate three interesting assignments, each under 100 words.
5.  **Display**: The generated assignments are presented to the user.

### Tech Stack

* **Backend**: Python
* **Frontend**: Streamlit
* **Database**: PostgreSQL (on Neon)
* **AI/LLMs**: Google Gemini (`gemini-2.5-flash`), Sentence Transformers (`all-MiniLM-L6-v2`)
* **APIs**: GNews API
* **CI/CD & Automation**: GitHub Actions
* **Testing**: Pytest

---

## ⚙️ Local Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harshbari-153/Next-Assignment.git](https://github.com/harshbari-153/Next-Assignment.git)
    cd Next-Assignment
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys and database URL:
    ```env
    GNEWS_API_KEY="YOUR_GNEWS_API_KEY"
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    DATABASE_URL="YOUR_NEON_POSTGRESQL_URL"
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app/home_page.py
    ```

---

## 🔄 Continuous Integration & Deployment

This project uses **GitHub Actions** for automation and testing.

* **`etl_daily.yaml`**: This workflow runs on a schedule (daily at 12 AM) to execute the `etl/etl.py` script, ensuring the news data is always up-to-date.
* **`run-tests.yaml`**: Triggered on every push or pull request, this workflow installs dependencies and runs the unit tests located in the `tests/` directory to validate the embedding model's functionality and maintain code quality.

The application is deployed and publicly accessible on **Streamlit Community Cloud**.

---

Built with ❤️ by **Harsh Bari** (September 15-17, 2025).
