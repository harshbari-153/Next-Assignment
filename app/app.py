import streamlit as st
import psycopg2
import json
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
from itertools import islice

# Load .env from one level above
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.getcwd()), '.env'))

# API Keys
DB_URL = os.getenv("PostgreSQL_API")
GEMINI_API_KEY = os.getenv("Gemini_API")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Helper: Chunk a list
def chunked(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

# DB Fetch
def fetch_news_records():
    with st.spinner("Fetching From Database"):
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "url": row[1],
                "news_type": row[2],
                "key_1": row[3],
                "key_2": row[4],
                "key_3": row[5]
            })
        return result

# Gemini Score Calculation
def calculate_scores(records, preferences):
    scored_results = []
    st.spinner("Getting Relevant Score")
    with st.spinner("Getting Relevant Score"):
        chunks = chunked(records, 3)
        for i, chunk in enumerate(chunks):
            prompt = f"""
You are a data analyst. Given a JSON array of records and a user's preferences, calculate a similarity score for each record from 0 to 10 with 2 decimal places. The score should be based on the relevance of the record's content to the user's preferences.

**Records (JSON array):**
{json.dumps(chunk)}

**User Preferences:**
{json.dumps(preferences)}

**Scoring Logic:**
- Assign a high weight to matches between `news_type` and `subject`.
- Assign a moderate weight to the number of matching skills and keys.
- A perfect match on all attributes should result in a score of 10.

**Response Format:**
Provide a single JSON array of objects. Each object must contain the `id` and the final `score`.

Example response:
[
{{ "id": "3288179e8a01b82ac8a1a13a85753632", "score": 8.75 }},
{{ "id": "5b88279e1ay1b82ag8a1ad3a8s75363a", "score": 2.50 }}
]
"""
            response = model.generate_content(prompt)
            try:
                scores = json.loads(response.text)
                scored_results.extend(scores)
            except:
                st.error("Failed to parse response from Gemini")
            time.sleep(8)
    return scored_results

# Fetch News for Top 3
def fetch_top3_news(top3_ids):
    with st.spinner("Fetching News From Database"):
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        placeholders = ','.join(['%s'] * len(top3_ids))
        cur.execute(f"SELECT id, headline, news, impact, emotion, url FROM news_db WHERE id IN ({placeholders})", tuple(top3_ids))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        results = {}
        for row in rows:
            results[row[0]] = {
                "headline": row[1],
                "news": row[2],
                "impact": row[3],
                "emotion": row[4],
                "url": row[5]
            }
        return results

# Generate Assignment
def generate_assignment(news_item, preferences):
    with st.spinner(f"Generating Assignment"):
        prompt = f"""
You are a project generator. Your task is to create an excellent project assignment based on the provided variables.

Variables:
headline: {news_item['headline']}
news: {news_item['news']}
impact: {news_item['impact']}
emotion: {news_item['emotion']}
subject: {preferences['subject']}
skill_1: {preferences['skill_1']}
skill_2: {preferences['skill_2']}
skill_3: {preferences['skill_3']}

Instructions:
1.  **Integrate all variables:** The project must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
2.  **Focus on skills:** The project must require students to demonstrate and apply the `skill_1`, `skill_2`, and `skill_3` within the `subject` context.
3.  **Structure the output strictly:**
    -   **Assignment Headline:** A title, max 10 words.
    -   **Assignment Overview:** A brief summary of the project's goal, max 20 words.
    -   **Assignment Instructions:** The steps students must follow, max 40 words.
4.  **Tone:** The assignment should be compelling, relevant, and inspiring for students.

**Assignment Headline:**
**Assignment Overview:**
**Assignment Instructions:**
"""
        response = model.generate_content(prompt)
        time.sleep(8)
        return response.text

# Page Routing
st.set_page_config(page_title="Next Assignment", layout="centered")
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.title("📘 Next Assignment")
    st.subheader("Enter Your Preferences")

    with st.form(key="input_form"):
        subject = st.text_input("Subject Name", max_chars=30)
        skill_1 = st.text_input("Skill 1", max_chars=30)
        skill_2 = st.text_input("Skill 2", max_chars=30)
        skill_3 = st.text_input("Skill 3", max_chars=30)
        submit = st.form_submit_button("Find Assignment")

    if submit:
        if not all([subject, skill_1, skill_2, skill_3]):
            st.warning("Please fill all fields.")
        else:
            preferences = {
                "subject": subject,
                "skill_1": skill_1,
                "skill_2": skill_2,
                "skill_3": skill_3,
            }

            records = fetch_news_records()
            scores = calculate_scores(records, preferences)
            top3 = sorted(scores, key=lambda x: x['score'], reverse=True)[:3]
            top3_ids = [item['id'] for item in top3]

            st.session_state.top3_ids = top3_ids
            st.session_state.preferences = preferences

            top3_news = fetch_top3_news(top3_ids)
            assignments = []

            for i, id in enumerate(top3_ids):
                st.spinner(f"Generating Assignment {i+1}")
                assignment = generate_assignment(top3_news[id], preferences)
                assignments.append({
                    "id": id,
                    "content": assignment,
                    "url": top3_news[id]["url"]
                })

            st.session_state.assignments = assignments
            st.session_state.page = 'results'
            st.experimental_rerun()

elif st.session_state.page == 'results':
    st.title("📝 Best 3 Assignments")
    st.subheader("Click on assignment to view full details")

    for i, a in enumerate(st.session_state.assignments):
        if st.button(f"{a['content'].splitlines()[1].replace('**Assignment Headline:**','').strip()}", key=f"btn_{i}"):
            st.session_state.selected_assignment = a
            st.session_state.page = 'details'
            st.experimental_rerun()

elif st.session_state.page == 'details':
    st.title("📄 Assignment Details")
    assignment = st.session_state.selected_assignment
    lines = assignment["content"].splitlines()
    parsed = {
        "Assignment Headline": lines[1].replace("**Assignment Headline:**", "").strip(),
        "Assignment Overview": lines[2].replace("**Assignment Overview:**", "").strip(),
        "Assignment Instructions": lines[3].replace("**Assignment Instructions:**", "").strip(),
        "URL": assignment["url"]
    }

    for key, value in parsed.items():
        st.markdown(f"**{key}:** {value}")

    if st.button("🔙 Back to Assignments"):
        st.session_state.page = 'results'
        st.experimental_rerun()
