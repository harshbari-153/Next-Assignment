import streamlit as st
import os
import time
import psycopg2
import numpy as np
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(
    page_title="Next Assignment",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Load Environment Variables ---
# Assumes .env file is in the parent directory of the 'app' folder
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

NEON_DATABASE_URL = os.getenv("Neon_Database")
GEMINI_API_KEYS = [
    os.getenv("Gemini_API_1"),
    os.getenv("Gemini_API_2"),
    os.getenv("Gemini_API_3"),
]

# --- Caching ---
# Cache the sentence transformer model to prevent reloading
@st.cache_resource
def load_sentence_model():
    """Loads the SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cache the database connection
@st.cache_resource
def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        connection = psycopg2.connect(NEON_DATABASE_URL)
        return connection
    except psycopg2.OperationalError as e:
        st.error(f"🚨 Database Connection Error: Could not connect to the database. Please check your connection string. Details: {e}")
        return None

# --- Helper Functions ---

def fetch_initial_data(conn):
    """Fetches preliminary data from the database."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db")
            records = cur.fetchall()
            return records
    except Exception as e:
        st.error(f"🚨 Database Fetch Error: Failed to retrieve initial data. Details: {e}")
        return []

def fetch_detailed_news(conn, ids):
    """Fetches detailed news for the top 3 IDs."""
    try:
        with conn.cursor() as cur:
            # Using tuple(ids) for the IN clause
            cur.execute("SELECT id, headline, news, impact, emotion FROM news_db WHERE id IN %s", (tuple(ids),))
            records = cur.fetchall()
            # Convert list of tuples to a dictionary for easy lookup
            return {rec[0]: {'headline': rec[1], 'news': rec[2], 'impact': rec[3], 'emotion': rec[4]} for rec in records}
    except Exception as e:
        st.error(f"🚨 Database Fetch Error: Failed to retrieve detailed news. Details: {e}")
        return {}

def fetch_image_from_url(url):
    """Scrapes the first image from a given URL."""
    placeholder_image = "https://placehold.co/600x400/EEE/31343C?text=Image+Not+Found"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        image_tag = soup.find('img')
        
        if image_tag and image_tag.get('src'):
            img_src = image_tag['src']
            # Create absolute URL if the source is relative
            return urljoin(url, img_src)
        return placeholder_image
    except Exception:
        return placeholder_image

def parse_gemini_output(text):
    """Parses Gemini's structured output robustly."""
    try:
        headline = text.split("**Assignment Headline:**")[1].split("**Assignment Overview:**")[0].strip()
        overview = text.split("**Assignment Overview:**")[1].split("**Assignment Instructions:**")[0].strip()
        instructions = text.split("**Assignment Instructions:**")[1].strip()
        return headline, overview, instructions
    except IndexError:
        # If parsing fails, return the raw text with error messages
        return "Could not parse headline.", "Could not parse overview.", text

def generate_assignment(api_key, index, content):
    """Calls Gemini API to generate an assignment and handles potential errors."""
    st.session_state.status_text.text(f"Generating Assignment {index+1}...")
    time.sleep(5)  # 5-second delay between API calls
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(content)
        return parse_gemini_output(response.text)
    except Exception as e:
        st.warning(f"⚠️ Gemini API Error for Assignment {index+1}: {e}. Trying next key if available.")
        return "Error Generating Headline", "Error Generating Overview", f"Failed to generate instructions due to an API error: {e}"

# --- Custom Styling ---
st.markdown("""
<style>
.assignment-block {
    background-color: #f8f9fa;
    border-left: 6px solid #17a2b8;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.assignment-block h3 {
    color: #343a40;
    margin-top: 0;
}
.assignment-block p {
    color: #495057;
}
.assignment-url {
    font-size: 0.9em;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# --- UI and Main Logic ---

st.title("📚 Next Assignment")
st.write("Enter a subject and three skills to find relevant project assignments based on recent news.")

# Initialize session state for assignments
if 'assignments' not in st.session_state:
    st.session_state.assignments = None

# Create a form for user inputs
with st.form(key="assignment_form"):
    user_subject = st.text_input(
        "Enter the name of the subject", 
        max_chars=30, 
        key="user_subject",
        placeholder="e.g., Data Science"
    )
    skill_1 = st.text_input(
        "Enter the first skill", 
        max_chars=30, 
        key="skill_1",
        placeholder="e.g., Python"
    )
    skill_2 = st.text_input(
        "Enter the second skill", 
        max_chars=30, 
        key="skill_2",
        placeholder="e.g., NLP"
    )
    skill_3 = st.text_input(
        "Enter the third skill", 
        max_chars=30, 
        key="skill_3",
        placeholder="e.g., Data Visualization"
    )
    submit_button = st.form_submit_button(label="✨ Find Assignment")

if submit_button:
    # --- Input Validation ---
    if not all([user_subject, skill_1, skill_2, skill_3]):
        st.error("🚫 All fields are compulsory. Please fill them all out.")
    else:
        # --- Start Processing ---
        st.session_state.assignments = None # Clear previous results
        progress_bar = st.progress(0)
        st.session_state.status_text = st.empty()

        with st.spinner("🚀 Launching the assignment finder... Please wait."):
            
            # Step 1: Fetch From Database
            st.session_state.status_text.text("Step 1/7: Fetching From Database...")
            conn = get_db_connection()
            if not conn:
                st.stop() # Stop execution if DB connection failed
            
            db_records = fetch_initial_data(conn)
            if not db_records:
                st.error("No records found in the database.")
                st.stop()
            progress_bar.progress(10)

            # Step 2: Merge Keywords
            st.session_state.status_text.text("Step 2/7: Merging keywords...")
            user_str = f"{user_subject}, {skill_1}, {skill_2}, {skill_3}"
            
            processed_data = []
            for record in db_records:
                id_val, url_val, news_type, k1, k2, k3 = record
                # Filter out None values before joining
                parts = [part for part in [news_type, k1, k2, k3] if part]
                merged_str = ", ".join(parts)
                processed_data.append({'id': id_val, 'url': url_val, 'str': merged_str})
            progress_bar.progress(20)

            # Step 3: Get Embeddings
            st.session_state.status_text.text("Step 3/7: Getting Embeddings...")
            model = load_sentence_model()
            db_strings = [item['str'] for item in processed_data]
            all_strings = [user_str] + db_strings
            
            embeddings = model.encode(all_strings, show_progress_bar=False)
            user_str_embd = embeddings[0]
            db_embds = embeddings[1:]

            for i, item in enumerate(processed_data):
                item['embd'] = db_embds[i]
            progress_bar.progress(50)

            # Step 4: Find Cosine Similarity
            st.session_state.status_text.text("Step 4/7: Finding Cosine Similarity...")
            for item in processed_data:
                item['score'] = cosine_similarity(
                    user_str_embd.reshape(1, -1),
                    item['embd'].reshape(1, -1)
                )[0][0]
            
            # Sort by score and get top 3
            sorted_data = sorted(processed_data, key=lambda x: x['score'], reverse=True)
            top_3 = sorted_data[:3]
            top_3_ids = [item['id'] for item in top_3]
            progress_bar.progress(60)

            # Step 5: Fetch Detailed News
            st.session_state.status_text.text("Step 5/7: Fetching News From Database...")
            detailed_news_map = fetch_detailed_news(conn, top_3_ids)
            conn.close() # Close connection as we are done with the DB

            for item in top_3:
                details = detailed_news_map.get(item['id'], {})
                item.update(details)
            progress_bar.progress(70)

            # Step 6: Generate Assignments
            final_assignments = []
            api_key_cycle = 0
            
            prompt_template = """
            You are a project generator. Your task is to create an excellent project assignment based on the provided variables.

            Variables:
            headline: {headline}
            news: {news}
            impact: {impact}
            emotion: {emotion}
            subject: {subject}
            skills: {skills}

            Instructions:
            1. **Integrate all variables:** The assignment must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
            2. **Focus on skills:** The project must require students to demonstrate and apply {skill_1}, {skill_2}, and {skill_3} within the {subject} context.
            3. **Structure the output strictly as shown below with the exact bolded labels:**
               - **Assignment Headline:** A title, max 10 words.
               - **Assignment Overview:** A brief summary of the project's goal, max 20 words.
               - **Assignment Instructions:** The steps students must follow, max 40 words.
            4. **Tone:** The assignment should be relevant to emotion, and inspiring for students.

            **Assignment Headline:**
            **Assignment Overview:**
            **Assignment Instructions:**
            """

            for i, item in enumerate(top_3):
                prompt = prompt_template.format(
                    headline=item.get('headline', 'N/A'),
                    news=item.get('news', 'N/A'),
                    impact=item.get('impact', 'N/A'),
                    emotion=item.get('emotion', 'N/A'),
                    subject=user_subject,
                    skills=user_str,
                    skill_1=skill_1,
                    skill_2=skill_2,
                    skill_3=skill_3
                )
                
                ass_headline, ass_overview, ass_instructions = generate_assignment(
                    GEMINI_API_KEYS[api_key_cycle], i, prompt
                )
                
                item['ass_headline'] = ass_headline
                item['ass_overview'] = ass_overview
                item['ass_instructions'] = ass_instructions
                
                final_assignments.append(item)
                api_key_cycle = (api_key_cycle + 1) % len(GEMINI_API_KEYS)
                progress_bar.progress(70 + (i + 1) * 10)

            # Step 7: Fetch Images
            st.session_state.status_text.text("Step 7/7: Fetching News Images...")
            for item in final_assignments:
                item['image_url'] = fetch_image_from_url(item['url'])
            
            st.session_state.assignments = final_assignments
            progress_bar.progress(100)
            st.session_state.status_text.text("✅ All done! Here are your assignments.")
            time.sleep(2)
            st.session_state.status_text.empty()
            progress_bar.empty()

# --- Display Results ---
if st.session_state.assignments:
    st.markdown("---")
    st.header("Generated Assignments")
    
    for assignment in st.session_state.assignments:
        with st.container():
            st.markdown(
                f"""
                <div class="assignment-block">
                    <h3>{assignment.get('ass_headline', 'Assignment Headline Not Available')}</h3>
                </div>
                """, unsafe_allow_html=True
            )
            
            st.image(
                assignment.get('image_url', ''), 
                caption=f"Source News: {assignment.get('headline', 'N/A')}"
            )
            
            st.markdown(
                f"""
                <div class="assignment-block">
                    <p><strong>Overview:</strong> {assignment.get('ass_overview', 'Not available.')}</p>
                    <p><strong>Instructions:</strong> {assignment.get('ass_instructions', 'Not available.')}</p>
                    <p class="assignment-url">For detailed news, visit: <a href="{assignment.get('url', '#')}" target="_blank">Source Article</a></p>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown("---")