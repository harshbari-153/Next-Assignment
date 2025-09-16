# app/home_page.py

import streamlit as st
import os
import time
import psycopg2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. INITIAL CONFIGURATION & SETUP ---

# Load environment variables from the parent directory's .env file
# This path is crucial for running `streamlit run app/home_page.py` from the root directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Page configuration
st.set_page_config(
    page_title="Next Assignment",
    page_icon="📚",
    layout="centered"
)

# Load API keys and credentials from .env
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
GEMINI_API_KEYS = [
    os.getenv("Gemini_API_1"),
    os.getenv("Gemini_API_2"),
    os.getenv("Gemini_API_3"),
]

# --- 2. SESSION STATE INITIALIZATION ---
# This is key for data persistence and multi-page simulation

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'assignments' not in st.session_state:
    st.session_state.assignments = None
if 'current_assignment_index' not in st.session_state:
    st.session_state.current_assignment_index = None
if 'api_key_index' not in st.session_state:
    st.session_state.api_key_index = 0

# --- 3. CACHED RESOURCES (for performance) ---

@st.cache_resource
def get_db_connection():
    """Establishes and caches the database connection."""
    try:
        conn = psycopg2.connect(SUPABASE_URL)
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# --- 4. HELPER FUNCTIONS ---

def get_gemini_model():
    """Rotates through Gemini API keys for reliability."""
    api_key = GEMINI_API_KEYS[st.session_state.api_key_index]
    st.session_state.api_key_index = (st.session_state.api_key_index + 1) % len(GEMINI_API_KEYS)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def parse_gemini_output(text):
    """Safely parses the structured output from Gemini."""
    try:
        headline = text.split("Assignment Headline:")[1].split("Assignment Overview:")[0].strip()
        overview = text.split("Assignment Overview:")[1].split("Assignment Instructions:")[0].strip()
        instructions = text.split("Assignment Instructions:")[1].strip()
        return headline, overview, instructions
    except IndexError:
        # If parsing fails, return the raw text and a flag
        return "Parsing Failed", "Could not extract details from the response.", text

# --- 5. CORE PROCESSING LOGIC ---

def find_and_generate_assignments(user_subject, skill_1, skill_2, skill_3):
    """
    The main pipeline function that executes all steps from data fetching to assignment generation.
    """
    status_container = st.empty()
    progress_bar = st.progress(0)
    timer_container = st.empty()
    start_time = time.time()

    try:
        # --- Step 1: Fetching From Database ---
        status_container.info("Step 1/6:  fetching from database...")
        conn = get_db_connection()
        if not conn: return
        cursor = conn.cursor()
        cursor.execute("SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db")
        records = cursor.fetchall()
        progress_bar.progress(10)

        # --- Step 2: Merging keywords ---
        status_container.info("Step 2/6: Merging keywords...")
        db_data = []
        for rec in records:
            id, url, news_type, k1, k2, k3 = rec
            # Filter out None values before joining
            parts = [part for part in [news_type, k1, k2, k3] if part]
            merged_str = ", ".join(parts)
            db_data.append({'id': id, 'url': url, 'str': merged_str})
        
        user_str = f"{user_subject}, {skill_1}, {skill_2}, {skill_3}"
        progress_bar.progress(20)

        # --- Step 3: Getting Embeddings ---
        status_container.info("Step 3/6: Getting embeddings...")
        model = load_embedding_model()
        if not model: return
        
        db_strings = [item['str'] for item in db_data]
        db_embeddings = model.encode(db_strings, show_progress_bar=True)
        user_embedding = model.encode([user_str])
        
        for i, item in enumerate(db_data):
            item['embd'] = db_embeddings[i]
        progress_bar.progress(40)

        # --- Step 4: Finding Cosine Similarity ---
        status_container.info("Step 4/6: Finding cosine similarity...")
        similarities = cosine_similarity(user_embedding, db_embeddings)[0]
        
        for i, item in enumerate(db_data):
            item['score'] = similarities[i]

        # Get top 3 indices, handling ties by taking the first ones found
        top_3_indices = np.argsort(similarities)[-3:][::-1]
        top_3_data = [db_data[i] for i in top_3_indices]
        top_3_ids = tuple(item['id'] for item in top_3_data)
        progress_bar.progress(60)

        # --- Step 5: Fetching News From Database ---
        status_container.info("Step 5/6: Fetching full news for top 3...")
        if not top_3_ids:
            st.warning("Could not find any relevant news articles. Please try different skills.")
            return

        cursor.execute("SELECT id, headline, news, impact, emotion FROM news_db WHERE id IN %s", (top_3_ids,))
        news_details = cursor.fetchall()
        
        # Map details back to the top_3_data list
        details_map = {det[0]: det[1:] for det in news_details}
        for item in top_3_data:
            details = details_map.get(item['id'])
            if details:
                item['headline'], item['news'], item['impact'], item['emotion'] = details
        
        progress_bar.progress(80)
        
        # --- Step 6: Generating Assignments ---
        final_assignments = []
        for i, item in enumerate(top_3_data):
            status_container.info(f"Step 6/6: Generating assignment {i+1}/3...")
            
            prompt = f"""
            You are a project generator. Your task is to create an excellent project assignment based on the provided variables.

            Variables:
            headline: {item.get('headline', 'N/A')}
            news: {item.get('news', 'N/A')}
            impact: {item.get('impact', 'N/A')}
            emotion: {item.get('emotion', 'N/A')}
            subject: {user_subject}
            skills: {user_str}

            Instructions:
            1. **Integrate all variables:** The assignment must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
            2. **Focus on skills:** The project must require students to demonstrate and apply the skills: {skill_1}, {skill_2}, and {skill_3} within the {user_subject} context.
            3. **Structure the output strictly as follows and do not add any other text or markdown:**
            - **Assignment Headline:** A title, max 10 words.
            - **Assignment Overview:** A brief summary of the project's goal, max 20 words.
            - **Assignment Instructions:** The steps students must follow, max 40 words.
            4. **Tone:** The assignment should be relevant to the emotion and inspiring for students.

            **Assignment Headline:**
            **Assignment Overview:**
            **Assignment Instructions:**
            """
            try:
                model = get_gemini_model()
                response = model.generate_content(prompt)
                
                # Robust parsing
                headline, overview, instructions = parse_gemini_output(response.text)
                item['ass_headline'] = headline
                item['ass_overview'] = overview
                item['ass_instructions'] = instructions

            except Exception as e:
                st.error(f"An error occurred with the Gemini API for assignment {i+1}: {e}")
                item['ass_headline'] = "Error Generating Assignment"
                item['ass_overview'] = "The model failed to generate content."
                item['ass_instructions'] = f"Details: {str(e)}"
            
            final_assignments.append(item)
            time.sleep(8) # Mandatory 8-second delay

        st.session_state.assignments = final_assignments
        progress_bar.progress(100)
        status_container.success("✅ All assignments generated successfully!")
        time.sleep(2)

    except Exception as e:
        status_container.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up UI elements
        progress_bar.empty()
        status_container.empty()
        timer_container.empty()
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

# --- 6. UI RENDERING LOGIC ---

def render_home_page():
    """Renders the main input form and results."""
    st.title("📚 Next Assignment Generator")
    st.markdown("Enter a subject and three skills to discover your next project, inspired by recent news.")
    
    with st.form(key="assignment_form"):
        user_subject = st.text_input(
            "Enter Subject Name",
            max_chars=30,
            placeholder="e.g., Data Science"
        )
        skill_1 = st.text_input(
            "Enter Skill 1",
            max_chars=30,
            placeholder="e.g., Python"
        )
        skill_2 = st.text_input(
            "Enter Skill 2",
            max_chars=30,
            placeholder="e.g., NLP"
        )
        skill_3 = st.text_input(
            "Enter Skill 3",
            max_chars=30,
            placeholder="e.g., Visualization"
        )
        submit_button = st.form_submit_button(label="Find Assignment ✨")

    if submit_button:
        if not all([user_subject, skill_1, skill_2, skill_3]):
            st.warning("All fields are compulsory. Please fill them all out.")
        else:
            st.session_state.assignments = None # Clear old results
            find_and_generate_assignments(user_subject, skill_1, skill_2, skill_3)

    if st.session_state.assignments:
        st.subheader("Generated Assignments:")
        for i, assignment in enumerate(st.session_state.assignments):
            if st.button(f"Assignment {i+1}: {assignment.get('ass_headline', 'Untitled')}", key=f"view_{i}"):
                st.session_state.current_assignment_index = i
                st.session_state.page = 'details'
                st.rerun()

def render_details_page():
    """Renders the detailed view of a selected assignment."""
    st.title("Assignment Details")
    
    if st.session_state.current_assignment_index is not None:
        assignment = st.session_state.assignments[st.session_state.current_assignment_index]

        st.subheader(f"Assignment Headline:")
        st.write(assignment.get('ass_headline', 'Not available.'))
        
        st.subheader("Assignment Overview:")
        st.write(assignment.get('ass_overview', 'Not available.'))
        
        st.subheader("Assignment Instructions:")
        st.write(assignment.get('ass_instructions', 'Not available.'))

        st.markdown("---")
        st.write(f"**Source News ID:** `{assignment.get('id')}`")
        st.write(f"**Source URL:** [Read full article]({assignment.get('url')})")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'home'
        st.session_state.current_assignment_index = None
        st.rerun()

# --- MAIN APP ROUTER ---
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'details':
    render_details_page()
