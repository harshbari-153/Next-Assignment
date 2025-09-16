import streamlit as st
import os
import time
import psycopg2
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Next Assignment",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load Environment Variables ---
# Load from the parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- API Keys ---
GEMINI_API_KEYS = [
    os.getenv("Gemini_API_1"),
    os.getenv("Gemini_API_2"),
    os.getenv("Gemini_API_3"),
]
POSTGRESQL_API = os.getenv("PostgreSQL_API")

# --- Helper Functions ---

def get_gemini_api_key():
    """Cycles through the available Gemini API keys."""
    if 'api_key_index' not in st.session_state:
        st.session_state.api_key_index = 0
    
    key = GEMINI_API_KEYS[st.session_state.api_key_index]
    st.session_state.api_key_index = (st.session_state.api_key_index + 1) % len(GEMINI_API_KEYS)
    return key

def connect_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(POSTGRESQL_API)
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

def get_embedding(text, model="models/embedding-001"):
    """Generates embeddings for a given text using a specified model, with error handling."""
    try:
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        embedding = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
        time.sleep(8) # 8-second delay as requested
        return embedding['embedding']
    except Exception as e:
        st.warning(f"Embedding failed for a text snippet. Retrying with next key. Error: {e}")
        # Simple retry logic with the next key
        try:
            api_key = get_gemini_api_key()
            genai.configure(api_key=api_key)
            embedding = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
            time.sleep(8)
            return embedding['embedding']
        except Exception as e2:
            st.error(f"Fatal Error during embedding: {e2}. Could not process the request.")
            return None

def generate_assignment_from_gemini(prompt_data):
    """Generates assignment content using Gemini Pro, with robust parsing."""
    try:
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a project generator. Your task is to create an excellent project assignment based on the provided variables.

        Variables:
        headline: {prompt_data['headline']}
        news: {prompt_data['news']}
        impact: {prompt_data['impact']}
        emotion: {prompt_data['emotion']}
        subject: {prompt_data['subject']}
        skills: {", ".join(prompt_data['skills'])}

        Instructions:
        1. **Integrate all variables:** The assignment must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
        2. **Focus on skills:** The project must require students to demonstrate and apply the skills within the subject context.
        3. **Structure the output strictly:**
           - **Assignment Headline:** A title, max 10 words.
           - **Assignment Overview:** A brief summary of the project's goal, max 20 words.
           - **Assignment Instructions:** The steps students must follow, max 40 words.
        4. **Tone:** The assignment should be relevant to emotion, and inspiring for students.

        **Assignment Headline:**
        **Assignment Overview:**
        **Assignment Instructions:**
        """
        
        response = model.generate_content(prompt)
        time.sleep(8) # 8-second delay
        
        # Robust parsing of the response
        text = response.text
        headline_match = re.search(r"Assignment Headline:(.*?)Assignment Overview:", text, re.DOTALL)
        overview_match = re.search(r"Assignment Overview:(.*?)Assignment Instructions:", text, re.DOTALL)
        instructions_match = re.search(r"Assignment Instructions:(.*)", text, re.DOTALL)

        if headline_match and overview_match and instructions_match:
            return {
                "headline": headline_match.group(1).strip(),
                "overview": overview_match.group(1).strip(),
                "instructions": instructions_match.group(1).strip()
            }
        else:
            # Fallback if regex fails
            st.warning("Could not parse Gemini response perfectly, using fallback.")
            parts = text.split("\n")
            return {
                "headline": parts[0].replace("**Assignment Headline:**", "").strip(),
                "overview": parts[1].replace("**Assignment Overview:**", "").strip(),
                "instructions": " ".join(parts[2:]).replace("**Assignment Instructions:**", "").strip(),
            }

    except Exception as e:
        st.error(f"Failed to generate assignment. Error: {e}")
        return None

# --- Page Rendering Functions ---

def render_home_page():
    """Renders the main input form and results display."""
    st.title("📚 Next Assignment Generator")
    st.markdown("Enter a subject and three key skills to discover relevant project assignments based on recent news.")

    with st.form("assignment_form"):
        user_subject = st.text_input("Enter Subject Name", max_chars=30, key="user_subject")
        skill_1 = st.text_input("Enter Skill 1", max_chars=30, key="skill_1")
        skill_2 = st.text_input("Enter Skill 2", max_chars=30, key="skill_2")
        skill_3 = st.text_input("Enter Skill 3", max_chars=30, key="skill_3")
        submitted = st.form_submit_button("Find Assignment")

    if submitted:
        if not all([user_subject, skill_1, skill_2, skill_3]):
            st.error("All fields are compulsory. Please fill them all out.")
        else:
            # --- Start the main process ---
            st.session_state.assignments = [] # Clear old results
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            timer_placeholder = st.empty()
            start_time = time.time()

            # 1. Fetching from Database
            status_placeholder.info("Step 1/6: Fetching from Database...")
            timer_placeholder.text("Elapsed time: 0s")
            conn = connect_db()
            if not conn: st.stop()
            cur = conn.cursor()
            cur.execute("SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db;")
            records = cur.fetchall()
            conn.close()
            progress_bar.progress(5)

            # 2. Merging Keywords
            status_placeholder.info("Step 2/6: Merging keywords...")
            user_str = f"{user_subject}, {skill_1}, {skill_2}, {skill_3}"
            db_data = []
            for rec in records:
                # Filter out None values before joining
                keywords = [rec[2], rec[3], rec[4], rec[5]]
                merged_str = ", ".join(filter(None, keywords))
                db_data.append({
                    "id": rec[0],
                    "url": rec[1],
                    "str": merged_str
                })
            progress_bar.progress(10)

            # 3. Getting Embeddings
            status_placeholder.info("Step 3/6: Getting Embeddings...")
            total_items = len(db_data) + 1
            for i, item in enumerate(db_data):
                elapsed_time = f"{int(time.time() - start_time)}s"
                timer_placeholder.text(f"Elapsed time: {elapsed_time}")
                status_placeholder.info(f"Step 3/6: Getting Embeddings... ({i+1}/{total_items})")
                item['embedding'] = get_embedding(item['str'])
                if item['embedding'] is None:
                    st.error("Process stopped due to a critical embedding error.")
                    st.stop()
                progress_bar.progress(10 + int(60 * (i + 1) / total_items))
            
            user_str_embd = get_embedding(user_str)
            if user_str_embd is None:
                st.error("Failed to process user input. Please try again.")
                st.stop()
            progress_bar.progress(70)

            # 4. Finding Cosine Similarity
            status_placeholder.info("Step 4/6: Finding Cosine Similarity...")
            db_embeddings = np.array([item['embedding'] for item in db_data])
            user_embedding = np.array(user_str_embd).reshape(1, -1)
            similarities = cosine_similarity(user_embedding, db_embeddings)[0]
            
            for i, item in enumerate(db_data):
                item['score'] = similarities[i]

            # Get top 3 indices
            top_3_indices = np.argsort(similarities)[-3:][::-1]
            top_3_records = [db_data[i] for i in top_3_indices]
            progress_bar.progress(75)

            # 5. Fetching Full News from Database
            status_placeholder.info("Step 5/6: Fetching News from Database...")
            conn = connect_db()
            if not conn: st.stop()
            cur = conn.cursor()
            top_ids = tuple(rec['id'] for rec in top_3_records)
            cur.execute("SELECT id, headline, news, impact, emotion FROM news_db WHERE id IN %s;", (top_ids,))
            news_details = cur.fetchall()
            conn.close()
            
            news_map = {detail[0]: {'headline': detail[1], 'news': detail[2], 'impact': detail[3], 'emotion': detail[4]} for detail in news_details}
            
            for rec in top_3_records:
                rec.update(news_map.get(rec['id'], {}))
            progress_bar.progress(80)

            # 6. Generating Assignments
            status_placeholder.info("Step 6/6: Generating Assignments...")
            final_assignments = []
            for i, rec in enumerate(top_3_records):
                status_placeholder.info(f"Step 6/6: Generating Assignment {i+1}/3...")
                elapsed_time = f"{int(time.time() - start_time)}s"
                timer_placeholder.text(f"Elapsed time: {elapsed_time}")
                
                prompt_data = {
                    'headline': rec['headline'], 'news': rec['news'], 'impact': rec['impact'], 
                    'emotion': rec['emotion'], 'subject': user_subject, 'skills': [skill_1, skill_2, skill_3]
                }
                assignment_content = generate_assignment_from_gemini(prompt_data)
                
                if assignment_content:
                    rec.update({
                        'ass_headline': assignment_content['headline'],
                        'ass_overview': assignment_content['overview'],
                        'ass_instructions': assignment_content['instructions'],
                    })
                    final_assignments.append(rec)
                progress_bar.progress(80 + int(20 * (i + 1) / 3))

            st.session_state.assignments = final_assignments
            status_placeholder.success("🎉 All assignments generated successfully!")
            progress_bar.empty()
            timer_placeholder.empty()

    if 'assignments' in st.session_state and st.session_state.assignments:
        st.markdown("---")
        st.subheader("Your Custom Assignments")
        
        cols = st.columns(len(st.session_state.assignments))
        for i, assignment in enumerate(st.session_state.assignments):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{assignment['ass_headline']}**")
                    if st.button("View Details", key=f"details_{i}"):
                        st.session_state.selected_assignment_index = i
                        st.session_state.page = "details"
                        st.rerun()

def render_details_page():
    """Renders the details of a selected assignment."""
    if 'selected_assignment_index' not in st.session_state:
        st.session_state.page = "home"
        st.rerun()
    
    assignment = st.session_state.assignments[st.session_state.selected_assignment_index]
    
    st.title("Assignment Details")
    st.markdown("---")

    st.header(assignment.get('ass_headline', 'No Headline'))
    
    with st.container(border=True):
        st.subheader("📋 Overview")
        st.write(assignment.get('ass_overview', 'No overview available.'))

        st.subheader("📝 Instructions")
        st.write(assignment.get('ass_instructions', 'No instructions available.'))
        
        st.subheader("🔗 Reference URL")
        st.markdown(f"[Read the original news article]({assignment.get('url', '#')})")
        
        st.subheader("🆔 Assignment ID")
        st.code(assignment.get('id', 'N/A'))

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# --- Main App Logic ---

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Page router
if st.session_state.page == 'home':
    render_home_page()
else:
    render_details_page()
