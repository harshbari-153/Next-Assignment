import streamlit as st
import os
import time
import datetime
import psycopg2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Next Assignment",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- LOAD ENVIRONMENT VARIABLES ---
# Load from the parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
GEMINI_API_KEYS = [
    os.getenv("Gemini_API_1"),
    os.getenv("Gemini_API_2"),
    os.getenv("Gemini_API_3"),
]

# --- SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'assignments' not in st.session_state:
    st.session_state.assignments = None
if 'selected_assignment_id' not in st.session_state:
    st.session_state.selected_assignment_id = None
if 'api_key_index' not in st.session_state:
    st.session_state.api_key_index = 0

# --- STYLING ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background-color: #4F8BF9;
        color: white;
        border: none;
        padding: 10px 0;
    }
    .stButton>button:hover {
        background-color: #3A6DC2;
        color: white;
    }
    .assignment-button {
        text-align: left !important;
        padding: 15px 20px !important;
        background-color: #f0f2f6 !important;
        border: 1px solid #dcdcdc !important;
        border-radius: 10px !important;
        color: #333 !important;
        width: 100% !important;
        margin-bottom: 10px !important;
    }
    .assignment-button:hover {
        background-color: #e6e8eb !important;
        border-color: #4F8BF9 !important;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(SUPABASE_URL)
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

def rotate_api_key():
    """Rotates to the next Gemini API key."""
    st.session_state.api_key_index = (st.session_state.api_key_index + 1) % len(GEMINI_API_KEYS)
    return GEMINI_API_KEYS[st.session_state.api_key_index]

def get_embedding(text):
    """Gets embeddings for a given text using a rotating Gemini API key."""
    time.sleep(50) # 50-second delay as requested
    try:
        api_key = rotate_api_key()
        genai.configure(api_key=api_key)
        model = 'models/embedding-001'
        embedding = genai.embed_content(model=model, content=text, task_type="RETRIEVAL_DOCUMENT")
        return embedding['embedding']
    except Exception as e:
        st.warning(f"Embedding failed for a record, skipping. Error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def parse_assignment_response(response_text):
    """Parses the generated assignment text into structured parts."""
    try:
        headline = response_text.split("Assignment Headline:")[1].split("Assignment Overview:")[0].strip()
        overview = response_text.split("Assignment Overview:")[1].split("Assignment Instructions:")[0].strip()
        instructions = response_text.split("Assignment Instructions:")[1].strip()
        return headline, overview, instructions
    except IndexError:
        # Fallback if the model doesn't follow the format strictly
        return "Could not parse headline.", "Could not parse overview.", response_text

# --- CORE LOGIC ---

def find_assignments(user_subject, skill_1, skill_2, skill_3):
    """The main function to perform all steps from data fetching to assignment generation."""
    start_time = time.time()
    status_container = st.empty()
    progress_bar = st.progress(0)
    timer_container = st.empty()

    try:
        # --- STEP 1: Fetch initial data from Database ---
        status_container.info("⏳ Step 1/6: Fetching records from database...")
        timer_container.caption(f"Elapsed Time: 0s")
        conn = get_db_connection()
        if not conn: return
        with conn.cursor() as cur:
            cur.execute("SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db;")
            records = cur.fetchall()
        conn.close()
        progress_bar.progress(5)

        # --- STEP 2: Merge keywords ---
        status_container.info("🔄 Step 2/6: Merging keywords...")
        db_data = []
        for rec in records:
            # Ensure all parts are strings before joining
            parts = [str(p) for p in [rec[2], rec[3], rec[4], rec[5]] if p]
            merged_str = ", ".join(parts)
            db_data.append({'id': rec[0], 'url': rec[1], 'str': merged_str})

        user_str = f"{user_subject}, {skill_1}, {skill_2}, {skill_3}"
        progress_bar.progress(10)

        # --- STEP 3: Get Embeddings ---
        total_items = len(db_data) + 1
        status_container.info(f"✨ Step 3/6: Getting embeddings (0/{total_items})...")
        
        for i, item in enumerate(db_data):
            elapsed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            timer_container.caption(f"Elapsed Time: {elapsed_time}")
            status_container.info(f"✨ Step 3/6: Getting embeddings ({i+1}/{total_items})... This takes a while.")
            
            item['embd'] = get_embedding(item['str'])
            progress_bar.progress(10 + int((i+1) / total_items * 40))

        # Filter out items where embedding failed
        db_data = [item for item in db_data if item.get('embd') is not None]
        if not db_data:
            st.error("Could not generate embeddings for any database records. Please check Gemini API keys.")
            return

        user_embd = get_embedding(user_str)
        if not user_embd:
            st.error("Could not generate embedding for your input. Please check Gemini API keys and try again.")
            return
        progress_bar.progress(50)

        # --- STEP 4: Find Cosine Similarity ---
        status_container.info("⚖️ Step 4/6: Finding cosine similarity...")
        for item in db_data:
            item['score'] = cosine_similarity(item['embd'], user_embd)

        # Get top 3
        top_3 = sorted(db_data, key=lambda x: x['score'], reverse=True)[:3]
        top_3_ids = [item['id'] for item in top_3]
        
        if not top_3_ids:
            st.warning("No relevant articles found to generate assignments. Try different skills or subjects.")
            return
            
        progress_bar.progress(60)

        # --- STEP 5: Fetch full news for top 3 ---
        status_container.info("📰 Step 5/6: Fetching news details for top results...")
        conn = get_db_connection()
        if not conn: return
        with conn.cursor() as cur:
            cur.execute("SELECT id, headline, news, impact, emotion FROM news_db WHERE id = ANY(%s);", (top_3_ids,))
            news_details = cur.fetchall()
        conn.close()

        # Map details back to top_3 items
        details_map = {rec[0]: {'headline': rec[1], 'news': rec[2], 'impact': rec[3], 'emotion': rec[4]} for rec in news_details}
        for item in top_3:
            item.update(details_map.get(item['id'], {}))
        
        progress_bar.progress(70)
        
        # --- STEP 6: Generate Assignments ---
        status_container.info("🤖 Step 6/6: Generating assignments...")
        genai.configure(api_key=rotate_api_key())
        generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
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
        2. **Focus on skills:** The project must require students to demonstrate and apply the skills within the {subject} context.
        3. **Structure the output strictly as shown below with the exact labels:**
           - **Assignment Headline:** A title, max 10 words.
           - **Assignment Overview:** A brief summary of the project's goal, max 20 words.
           - **Assignment Instructions:** The steps students must follow, max 40 words.
        4. **Tone:** The assignment should be relevant to emotion, and inspiring for students.
        
        **Assignment Headline:**
        **Assignment Overview:**
        **Assignment Instructions:**
        """

        for i, item in enumerate(top_3):
            elapsed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            timer_container.caption(f"Elapsed Time: {elapsed_time}")
            status_container.info(f"🤖 Step 6/6: Generating Assignment {i+1} of 3...")
            time.sleep(50) # 50-second delay

            prompt = prompt_template.format(
                headline=item.get('headline', 'N/A'),
                news=item.get('news', 'N/A'),
                impact=item.get('impact', 'N/A'),
                emotion=item.get('emotion', 'N/A'),
                subject=user_subject,
                skills=user_str
            )
            try:
                response = generation_model.generate_content(prompt)
                headline, overview, instructions = parse_assignment_response(response.text)
                item['ass_headline'] = headline
                item['ass_overview'] = overview
                item['ass_instructions'] = instructions
            except Exception as e:
                item['ass_headline'] = "Error Generating Assignment"
                item['ass_overview'] = f"An error occurred: {e}"
                item['ass_instructions'] = "Please try again with different inputs."

            progress_bar.progress(70 + int((i+1)/3 * 30))

        st.session_state.assignments = top_3
        status_container.success("✅ All assignments generated successfully!")
        timer_container.empty()
        progress_bar.empty()

    except Exception as e:
        status_container.error(f"An unexpected error occurred: {e}")
        timer_container.empty()
        progress_bar.empty()


# --- UI RENDERING FUNCTIONS ---

def render_home_page():
    """Renders the main input form and results."""
    st.title("📚 Next Assignment Generator")
    st.caption("Enter a subject and three skills to discover relevant, real-world project assignments based on current news.")

    with st.form(key="assignment_form"):
        user_subject = st.text_input("Enter Subject Name", max_chars=30, placeholder="e.g., Data Science")
        skill_1 = st.text_input("Enter Skill 1", max_chars=30, placeholder="e.g., Python")
        skill_2 = st.text_input("Enter Skill 2", max_chars=30, placeholder="e.g., NLP")
        skill_3 = st.text_input("Enter Skill 3", max_chars=30, placeholder="e.g., Visualization")
        
        submit_button = st.form_submit_button(label="Find Assignment")

    if submit_button:
        if all([user_subject, skill_1, skill_2, skill_3]):
            st.session_state.assignments = None # Clear old results
            find_assignments(user_subject, skill_1, skill_2, skill_3)
        else:
            st.warning("Please fill out all four fields.")

    if st.session_state.assignments:
        st.markdown("---")
        st.subheader("Your Custom Assignments:")
        for assignment in st.session_state.assignments:
            if st.button(assignment.get('ass_headline', 'Untitled Assignment'), key=f"btn_{assignment['id']}", help="Click to see details", use_container_width=True):
                # Using a custom class for styling is not directly supported by st.button,
                # but we can apply global styles as done at the top.
                # Here, we'll just handle the navigation logic.
                st.session_state.selected_assignment_id = assignment['id']
                st.session_state.page = 'details'
                st.rerun()

def render_details_page():
    """Renders the detailed view of a selected assignment."""
    selected_id = st.session_state.selected_assignment_id
    assignment = next((a for a in st.session_state.assignments if a['id'] == selected_id), None)

    if assignment:
        st.title(assignment.get('ass_headline', 'Assignment Details'))
        st.markdown("---")
        
        st.subheader("📌 Overview")
        st.write(assignment.get('ass_overview', 'Not available.'))

        st.subheader("📋 Instructions")
        st.write(assignment.get('ass_instructions', 'Not available.'))
        
        st.subheader("🔗 Source URL")
        st.write(assignment.get('url', 'Not available.'))

        st.info(f"**Assignment ID:** `{assignment.get('id', 'N/A')}`")

    else:
        st.error("Could not find assignment details. Please go back.")

    if st.button("⬅️ Back to All Assignments"):
        st.session_state.page = 'home'
        st.session_state.selected_assignment_id = None
        st.rerun()


# --- MAIN APP ROUTER ---
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'details':
    render_details_page()
