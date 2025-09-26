import streamlit as st
import os
import time
import requests
import psycopg2
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIGURATIONS AND INITIAL SETUP ---

# Load environment variables from .env file located in the parent directory
load_dotenv(dotenv_path='../.env') 

# Configure Streamlit page
st.set_page_config(
    page_title="Next Assignment",
    page_icon="📚",
    layout="centered"
)

# Function to apply some light-weight custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline styles if a css file is not used
        st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .stTextInput>div>div>input {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply styling
local_css("style.css") # Optional: you can create a style.css file in the app folder for more complex styling

# --- 2. CACHED FUNCTIONS & RESOURCE LOADING ---

@st.cache_resource
def load_sentence_model():
    """Loads the SentenceTransformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_image_from_url(url):
    """Fetches the first image from a URL, with caching."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Prioritize Open Graph image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
            
        # Fallback to the first found image
        first_image = soup.find('img')
        if first_image and first_image.get('src'):
            # Handle relative URLs
            src = first_image.get('src')
            if src.startswith(('http://', 'https://')):
                return src
            else:
                from urllib.parse import urljoin
                return urljoin(url, src)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
    
    # Return a default placeholder if no image is found
    return "https://via.placeholder.com/500x250.png?text=No+Image+Found"


# Load the model once
model = load_sentence_model()

# --- 3. DATABASE AND API FUNCTIONS ---

def get_db_connection():
    """Establishes and returns a new database connection."""
    try:
        connection = psycopg2.connect(os.environ["NEON_DATABASE"])
        return connection
    except psycopg2.OperationalError as e:
        st.error(f"🚨 Database Connection Error: Could not connect to the database. Please check credentials. Details: {e}")
        return None

def fetch_initial_data():
    """Fetches required fields from the database, excluding 'Technical Error' types."""
    conn = get_db_connection()
    if conn is None: return []
    
    records = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, url, news_type, key_1, key_2, key_3 FROM news_db WHERE news_type != 'Technical Error'"
            )
            records = cur.fetchall()
    except psycopg2.Error as e:
        st.error(f"🚨 Database Query Error: Failed to fetch initial data. Details: {e}")
    finally:
        conn.close()
    
    # Convert list of tuples to list of dictionaries
    data = [
        {'id': r[0], 'url': r[1], 'news_type': r[2], 'key_1': r[3], 'key_2': r[4], 'key_3': r[5]}
        for r in records
    ]
    return data

def fetch_news_details(ids):
    """Fetches detailed news content for a list of IDs."""
    if not ids: return {}
    conn = get_db_connection()
    if conn is None: return {}

    details = {}
    try:
        with conn.cursor() as cur:
            # Use tuple(ids) for the IN clause
            cur.execute(
                "SELECT id, headline, news, impact, emotion FROM news_db WHERE id IN %s", (tuple(ids),)
            )
            records = cur.fetchall()
            for r in records:
                details[r[0]] = {'headline': r[1], 'news': r[2], 'impact': r[3], 'emotion': r[4]}
    except psycopg2.Error as e:
        st.error(f"🚨 Database Query Error: Failed to fetch news details. Details: {e}")
    finally:
        conn.close()
    return details

def get_gemini_assignment(api_key, prompt):
    """Calls the Gemini API to generate an assignment and handles potential errors."""
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-002') # Updated model name
        response = gemini_model.generate_content(prompt)
        
        # Simple parsing of the response based on the requested format
        text = response.text
        ass_headline = text.split("Assignment Headline:")[1].split("Assignment Overview:")[0].strip()
        ass_overview = text.split("Assignment Overview:")[1].split("Assignment Instructions:")[0].strip()
        ass_instructions = text.split("Assignment Instructions:")[1].strip()

        if not all([ass_headline, ass_overview, ass_instructions]):
             raise ValueError("Malformed response from Gemini API.")

        return {
            "ass_headline": ass_headline,
            "ass_overview": ass_overview,
            "ass_instructions": ass_instructions
        }
    except Exception as e:
        print(f"Gemini API Error: {e}")
        st.warning(f"⚠️ Could not generate an assignment. Error: {e}", icon="🤖")
        return None


# --- 4. MAIN APPLICATION UI AND LOGIC ---

st.title("📚 Next Assignment")
st.markdown("Enter a subject and three skills to discover relevant, real-world project assignments based on current news.")

# Initialize session state for assignments and API key rotation
if 'assignments' not in st.session_state:
    st.session_state.assignments = []
if 'gemini_key_index' not in st.session_state:
    st.session_state.gemini_key_index = 0

st.markdown("""
    <style>
    input {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a form for user inputs
with st.form(key="assignment_form"):
    user_subject = st.text_input(
        "Enter the name of the subject",
        max_chars=30,
        placeholder="e.g., Data Science"
    )
    skill_1 = st.text_input(
        "Enter a required skill",
        max_chars=30,
        placeholder="e.g., Python"
    )
    skill_2 = st.text_input(
        "Enter a second skill",
        max_chars=30,
        placeholder="e.g., NLP"
    )
    skill_3 = st.text_input(
        "Enter a third skill",
        max_chars=30,
        placeholder="e.g., Classification"
    )
    submitted = st.form_submit_button(label="🚀 Find Assignment")


if submitted:
    # Validate inputs
    if not all([user_subject, skill_1, skill_2, skill_3]):
        st.error("All fields are compulsory. Please fill them all out.")
    else:
        # Clear previous results
        st.session_state.assignments = []
        final_assignments = []
        
        # Use st.status to show progress of the background tasks
        with st.status("🚀 Launching assignment search...", expanded=True) as status:
            
            # Step 1: Fetch from Database
            status.update(label="Step 1/6: Fetching records from database...")
            all_records = fetch_initial_data()

            if not all_records:
                status.update(label="No records found in the database. Please check the database connection or content.", state="error")
            else:
                # Step 2: Merge Keywords
                status.update(label="Step 2/6: Merging keywords...")
                user_str = f"{user_subject}, {skill_1}, {skill_2}, {skill_3}"
                for record in all_records:
                    record['str'] = f"{record.get('news_type', '')}, {record.get('key_1', '')}, {record.get('key_2', '')}, {record.get('key_3', '')}"

                # Step 3: Get Embeddings
                status.update(label="Step 3/6: Generating text embeddings...")
                news_strings = [rec['str'] for rec in all_records]
                all_strings = [user_str] + news_strings
                embeddings = model.encode(all_strings, show_progress_bar=False)
                
                user_embd = embeddings[0]
                news_embds = embeddings[1:]
                for i, record in enumerate(all_records):
                    record['embd'] = news_embds[i]

                # Step 4: Find Cosine Similarity
                status.update(label="Step 4/6: Calculating similarity scores...")
                cosine_scores = util.pytorch_cos_sim(user_embd, news_embds)[0].cpu().numpy()
                for i, record in enumerate(all_records):
                    record['score'] = cosine_scores[i]
                
                # Sort by score and get top 3
                sorted_records = sorted(all_records, key=lambda x: x['score'], reverse=True)
                top_n = min(3, len(sorted_records))
                top_articles = sorted_records[:top_n]
                
                if top_n == 0:
                     status.update(label="Could not find any relevant news articles.", state="error")
                else:
                    top_ids = [article['id'] for article in top_articles]
                    
                    # Step 5: Fetch Full News for Top 3
                    status.update(label=f"Step 5/6: Fetching full details for top {top_n} articles...")
                    news_details = fetch_news_details(top_ids)
                    
                    # Merge details into top_articles
                    for article in top_articles:
                        if article['id'] in news_details:
                            article.update(news_details[article['id']])
                    
                    # Step 6: Generate Assignments with Gemini
                    gemini_apis = [
                        os.getenv("GEMINI_API_1"),
                        os.getenv("GEMINI_API_2"),
                        os.getenv("GEMINI_API_3")
                    ]
                    
                    for i, article in enumerate(top_articles):
                        status.update(label=f"Step 6/6: Generating assignment {i+1}/{top_n}...")
                        
                        prompt = f"""
                        You are a project generator. Your task is to create an excellent project assignment based on the provided variables.

                        Variables:
                        headline: {article.get('headline')}
                        news: {article.get('news')}
                        impact: {article.get('impact')}
                        emotion: {article.get('emotion')}
                        subject: {user_subject}
                        skills: {skill_1}, {skill_2}, {skill_3}

                        Instructions:
                        1. **Integrate all variables:** The assignment must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
                        2. **Focus on skills:** The project must require students to demonstrate and apply the skills {skill_1}, {skill_2}, and {skill_3} within the {user_subject} context.
                        3. **Structure the output strictly:**
                        - **Assignment Headline:** A title, max 10 words.
                        - **Assignment Overview:** A brief summary of the project's goal, max 30 words.
                        - **Assignment Instructions:** The steps students must follow, max 60 words.
                        4. **Tone:** The assignment should be relevant to the emotion: '{article.get('emotion')}', and inspiring for students.

                        **Assignment Headline:**
                        **Assignment Overview:**
                        **Assignment Instructions:**
                        """
                        
                        # Rotate API keys
                        api_key = gemini_apis[st.session_state.gemini_key_index]
                        st.session_state.gemini_key_index = (st.session_state.gemini_key_index + 1) % len(gemini_apis)

                        assignment_parts = get_gemini_assignment(api_key, prompt)
                        if assignment_parts:
                            article.update(assignment_parts)
                            final_assignments.append(article)
                        
                        # Delay between API calls
                        if i < top_n - 1:
                            time.sleep(5)

            status.update(label="✅ All steps completed!", state="complete", expanded=False)
            
        # Store final results in session state
        st.session_state.assignments = final_assignments

# --- 5. DISPLAY RESULTS ---

if st.session_state.assignments:
    st.markdown("---")
    st.subheader("Generated Assignments ✨")
    
    # The image fetching and display happens here
    for assignment in st.session_state.assignments:
        with st.container(border=True):
            # Assignment Headline generated by Gemini
            st.header(assignment.get("ass_headline", "Assignment Headline Not Available"))
            
            # Fetch and display image with news headline as caption
            image_url = get_image_from_url(assignment['url'])
            st.image(image_url, caption=f"Source: {assignment.get('headline', 'News Headline')}")
            
            # Display assignment details
            st.markdown(f"**Overview:** {assignment.get('ass_overview', 'N/A')}")
            st.markdown(f"**Instructions:** {assignment.get('ass_instructions', 'N/A')}")
            
            # Link to the original news article
            st.markdown(f"For detailed news, visit: [Link]({assignment['url']})")

elif submitted: # Show this only if the button was pressed but no results were found
    st.warning("Could not generate any assignments. This might be due to a lack of relevant news or an API issue.")
