
#from dotenv import load_dotenv

#################################################
def find_latest_headline_and_url(API: str):
    if not isinstance(API, str) or len(API) != 32:
        raise ValueError("API key must be a 32-character string.")

    url = f"https://gnews.io/api/v4/top-headlines?category=general&lang=en&country=in&max=5&apikey={API}"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    articles = data.get("articles", [])
    result = []

    for article in articles:
        published_at = article.get("publishedAt", "")
        timestamp = published_at.replace("T", " ")[:-1] if "T" in published_at and published_at.endswith("Z") else published_at

        result.append({
            "id": article.get("id"),
            "headline": article.get("title"),
            "url": article.get("url"),
            "timestamp": timestamp
        })


    return result
#################################################


#################################################
def get_body(news_list):
    for news in news_list:
        try:
            response = requests.get(news["url"], timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Collect all paragraph texts
            paragraphs = soup.find_all("p")
            body_text = " ".join(p.get_text(strip=True) for p in paragraphs)

            news["body"] = body_text.strip()
        except Exception as e:
            news["body"] = f"Error fetching content: {e}"

    return news_list

#################################################



#################################################
def get_metadata(news_list, Gemini_API: str):

    if not isinstance(Gemini_API, str) or len(Gemini_API) != 39:
        raise ValueError("Gemini_API must be a 39-character string.")

    # Configure the API
    genai.configure(api_key=Gemini_API)

    # Initialize model
    model = genai.GenerativeModel("gemini-2.5-flash")
    # or "gemini-2.5-flash" or "gemini-pro" or "gemini-1.5-pro"

    # Prompt Template
    prompt_template = '''You are an expert news summarizer. Your task is to analyze the following news text and provide a structured summary based on the specified JSON format.

The structured output must be in a single JSON object with the following keys:
- "news": A concise summary of the main content, no more than 50 words.
- "impact": A brief description of the impact of the news, no more than 40 words.
- "news_type": The type of news, in 1 or 2 words (e.g., "Politics," "Technology," "Finance").
- "emotion": A single word describing the primary emotion the news evokes in a reader (e.g., "Sadness," "Joy," "Anger," "Excitement").
- "key_1": A single, relevant keyword.
- "key_2": A single, relevant keyword.
- "key_3": A single, relevant keyword.

The output must contain only the JSON object, with no other text, explanations, or code block formatting.

Text to summarize:
\"\"\"
{body_text}
\"\"\"
'''

    # Process each news item
    for news in news_list:
        body_text = news.get("body", "").strip()

        if not body_text:
            # If no body, skip processing
            news.update({
                "news": "",
                "impact": "",
                "news_type": "",
                "emotion": "",
                "key_1": "",
                "key_2": "",
                "key_3": ""
            })
            continue

        # Build final prompt
        prompt = prompt_template.format(body_text=body_text)

        try:
            # Generate response
            response = model.generate_content(prompt)
            output_text = response.text.strip()

            # Parse JSON from Gemini response
            metadata = json.loads(output_text)

            # Add metadata to news dictionary
            news.update({
                "news": metadata.get("news", ""),
                "impact": metadata.get("impact", ""),
                "news_type": metadata.get("news_type", ""),
                "emotion": metadata.get("emotion", ""),
                "key_1": metadata.get("key_1", ""),
                "key_2": metadata.get("key_2", ""),
                "key_3": metadata.get("key_3", "")
            })

        except Exception as e:
            # Handle any parsing or API issues
            news.update({
                "news": f"Error: {str(e)}",
                "impact": "",
                "news_type": "",
                "emotion": "",
                "key_1": "",
                "key_2": "",
                "key_3": ""
            })

    return news_list
#################################################




#################################################
def add_to_database(news_list, database_url: str):

    # Remove 'body' from each news item (not needed in DB)
    for news in news_list:
         news.pop("body", None)

    # Extract column names from first dictionary
    columns = news_list[0].keys()
    column_names = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))

    # Convert list of dicts into list of tuples (ordered by columns)
    values = [tuple(news[col] for col in columns) for news in news_list]


    try:
        # Connect to Supabase
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()

        # Insert all news items in one query
        insert_query = f"INSERT INTO news_db ({column_names}) VALUES %s"
        execute_values(cursor, insert_query, values)

        # Delete oldest records if count > 30
        while True:
            cursor.execute("SELECT COUNT(*) FROM news_db")
            count = cursor.fetchone()[0]

            if count <= 30:
                break

            # Delete the oldest record based on timestamp
            cursor.execute("""
                DELETE FROM news_db
                WHERE ctid IN (
                    SELECT ctid FROM news_db
                    ORDER BY timestamp ASC
                    LIMIT 1
                )
            """)

        # Commit changes and close
        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Database error: {e}")
#################################################




#################### Testing ####################
'''
# Specify the path to your .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

GNews_API = os.getenv("GNews_API")
Gemini_API = os.getenv("Gemini_API")
PostgreSQL_API = os.getenv("PostgreSQL_API")


ans_1 = find_latest_headline_and_url(GNews_API)
with open("latest_news_1.txt", "w", encoding="utf-8") as f:
    f.write(str(ans_1))
print("News Fetched")


ans_2 = get_body(ans_1)
with open("latest_news_2.txt", "w", encoding="utf-8") as f:
    f.write(str(ans_2))
print("Content Fetched")


ans_3 = get_metadata(ans_2, Gemini_API)
with open("latest_news_3.txt", "w", encoding="utf-8") as f:
    f.write(str(ans_3))
print("Meta Data Fetched")

add_to_database(ans_3, PostgreSQL_API)
print("Inserted to database")
'''
#################################################