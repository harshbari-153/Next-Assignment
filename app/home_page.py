import time
import os
import random
import math
import traceback
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import numpy as np

# Prefer the new google-genai client; fall back to google-generativeai if needed.
USE_NEW_CLIENT = True
try:
    from google import genai as genai_new  # google-genai
except Exception:
    USE_NEW_CLIENT = False
try:
    import google.generativeai as genai_old  # google-generativeai
except Exception:
    pass

load_dotenv()

# --------- Styling (lightweight) ---------
st.set_page_config(page_title="Next Assignment", page_icon="🧭", layout="centered")
st.markdown("""
<style>
.small {font-size: 0.9rem; color:#444;}
.step {padding:8px 12px; border:1px solid #e5e5e5; border-radius:8px; margin:8px 0; background:#fafafa;}
.ok {color:#0a7f2e;}
.warn {color:#b26a00;}
.err {color:#a10000;}
.card {border:1px solid #e6e6e6; border-radius:10px; padding:14px; margin:8px 0; background:white;}
h1, h2, h3 {color:#202124;}
:root {--accent:#1a73e8;}
.stProgress > div > div > div > div {background-color: var(--accent);}
.assign-btn {border:none; background:#f1f3f4; padding:10px 12px; border-radius:8px; cursor:pointer;}
.assign-btn:hover {background:#e7ebee;}
</style>
""", unsafe_allow_html=True)

# --------- Env and keys ---------
PG_URL = os.getenv("PostgreSQL_API_2", "").strip()
GEM_KEYS = list(filter(None, [
    os.getenv("Gemini_API_1", "").strip(),
    os.getenv("Gemini_API_2", "").strip(),
    os.getenv("Gemini_API_3", "").strip(),
]))
EMBED_MODEL = "gemini-embedding-001"  # text embeddings
GEN_MODEL = "gemini-2.5-flash"        # generation

API_CALL_DELAY_SEC = 8  # enforced delay
MAX_INPUT_LEN = 30

# --------- Session init ---------
def _init_session():
    ss = st.session_state
    ss.setdefault("db_rows", [])
    ss.setdefault("work_items", [])  # [{id, str, url, emb, score, ...}]
    ss.setdefault("assignments", []) # 3 dicts after generation
    ss.setdefault("user_subject", "")
    ss.setdefault("skill_1", "")
    ss.setdefault("skill_2", "")
    ss.setdefault("skill_3", "")
    ss.setdefault("user_str", "")
    ss.setdefault("timer_start", None)
    ss.setdefault("progress", 0.0)
    ss.setdefault("status_lines", [])
    ss.setdefault("last_api_index", -1)
    ss.setdefault("ready", False)
    ss.setdefault("nav_target_id", None)
    ss.setdefault("run_id", 0)  # increment per successful run to invalidate old data

_init_session()


# --------- Utilities ---------
def add_status(line, level="info"):
    # level: info/ok/warn/err
    color_class = {"info":"", "ok":"ok", "warn":"warn", "err":"err"}.get(level, "")
    st.session_state.status_lines.append((line, color_class))

def enforce_len(s):
    return s[:MAX_INPUT_LEN]

def rotate_api_key():
    # rotate across 3 keys such that no two consecutive calls reuse adjacent keys
    # Strategy: maintain circular index but skip the immediately previous index
    if not GEM_KEYS:
        raise RuntimeError("No Gemini API keys found in .env")
    n = len(GEM_KEYS)
    prev = st.session_state.last_api_index
    # pick next index that is not prev+1 mod n (not consecutive)
    candidates = [i for i in range(n)]
    # ensure strictly alternating 1->3->2->1 pattern isn't required; just avoid consecutive
    if prev >= 0:
        bad = (prev + 1) % n
        # prefer the farthest index
        candidates = [i for i in candidates if i != bad]
        # also avoid reusing prev immediately
        candidates = [i for i in candidates if i != prev] or [i for i in range(n) if i != bad]
    idx = candidates if candidates else (0 if prev != 0 else (1 % n))
    st.session_state.last_api_index = idx
    return GEM_KEYS[idx]

def new_genai_client(api_key):
    if USE_NEW_CLIENT:
        client = genai_new.Client(api_key=api_key)
        return ("new", client)
    else:
        genai_old.configure(api_key=api_key)
        return ("old", None)

def embed_text(api_key, text):
    mode, client = new_genai_client(api_key)
    if mode == "new":
        # google-genai
        res = client.models.embed_content(model=EMBED_MODEL, contents=text)
        # result.embeddings can be a single or list; normalize shape
        if hasattr(res, "embeddings"):
            emb = res.embeddings if isinstance(res.embeddings, list) else res.embeddings
            vec = getattr(emb, "values", None) or getattr(emb, "embedding", None) or emb
            return np.array(vec, dtype=np.float32)
        # Fallback structure
        raise RuntimeError("Malformed embedding response (new client)")
    else:
        # google-generativeai
        m = genai_old.GenerativeModel(EMBED_MODEL)
        res = m.embed_content(text)
        # Common structure: {'embedding': {'values': [...]}}
        if isinstance(res, dict):
            emb = res.get("embedding") or {}
            vals = emb.get("values") or emb.get("embedding") or []
            return np.array(vals, dtype=np.float32)
        # Some SDK versions: res.values
        vals = getattr(res, "values", None)
        if vals:
            return np.array(vals, dtype=np.float32)
        raise RuntimeError("Malformed embedding response (old client)")

def generate_assignment(api_key, prompt):
    mode, client = new_genai_client(api_key)
    if mode == "new":
        out = client.models.generate_content(model=GEN_MODEL, contents=prompt)
        # Attempt to extract plain text
        txt = getattr(out, "text", None)
        if not txt and hasattr(out, "candidates") and out.candidates:
            # fallback extraction
            parts = []
            for cand in out.candidates:
                ct = getattr(cand, "content", None)
                if ct and getattr(ct, "parts", None):
                    for p in ct.parts:
                        t = getattr(p, "text", None)
                        if t: parts.append(t)
            txt = "\n".join(parts).strip() if parts else ""
        return txt.strip()
    else:
        model = genai_old.GenerativeModel(GEN_MODEL)
        out = model.generate_content(prompt)
        txt = getattr(out, "text", None) or ""
        return txt.strip()

def cosine_sim(a, b):
    if a is None or b is None: return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return -1.0
    return float(np.dot(a, b) / (na * nb))

def parse_assignment_block(raw_text):
    # Expect strict sections; still robust to minor formatting noise.
    head, over, inst = "", "", ""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    cur = None
    buf = []
    def flush(c):
        return " ".join(buf).strip()
    for ln in lines:
        if ln.lower().startswith("**assignment headline"):
            if cur == "over": over = flush("over")
            if cur == "inst": inst = flush("inst")
            buf = []
            cur = "head"
            continue
        if ln.lower().startswith("**assignment overview"):
            if cur == "head": head = flush("head")
            if cur == "inst": inst = flush("inst")
            buf = []
            cur = "over"
            continue
        if ln.lower().startswith("**assignment instructions"):
            if cur == "head": head = flush("head")
            if cur == "over": over = flush("over")
            buf = []
            cur = "inst"
            continue
        buf.append(ln)
    # Final flush
    if cur == "head": head = flush("head")
    elif cur == "over": over = flush("over")
    elif cur == "inst": inst = flush("inst")

    # Fallback heuristic if markers were stripped
    if not head or not over or not inst:
        text = "\n".join(lines)
        # naive splits
        parts = text.split("Assignment Headline:")
        if len(parts) > 1:
            remainder = parts[1]
            seg_over = remainder.split("Assignment Overview:")
            head = seg_over.strip() if seg_over else remainder.strip()
            if len(seg_over) > 1:
                rem2 = seg_over[1]
                seg_inst = rem2.split("Assignment Instructions:")
                over = seg_inst.strip() if seg_inst else rem2.strip()
                if len(seg_inst) > 1:
                    inst = seg_inst[1].strip()
    return head[:90].strip(), over[:160].strip(), inst[:220].strip()

def guarded_call(fn, label_for_errors):
    # Wrap Gemini calls to catch frequent issues and return None on failure.
    try:
        return fn()
    except Exception as e:
        # Map common errors into labels
        msg = str(e)
        if any(k in msg.lower() for k in [
            "parse", "failure", "timeout", "time out", "runtime", "malformed", "429", "quota"
        ]):
            add_status(f"{label_for_errors} failed: {msg}", level="warn")
        else:
            add_status(f"{label_for_errors} error: {msg}", level="warn")
        return None

def connect_pg(url):
    # psycopg2 from connection string
    p = urlparse(url)
    conn_kwargs = {
        "dbname": p.path[1:],
        "user": p.username,
        "password": p.password,
        "port": p.port,
        "host": p.hostname,
        "sslmode": "require"
    }
    return psycopg2.connect(**conn_kwargs, cursor_factory=psycopg2.extras.DictCursor)

# --------- UI Header ---------
st.title("Next Assignment")  # home page
st.caption("Generate assignments tailored to subject and skills from real news.")  # [docs on multipage + session] not cited here as it's just UI

# --------- Inputs ---------
with st.form("input_form", clear_on_submit=False):
    user_subject = st.text_input("Subject name", max_chars=MAX_INPUT_LEN, key="user_subject", placeholder="e.g., data science")
    skill_1 = st.text_input("Skill 1", max_chars=MAX_INPUT_LEN, key="skill_1", placeholder="e.g., python")
    skill_2 = st.text_input("Skill 2", max_chars=MAX_INPUT_LEN, key="skill_2", placeholder="e.g., nlp")
    skill_3 = st.text_input("Skill 3", max_chars=MAX_INPUT_LEN, key="skill_3", placeholder="e.g., visualization")
    submitted = st.form_submit_button("Find Assignment", use_container_width=True)

# Timer visualization
timer_placeholder = st.empty()
progress_ph = st.empty()
status_box = st.container()

def render_status():
    with status_box:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.timer_start:
            elapsed = int(time.time() - st.session_state.timer_start)
            mm = elapsed // 60
            ss = elapsed % 60
            timer_placeholder.markdown(f"<div class='small'>⏱️ Elapsed: {mm:02d}:{ss:02d}</div>", unsafe_allow_html=True)
        st.progress(st.session_state.progress)
        for line, cls in st.session_state.status_lines[-10:]:
            st.markdown(f"<div class='step {cls}'>{line}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def reset_for_new_run():
    ss = st.session_state
    ss.db_rows = []
    ss.work_items = []
    ss.assignments = []
    ss.user_str = ""
    ss.timer_start = None
    ss.progress = 0.0
    ss.status_lines = []
    ss.last_api_index = -1
    ss.ready = False
    ss.nav_target_id = None

# --------- Run Pipeline ---------
if submitted:
    # Validate inputs
    vals = [st.session_state.user_subject, st.session_state.skill_1, st.session_state.skill_2, st.session_state.skill_3]
    if any(not v or not v.strip() for v in vals):
        st.error("All four fields are required.")
    elif any(len(v.strip()) > MAX_INPUT_LEN for v in vals):
        st.error(f"Each input must be ≤ {MAX_INPUT_LEN} characters.")
    elif not PG_URL or len(GEM_KEYS) < 3:
        st.error("Missing PostgreSQL_API or fewer than 3 Gemini API keys in .env")
    else:
        reset_for_new_run()
        st.session_state.run_id += 1
        rid = st.session_state.run_id
        st.session_state.timer_start = time.time()
        add_status("Fetching From Database")
        render_status()

        # Step 1: DB fetch
        try:
            with connect_pg(PG_URL) as conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT id, url, news_type, key_1, key_2, key_3
                    FROM news_db
                    ORDER BY id ASC
                """)
                rows = cur.fetchall()
                st.session_state.db_rows = [dict(r) for r in rows]
                add_status(f"Fetched {len(rows)} records", "ok")
        except Exception as e:
            add_status(f"Database fetch failed: {e}", "err")
            render_status()
            st.stop()
        st.session_state.progress = 0.1
        render_status()

        # Step 2: Merge strings
        add_status("Merging keywords")
        work = []
        for r in st.session_state.db_rows:
            merged = ", ".join([str(r.get("news_type", "")).strip()] + [str(r.get("key_1","")).strip(), str(r.get("key_2","")).strip(), str(r.get("key_3","")).strip()]).strip().strip(",")
            work.append({
                "id": r["id"],
                "url": r.get("url",""),
                "str": merged
            })
        st.session_state.user_str = ", ".join([
            st.session_state.user_subject.strip(),
            st.session_state.skill_1.strip(),
            st.session_state.skill_2.strip(),
            st.session_state.skill_3.strip()
        ])
        st.session_state.work_items = work
        add_status("Keywords merged", "ok")
        st.session_state.progress = 0.2
        render_status()

        # Step 3: Embeddings
        add_status("Getting Embeddings")
        # user embedding first
        def emb_user():
            k = rotate_api_key()
            return embed_text(k, st.session_state.user_str)
        user_emb = guarded_call(emb_user, "User embedding")
        if user_emb is None:
            add_status("Aborting due to embedding failure for user_str", "err")
            render_status()
            st.stop()
        last_tick = time.time()
        # enforce delay before next call
        time.sleep(max(0, API_CALL_DELAY_SEC - (time.time() - last_tick)))
        last_tick = time.time()
        # each record
        n = len(st.session_state.work_items)
        done = 0
        for item in st.session_state.work_items:
            def emb_one(it=item):
                k = rotate_api_key()
                return embed_text(k, it["str"])
            vec = guarded_call(emb_one, f"Embedding id={item['id']}")
            if vec is None:
                # Put a tiny non-zero vector to avoid zero-sim issues; but still allow similarity
                vec = np.random.normal(0, 1e-6, size=user_emb.shape).astype(np.float32)
            item["emb"] = vec
            done += 1
            frac = 0.2 + 0.5 * (done / max(1, n))  # progress up to 0.7
            st.session_state.progress = min(0.7, frac)
            add_status(f"Embedding {done}/{n} done")
            render_status()
            # 8s delay between API calls
            time.sleep(max(0, API_CALL_DELAY_SEC - (time.time() - last_tick)))
            last_tick = time.time()

        # Step 4: Cosine similarity
        add_status("Finding Cosine Similarity")
        for item in st.session_state.work_items:
            item["score"] = cosine_sim(item["emb"], user_emb)
        # Protect against all zeros leading to ties: shuffle before sort
        random.shuffle(st.session_state.work_items)
        st.session_state.work_items.sort(key=lambda x: x.get("score", -1.0), reverse=True)
        top3 = st.session_state.work_items[:3]
        st.session_state.progress = 0.8
        render_status()

        # Step 5: Fetch full news fields for top3
        add_status("Fetching News From Database")
        ids = tuple([t["id"] for t in top3])
        details_map = {}
        try:
            with connect_pg(PG_URL) as conn, conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, headline, news, impact, emotion
                    FROM news_db
                    WHERE id IN %s
                """, (ids,))
                for r in cur.fetchall():
                    details_map[r["id"]] = dict(r)
            # attach
            for t in top3:
                det = details_map.get(t["id"], {})
                t["headline"] = det.get("headline", "")
                t["news"] = det.get("news", "")
                t["impact"] = det.get("impact", "")
                t["emotion"] = det.get("emotion", "")
            add_status("News details attached", "ok")
        except Exception as e:
            add_status(f"Details fetch failed: {e}", "err")
            render_status()
            st.stop()
        st.session_state.progress = 0.86
        render_status()

        # Step 6: Generate assignments for top3
        # Build prompt template
        def make_prompt(t):
            return f"""{{
You are a project generator. Your task is to create an excellent project assignment based on the provided variables.
Variables:
headline: {t["headline"]}
news: {t["news"]}
impact: {t["impact"]}
emotion: {t["emotion"]}
subject: {st.session_state.user_subject}
skills: {st.session_state.user_str}
Instructions:
1.  **Integrate all variables:** The assignment must be a practical application that uses the provided `headline`, `news`, `impact`, and `emotion`.
2.  **Focus on skills:** The project must require students to demonstrate and apply the {st.session_state.skill_1}, {st.session_state.skill_2}, and {st.session_state.skill_3} within the {st.session_state.user_subject} context.
3.  **Structure the output strictly:**
    -   **Assignment Headline:** A title, max 10 words.
    -   **Assignment Overview:** A brief summary of the project's goal, max 20 words.
    -   **Assignment Instructions:** The steps students must follow, max 40 words.
4.  **Tone:** The assignment should be relevant to emotion, and inspiring for students.
**Assignment Headline:**
**Assignment Overview:**
**Assignment Instructions:**
}}"""

        assign_out = []
        for i, t in enumerate(top3, start=1):
            add_status(f"Generating Assignment {i}")
            render_status()
            def gen_one():
                k = rotate_api_key()
                return generate_assignment(k, make_prompt(t))
            raw = guarded_call(gen_one, f"Assignment gen {i}")
            if not raw:
                # resilient fallback minimal structure
                raw = f"""**Assignment Headline:**
Insightful Project {i}
**Assignment Overview:**
Apply skills to analyze news impact.
**Assignment Instructions:**
Collect data, build model, evaluate outcomes, present insights."""
            # parse to structured fields
            ah, ao, ai = parse_assignment_block(raw)
            t[f"ass_headline"] = ah or f"Assignment {i}"
            t[f"ass_overview"] = ao or "A focused project aligning news and skills."
            t[f"ass_instructions"] = ai or "Gather data, implement solution, evaluate, and present."
            assign_out.append(t)
            # progress step
            st.session_state.progress = 0.86 + 0.04 * i
            render_status()
            time.sleep(API_CALL_DELAY_SEC)

        st.session_state.assignments = assign_out
        st.session_state.ready = True
        st.session_state.progress = 1.0
        add_status("Completed", "ok")
        render_status()

# --------- Results and Navigation ---------
if st.session_state.ready and st.session_state.assignments:
    st.subheader("Suggestions")
    for t in st.session_state.assignments:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            if st.button(f"🔗 {t['ass_headline']}", key=f"open_{t['id']}", use_container_width=True):
                # Navigate to details page
                st.session_state.nav_target_id = t["id"]
                st.switch_page("assignment_details.py")
        with col2:
            st.write(f"{t.get('score',0.0):.3f}")

st.markdown("<div class='small'>Tips: Keep this tab open; data persists until app closes.</div>", unsafe_allow_html=True)
