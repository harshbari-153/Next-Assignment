import os
import streamlit as st

st.set_page_config(page_title="Assignment Details", page_icon="📄", layout="centered")

st.title("Assignment Details")

if "assignments" not in st.session_state or not st.session_state.assignments or "nav_target_id" not in st.session_state or st.session_state.nav_target_id is None:
    st.warning("No assignment selected. Go back to Home.")
    if st.button("⬅️ Back to Home"):
        st.switch_page("home_page.py")
    st.stop()

aid = st.session_state.nav_target_id
assign = next((a for a in st.session_state.assignments if a["id"] == aid), None)

if not assign:
    st.warning("Assignment not found in current session.")
    if st.button("⬅️ Back to Home"):
        st.switch_page("home_page.py")
    st.stop()

with st.container():
    st.subheader(assign.get("ass_headline","Untitled"))
    st.write(f"ID: {assign['id']}")
    st.write(f"URL: {assign.get('url','')}")

    st.markdown("### Overview")
    st.write(assign.get("ass_overview",""))

    st.markdown("### Instructions")
    st.write(assign.get("ass_instructions",""))

st.divider()
if st.button("⬅️ Back to Home", use_container_width=True):
    st.switch_page("home_page.py")
