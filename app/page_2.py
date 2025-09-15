import streamlit as st

st.set_page_config(page_title="Assignment Details", layout="centered")

if "assignments" not in st.session_state or st.session_state.selected_assignment is None:
    st.warning("No assignment selected.")
    st.stop()

assign = st.session_state.assignments[st.session_state.selected_assignment]

st.title("📘 Assignment Details")
st.markdown(f"### {assign['Assignment Headline']}")
st.markdown(f"**Overview:** {assign['Assignment Overview']}")
st.markdown(f"**Instructions:** {assign['Assignment Instructions']}")
st.markdown(f"[🔗 Source URL]({assign['url']})")

st.markdown("---")
st.subheader("⬅️ Previous Assignments")
for idx, title in enumerate(st.session_state.headlines):
    if st.button(title, key=f"back_{idx}"):
        st.session_state.selected_assignment = idx
        st.rerun()

if st.button("🔙 Back to Home"):
    st.session_state.selected_assignment = None
    st.switch_page("app.py")
