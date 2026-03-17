import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.ai_agent import run_agent

st.title("AI Smart Agriculture Advisor")

query = st.text_area("Enter soil information or question")

if st.button("Ask AI Advisor"):

    result = run_agent(query)

    st.write(result)
