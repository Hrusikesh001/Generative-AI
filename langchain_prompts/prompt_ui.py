
import streamlit as st
from langchain_ollama import ChatOllama

# Initialize the LLM
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

st.header("Research Tool")

user_input = st.text_area("Enter your prompt")

if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            result = model.invoke(user_input)
            st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
