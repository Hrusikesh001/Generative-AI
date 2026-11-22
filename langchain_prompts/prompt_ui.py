
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, load_prompt

# Initialize LLM
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

st.header("Research Tool")

# Research Paper Selection
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Explanation Style
style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

# Explanation Length
length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template = load_prompt("template.json")



# Generate Response
if st.button("Generate Response"):
    chain = template | model

    with st.spinner("Generating response..."):
        try:
            result = chain.invoke({
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            })

            st.markdown(result.content)

        except Exception as e:
            st.error(f"Error: {e}")

