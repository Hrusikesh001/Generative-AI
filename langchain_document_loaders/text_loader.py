
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = PromptTemplate(
    template="Write a summary of the following text:\n{poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

# Locate cricket.txt relative to this Python file
file_path = Path(__file__).resolve().parent / "cricket.txt"

loader = TextLoader(str(file_path), encoding="utf-8")

docs = loader.load()

print("Type:", type(docs))
print("Number of documents:", len(docs))
print("Content:", docs[0].page_content)
print("Metadata:", docs[0].metadata)

# Create LangChain pipeline
chain = prompt | model | parser

# Generate summary
result = chain.invoke({
    "poem": docs[0].page_content
})

print("\n========== SUMMARY ==========\n")
print(result)
