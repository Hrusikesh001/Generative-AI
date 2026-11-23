from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)



# 1st prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": 'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})

result1 = model.invoke(prompt2)

print(result1.content)