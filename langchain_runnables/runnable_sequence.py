from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableSequence

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain1 = RunnableSequence(prompt1, model, parser)

joke = chain1.invoke({"topic": "AI"})

print("JOKE:")
print(joke)

# Second chain: Explain joke
chain2 = RunnableSequence(prompt2, model, parser)

explanation = chain2.invoke({"text": joke})

print("\nEXPLANATION:")
print(explanation)