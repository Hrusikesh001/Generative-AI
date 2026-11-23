from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
    
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= 'Give me 5 facts about {topics} \n {format_instruction}',
    input_variables=["topics"],
    partial_variables={"format_instruction":parser.get_format_instructions() }
)

chain = template | model | parser
result = chain.invoke({"topics": "black hole"})
print(result)


# prompt = template.invoke({"topics": "black hole"})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)