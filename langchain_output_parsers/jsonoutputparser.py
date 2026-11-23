from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)
parser = JsonOutputParser()

template = PromptTemplate(
    template= 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={"format_instruction":parser.get_format_instructions() }
)

chain = template | model | parser
result = chain.invoke({})
print(result)
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))