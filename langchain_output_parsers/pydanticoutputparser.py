from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
Generate the name, age, and city of a fictional {place} person.

Return actual person information, NOT a JSON schema.
Do not return properties, required, description, or type.
Return ONLY the JSON object.

{format_instruction}
""",
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


chain = template | model | parser
final_result = chain.invoke({"place": "New York"})
print(final_result)

# prompt = template.invoke({"place": "New York"})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)
