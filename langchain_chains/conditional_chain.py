
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough
)
from pydantic import BaseModel, Field
from typing import Literal

model = ChatOllama(
    model="llama3.2",
    temperature=0
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the customer feedback"
    )

prompt1 = PromptTemplate.from_template(
    """Classify the following customer feedback as positive or negative.

Feedback: {feedback}

Return the actual sentiment classification."""
)

structured_model = model.with_structured_output(
    Feedback,
    method="json_schema"
)

# Classify customer feedback
classifier_chain = prompt1 | structured_model

# Positive response
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Negative response
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Select a branch based on sentiment
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt2 | model | parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "Could not find sentiment")
)

# Preserve original feedback and add sentiment
chain = (
    RunnablePassthrough.assign(
        sentiment=classifier_chain
    )
    | branch_chain
)

result = chain.invoke({
    "feedback": "This is a terrible phone"
})

print(result)
