from langchain_anthropic import chatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = chatAnthropic(model='claude-3-5-sonnet-20241022')
result = model.invoke("What is the capital of France?")
print(result.content)