from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model1 = ChatOllama(
    model="llama3.2",
    temperature=0
)

model2 = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single documents \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain

text = """ Artificial Intelligence (AI) is one of the most transformative technologies of the modern world, changing how people live, work, communicate, and solve complex problems. Artificial Intelligence refers to the development of computer systems capable of performing tasks that normally require human intelligence, such as learning, reasoning, recognizing patterns, understanding language, and making decisions. One of the most important branches of AI is Machine Learning (ML), which allows computers to learn from data and improve their performance without requiring explicit instructions for every situation. Deep Learning, a specialized area of Machine Learning, uses artificial neural networks with multiple layers to process large amounts of information and recognize complicated patterns. These technologies are widely used in everyday applications such as voice assistants, recommendation systems, facial recognition, language translation, fraud detection, and autonomous vehicles. For example, streaming platforms use Machine Learning algorithms to recommend movies based on user preferences, while online shopping websites analyze customer behavior to suggest relevant products. In healthcare, AI assists medical professionals in analyzing medical images, identifying patterns that may indicate diseases, and developing personalized treatment strategies. In education, AI-powered learning platforms can adapt lessons to students' individual learning speeds, provide instant feedback, and generate personalized practice questions. In agriculture, AI helps farmers monitor crop conditions, predict weather-related risks, detect plant diseases, and improve resource management. In the financial sector, banks use AI-based systems to detect unusual transactions, evaluate risks, and improve customer support through intelligent chatbots. Another major development in Artificial Intelligence is Natural Language Processing (NLP), which enables computers to analyze, understand, and generate human language. NLP powers applications such as chatbots, automatic translation tools, sentiment analysis systems, and text summarization software. Large Language Models (LLMs), which are trained on extensive collections of text, can generate human-like responses, answer questions, explain technical concepts, summarize documents, and assist with programming. Generative AI goes beyond analyzing existing information by creating new content, including text, images, audio, video, and computer code. However, despite its benefits, Artificial Intelligence also presents significant challenges. AI systems can produce incorrect information, reflect biases present in training data, and raise concerns about privacy, security, and transparency. The growing use of automation may also change employment patterns by reducing demand for some repetitive tasks while creating opportunities in AI development, data science, cybersecurity, and other technical fields. Responsible AI development requires careful attention to fairness, accountability, human oversight, data protection, and ethical decision-making. Governments, researchers, educational institutions, and technology companies are working to develop policies and practices that encourage beneficial AI applications while reducing potential risks. In the future, AI is expected to play an increasingly important role in scientific research, environmental protection, intelligent transportation, robotics, and personalized education. Nevertheless, the success of Artificial Intelligence will depend not only on technological progress but also on how responsibly people design, deploy, and manage these systems. By combining human creativity, critical thinking, and ethical judgment with powerful AI technologies, society can use Artificial Intelligence to improve productivity, support innovation, and address important global challenges."""

result = chain.invoke({'text': text})

print(result)
chain.get_graph().print_ascii()