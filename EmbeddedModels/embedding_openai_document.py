from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimension = 32)

document = [
    'Karachi is the capital of Pakistan',
    'Paris is the capital of France',
    'Ottawa is the capital of Canada',
]

result = embedding.embed_documents(document)

print(str(result))