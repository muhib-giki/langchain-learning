# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


documents = [
    'Virat Kohli – Former Indian captain known for his aggressive batting style and consistency in all formats.',
    'Babar Azam – Pakistan’s top-order batsman recognized for his elegant cover drives and strong ODI average.',
    'Joe Root – England’s leading Test run-scorer of the modern era with exceptional technique.',
    'Kane Williamson – Calm and tactically sharp captain who led New Zealand to the World Test Championship title.',
    'Steve Smith – Unorthodox yet highly effective batsman with a remarkable Test batting average.'
]

query = 'Who has performed in every cricket format'

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]


print(query)
print(documents[index])
print("Similarity Score is: ", score)