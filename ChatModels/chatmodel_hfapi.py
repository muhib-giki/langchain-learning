from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = ChatHuggingFace(
    repo_id="Qwen/Qwen2.5-7B-Instruct"
)

result = llm.invoke("What is the capital of Pakistan?")
print(result.content)
