import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
model_name = os.getenv("OLLAMA_MODEL", "llama3.2")

def get_llm():
    return ChatOllama(model=model_name, temperature=0.2)

# Groq fallback:
# groq_model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
#
# def get_groq_llm():
#     return ChatGroq(model=groq_model_name)