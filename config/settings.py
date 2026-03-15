import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")



SERP_API = os.getenv("SERP_API_KEY")

TAVILY_API =os.getenv("TAVILY_API")

GEMINI_API = os.getenv("GEMINI_API")

GROQ_API = os.getenv("GROQ_API")

CEREBRAS_API = os.getenv("CEREBRAS")