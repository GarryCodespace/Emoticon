# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"Your OpenAI key is: {api_key}")
