# openai_analyzer.py
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_expression(event_text):
    prompt = f"The user displayed the following behaviors: {event_text}. What might this suggest emotionally?"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in reading subtle human emotions from facial expressions and gestures."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["choices"][0]["message"]["content"]
