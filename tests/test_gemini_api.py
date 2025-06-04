"""
https://ai.google.dev/gemini-api/docs/quickstart?lang=python
"""

from google import genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

# Initialize the client
client = genai.Client(api_key=api_key)

# Generate content
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="Write a short story about a surfer who is also a phd student in AI."
)
print(response.text)