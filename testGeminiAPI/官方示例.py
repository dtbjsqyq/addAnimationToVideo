from google import genai
from global_const import *
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="YOUR PROMPT"
)
print(response.text)