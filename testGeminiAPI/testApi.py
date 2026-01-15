from google import genai
from global_const import api_key

client = genai.Client(api_key=api_key)

# myfile = client.files.upload(file=r"")

GeminiPromptPath = r""
with open(GeminiPromptPath,mode='r',encoding='utf-8') as f:
    prompt = f.read()

print(prompt)
response = client.models.generate_content(
    model="gemini-2.5-flash", contents=[myfile, prompt]
)

print(response.text)