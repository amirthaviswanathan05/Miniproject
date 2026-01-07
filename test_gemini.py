from google import genai
from google.genai import types
client = genai.Client(api_key="AIzaSyBQvxrky9IppqXQEDeqzf4kpcytH7vHEYQ",http_options=types.HttpOptions(api_version='v1'))

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence"
)

print(resp.text)
