import google.generativeai as genai

# Use your key here
genai.configure(api_key="AIzaSyAf82qqA8Sk45k8s5ukqssZBH9i5fql7QA")

print("Checking available models for your key...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
except Exception as e:
    print(f"Error: {e}")