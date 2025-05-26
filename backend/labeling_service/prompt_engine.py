import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def label_text(text: str, label_options: list[str]) -> str:
    prompt = f"""You're a smart AI data assistant. Label this sentence using one of the following options: {', '.join(label_options)}.

Text: \"{text}\"

Label:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.3,
        max_tokens=10,
    )
    return response['choices'][0]['message']['content'].strip()
