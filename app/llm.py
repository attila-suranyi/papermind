import ollama

from app.model import Prompt


def chat(prompt: Prompt) -> str:
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ],
    )

    return response["message"]["content"]
