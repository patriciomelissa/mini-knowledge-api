import requests


class LocalLLMService:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3"

    def generate_answer(self, context: str, question: str) -> str:

        prompt = f"""
You are a helpful assistant.
Answer the question strictly using the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60,
        )

        response.raise_for_status()

        return response.json()["response"].strip()
