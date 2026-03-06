import requests


class LocalLLMService:
    """
    Service responsible for generating answers using a locally hosted LLM model.

    This class sends prompts to a local inference server and retrieves generated
    responses. The model is instructed to answer strictly based on the provided
    context. If the answer is not found in the context, the model should respond
    that it does not know.

    Attributes:
        url (str): Endpoint URL of the local LLM generation API.
        model (str): Name of the local LLM model used for inference.
    """

    def __init__(self) -> None:
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3"

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate an answer based on provided context using a local LLM.

        The method constructs a prompt enforcing strict context-based answering.
        If the answer is not present in the context, the model is instructed
        to respond that it does not know.

        Args:
            context (str): Reference context used to answer the question.
            question (str): User question to be answered.

        Returns:
            str: Generated answer from the local LLM model.

        Raises:
            requests.HTTPError: If the API request fails.
            KeyError: If the expected response field is missing in the API response.
        """
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
