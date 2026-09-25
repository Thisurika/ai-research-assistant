import os

from google import genai


class GeminiService:
    """
    Generate grounded answers using retrieved evidence.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Add it to the .env file."
            )

        self.model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict]
    ) -> str:
        """
        Answer a question using only retrieved chunks.
        """

        question = question.strip()

        if not question:
            raise ValueError(
                "Please enter a question."
            )

        if not retrieved_chunks:
            raise ValueError(
                "No relevant context was retrieved."
            )

        context_sections = []

        for result in retrieved_chunks:
            context_sections.append(
                (
                    f"[Source: {result['source']}, "
                    f"Page: {result['page_number']}]\n"
                    f"{result['text']}"
                )
            )

        context = "\n\n".join(context_sections)

        prompt = f"""
Use only the provided research-paper context to answer
the user's question.

Rules:
1. Do not use unsupported outside information.
2. If the answer is not present in the context, say:
   "I could not find enough information in the uploaded documents."
3. Give a clear and concise answer.
4. Cite supporting evidence using this format:
   [filename, page number]
5. Do not invent citations.

Context:
{context}

User question:
{question}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            system_instruction=(
                "You are a careful academic research "
                "assistant. Your answers must be grounded "
                "in the supplied document evidence."
            ),
            input=prompt,
            store=False
        )

        return interaction.output_text