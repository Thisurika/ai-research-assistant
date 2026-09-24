import tiktoken


class TextTokenizer:
    """
    Encode, decode and count text tokens.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """

        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        Convert token IDs back into text.
        """

        return self.encoding.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """
        Return the number of tokens in the text.
        """

        return len(self.encode(text))