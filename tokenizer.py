# tokenizer.py

class Tokenizer:
    def __init__(self, text: str):
        # Build the vocabulary from the provided text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Character ↔ index mappings
        self.stoi = { ch: i for i, ch in enumerate(self.chars) }
        self.itos = { i: ch for i, ch in enumerate(self.chars) }

    def encode(self, s: str) -> list[int]:
        # Convert a string into a list of integers (tokens)
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        # Convert a list of integers back into a string
        return ''.join([self.itos[i] for i in l])


# Optional: test it if run directly
if __name__ == "__main__":
    test_text = "Helo World! I am a test string in a tokenizer."
    tokenizer = Tokenizer(test_text)
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print("Vocab size:", tokenizer.vocab_size)
    print("Chars:", tokenizer.chars)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
