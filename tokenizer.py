from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    tokenizer = Tokenizer()
    text = "Hello, this is a test."
    tokens = tokenizer.encode(text)
    print(tokens)
    print(tokenizer.decode(tokens['input_ids'][0]))
