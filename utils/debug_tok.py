from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_oran")
text = "[#dup tsns 0],(2)[DATA](B)(E)[TSN:2450285146][SID:0][SSEQ 4][PPID 0x3e]"
tokens = tokenizer.tokenize(text)
ids = tokenizer(text)["input_ids"]

print("Original:", text)
print("Tokens:", tokens)
print("IDs:", ids)

# Also check specific numbers
print("Tokenize 2450285146:", tokenizer.tokenize("2450285146"))
