# dataset.py

import torch
import random
from tokenizer import Tokenizer


text = """In the far corners of Pruitland, whispers spoke of the Echo. 
It was not a voice, but a silence louder than war. The people feared what they could not hear."""

tokenizer = Tokenizer(text)

data = tokenizer.encode(text)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str, batch_size: int, block_size: int):
    data_source = train_data if split == "train" else val_data
    ix = [random.randint(0, len(data_source) - block_size - 1) for _ in range(batch_size)]
    x = torch.tensor([data_source[i:i+block_size] for i in ix])
    y = torch.tensor([data_source[i+1:i+block_size+1] for i in ix])
    return x, y

#Test the get_batch function
test_mode = False
if test_mode:
    batch_size = 4
    block_size = 12

    x, y = get_batch("train", batch_size, block_size)

    print("x (tokens):", x)
    print("y (tokens):", y)
