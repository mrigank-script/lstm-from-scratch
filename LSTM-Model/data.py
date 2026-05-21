import os
import config

# Load training data
with open(config.shakespeare_path, "r", encoding="utf-8") as f:
    text = f.read()[:config.dataset_limit]

# Build character mappings
chars_in_text = sorted(list(set(text)))
vocab_size = len(chars_in_text)
char_to_num = {}
num_to_char = {}

num = 0
for i in chars_in_text:
    char_to_num[i] = num
    num_to_char[num] = i
    num += 1

# Generate one-hot vectors
def one_hot(char_num, vocab_size):
    vector = [0.0] * vocab_size
    vector[char_num] = 1.0
    return [vector]

# Prepare sequence batches
sequence_length = config.sequence_length
inputs = []
targets = []

for i in range(0, len(text) - sequence_length, sequence_length):
    input_seq = text[i : i + sequence_length]
    target = text[i + sequence_length]
    input_nums = []
    for ch in input_seq:
        input_nums.append(char_to_num[ch])

    inputs.append(input_nums)
    targets.append(char_to_num[target])

print(f"sequences built: {len(inputs)}")
