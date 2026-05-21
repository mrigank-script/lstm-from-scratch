import random
from model import lstm_forward
from data import one_hot

# Generate text using LSTM model
def generate(length, seed_text, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, char_to_num, num_to_char, vocab_size, hidden_size=256):
    h1 = [[0.0] * hidden_size]
    c1 = [[0.0] * hidden_size]
    h2 = [[0.0] * hidden_size]
    c2 = [[0.0] * hidden_size]
    
    generated = ""
    # Process seed characters
    for char in seed_text:
        if char in char_to_num:
            x = one_hot(char_to_num[char], vocab_size)
            h1, c1, h2, c2, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2 = lstm_forward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by)
            generated += char
            
    curr_char = seed_text[-1] if seed_text else random.choice(list(char_to_num.keys()))
    
    # Generate char-by-char
    for _ in range(length):
        if curr_char not in char_to_num:
            curr_char = random.choice(list(char_to_num.keys()))
        x = one_hot(char_to_num[curr_char], vocab_size)
        h1, c1, h2, c2, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2 = lstm_forward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by)
        
        probs_flat = probs[0]
        s = sum(probs_flat)
        if s == 0:
            probs_flat = [1.0 / vocab_size] * vocab_size
        else:
            probs_flat = [p / s for p in probs_flat]
            
        sampled_idx = random.choices(range(vocab_size), weights=probs_flat)[0]
        curr_char = num_to_char[sampled_idx]
        generated += curr_char
        
    return generated
