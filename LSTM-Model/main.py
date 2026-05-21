import os
import sys
import time
from datetime import timedelta

# Import other components
import config
from data import inputs, targets, vocab_size, char_to_num, num_to_char, one_hot, sequence_length
from model import lstm_forward, lstm_backward, weights_intialization, bias_initalization, cross_entropy_loss
from weights import save_weights, load_weights
from generate import generate

# Define model dimensions
hidden_size = config.hidden_size
weights_path = config.weights_path

# Initialize or load weights
start_epoch = 0
lr = config.lr

loaded_weights = load_weights(weights_path)
if loaded_weights:
    Wf1 = loaded_weights["Wf1"]
    Wi1 = loaded_weights["Wi1"]
    Wc1 = loaded_weights["Wc1"]
    Wo1 = loaded_weights["Wo1"]
    Uf1 = loaded_weights["Uf1"]
    Ui1 = loaded_weights["Ui1"]
    Uc1 = loaded_weights["Uc1"]
    Uo1 = loaded_weights["Uo1"]
    bf1 = loaded_weights["bf1"]
    bi1 = loaded_weights["bi1"]
    bc1 = loaded_weights["bc1"]
    bo1 = loaded_weights["bo1"]
    
    Wf2 = loaded_weights["Wf2"]
    Wi2 = loaded_weights["Wi2"]
    Wc2 = loaded_weights["Wc2"]
    Wo2 = loaded_weights["Wo2"]
    Uf2 = loaded_weights["Uf2"]
    Ui2 = loaded_weights["Ui2"]
    Uc2 = loaded_weights["Uc2"]
    Uo2 = loaded_weights["Uo2"]
    bf2 = loaded_weights["bf2"]
    bi2 = loaded_weights["bi2"]
    bc2 = loaded_weights["bc2"]
    bo2 = loaded_weights["bo2"]
    
    Why = loaded_weights["Why"]
    by = loaded_weights["by"]
    
    start_epoch = loaded_weights.get("epoch", -1) + 1
    lr = loaded_weights.get("lr", 0.05)
    print(f"Resuming training from Epoch {start_epoch}")
else:
    # Layer 1
    Wf1 = weights_intialization(vocab_size, hidden_size)
    Wi1 = weights_intialization(vocab_size, hidden_size)
    Wc1 = weights_intialization(vocab_size, hidden_size)
    Wo1 = weights_intialization(vocab_size, hidden_size)
    Uf1 = weights_intialization(hidden_size, hidden_size)
    Ui1 = weights_intialization(hidden_size, hidden_size)
    Uc1 = weights_intialization(hidden_size, hidden_size)
    Uo1 = weights_intialization(hidden_size, hidden_size)
    bf1 = bias_initalization(hidden_size)
    bi1 = bias_initalization(hidden_size)
    bc1 = bias_initalization(hidden_size)
    bo1 = bias_initalization(hidden_size)
    
    # Layer 2
    Wf2 = weights_intialization(hidden_size, hidden_size)
    Wi2 = weights_intialization(hidden_size, hidden_size)
    Wc2 = weights_intialization(hidden_size, hidden_size)
    Wo2 = weights_intialization(hidden_size, hidden_size)
    Uf2 = weights_intialization(hidden_size, hidden_size)
    Ui2 = weights_intialization(hidden_size, hidden_size)
    Uc2 = weights_intialization(hidden_size, hidden_size)
    Uo2 = weights_intialization(hidden_size, hidden_size)
    bf2 = bias_initalization(hidden_size)
    bi2 = bias_initalization(hidden_size)
    bc2 = bias_initalization(hidden_size)
    bo2 = bias_initalization(hidden_size)

    Why = weights_intialization(hidden_size, vocab_size)
    by = bias_initalization(vocab_size)
    print("Initialized weights")

# Hyperparameters
epochs = config.epochs
# sequence_length is imported from data.py to prevent IndexError crashes

losses = []
accuracies = []

# Training loop
try:
    for epoch in range(start_epoch, epochs):
        if epoch > 0 and epoch % 10 == 0:
            lr = lr * 0.5
            
        total_loss = 0
        correct_preds = 0
        epoch_start = time.time()

        for i in range(len(inputs)):
            h1 = [[0.0] * hidden_size]
            c1 = [[0.0] * hidden_size]
            h2 = [[0.0] * hidden_size]
            c2 = [[0.0] * hidden_size]
            seq_start = time.time()

            for t in range(sequence_length - 1):
                x = one_hot(inputs[i][t], vocab_size)
                h1, c1, h2, c2, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2 = lstm_forward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by)
                
            x = one_hot(inputs[i][sequence_length - 1], vocab_size)
            Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, h1, c1, h2, c2, probs = lstm_backward(
                x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, targets[i], lr
            )

            loss = cross_entropy_loss(probs, targets[i])
            total_loss += loss
            seq_time = time.time() - seq_start
            
            # Track character prediction accuracy
            pred_idx = probs[0].index(max(probs[0]))
            if pred_idx == targets[i]:
                correct_preds += 1
            
            if (i + 1) % 100 == 0:
                print(f"epoch {epoch} seq {i+1} time: {timedelta(seconds=seq_time)}")
            if (i + 1) % 1000 == 0:
                print(f"epoch {epoch} step {i+1} loss: {total_loss / (i+1)}")

        epoch_time = time.time() - epoch_start
        accuracy = (correct_preds / len(inputs)) * 100
        print(f"epoch {epoch} loss: {total_loss} accuracy: {accuracy:.2f}% epoch_time: {timedelta(seconds=epoch_time)}")
        losses.append(total_loss / len(inputs))
        accuracies.append(accuracy)
        
        # Save weights after every epoch
        w_dict = {
            "Wf1": Wf1, "Wi1": Wi1, "Wc1": Wc1, "Wo1": Wo1,
            "Uf1": Uf1, "Ui1": Ui1, "Uc1": Uc1, "Uo1": Uo1,
            "bf1": bf1, "bi1": bi1, "bc1": bc1, "bo1": bo1,
            "Wf2": Wf2, "Wi2": Wi2, "Wc2": Wc2, "Wo2": Wo2,
            "Uf2": Uf2, "Ui2": Ui2, "Uc2": Uc2, "Uo2": Uo2,
            "bf2": bf2, "bi2": bi2, "bc2": bc2, "bo2": bo2,
            "Why": Why, "by": by,
            "epoch": epoch,
            "lr": lr
        }
        save_weights(weights_path, w_dict)
        
        # Print custom generated sample
        sample = generate(100, "ROMEO: ", Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, char_to_num, num_to_char, vocab_size, hidden_size)
        print(f"Sample generation:\n{sample}\n")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(losses)
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(os.path.dirname(__file__), "../loss.png"))
    print("Saved loss plot to loss.png")

    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(os.path.dirname(__file__), "../accuracy.png"))
    print("Saved accuracy plot to accuracy.png")
    
    # Print 5 custom generated samples
    print("\nTraining complete! Generating 5 classic Shakespearean samples...")
    seeds = ["ROMEO: ", "JULIET: ", "HAMLET: ", "OTHELLO: ", "MACBETH: "]
    for seed in seeds:
        sample = generate(100, seed, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, char_to_num, num_to_char, vocab_size, hidden_size)
        print(f"--- Seed: '{seed}' ---")
        print(f"{sample}\n")

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving current weights...")
    current_epoch = epoch - 1 if 'epoch' in locals() else -1
    current_lr = lr if 'lr' in locals() else 0.05
    w_dict = {
        "Wf1": Wf1, "Wi1": Wi1, "Wc1": Wc1, "Wo1": Wo1,
        "Uf1": Uf1, "Ui1": Ui1, "Uc1": Uc1, "Uo1": Uo1,
        "bf1": bf1, "bi1": bi1, "bc1": bc1, "bo1": bo1,
        "Wf2": Wf2, "Wi2": Wi2, "Wc2": Wc2, "Wo2": Wo2,
        "Uf2": Uf2, "Ui2": Ui2, "Uc2": Uc2, "Uo2": Uo2,
        "bf2": bf2, "bi2": bi2, "bc2": bc2, "bo2": bo2,
        "Why": Why, "by": by,
        "epoch": current_epoch,
        "lr": current_lr
    }
    save_weights(weights_path, w_dict)
    print("Weights saved successfully!")
    
    # Print 5 custom generated samples
    print("\nGenerating 5 classic Shakespearean samples...")
    seeds = ["ROMEO: ", "JULIET: ", "HAMLET: ", "OTHELLO: ", "MACBETH: "]
    for seed in seeds:
        sample = generate(100, seed, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, char_to_num, num_to_char, vocab_size, hidden_size)
        print(f"--- Seed: '{seed}' ---")
        print(f"{sample}\n")
