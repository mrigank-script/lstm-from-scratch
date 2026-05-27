import sys
import os
import random
import math

# Load custom matrix Backend library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Backend"))
from matrix import matmul, add_matrices, sub_matrices, transpose, scalar_multiply, tanh, tanh_derivative, sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, elem_mul, sigmoid_derivative_post, tanh_derivative_post, clip_matrix

# Initialize weights with Xavier uniform
def weights_intialization(neuron, weights):
    weight = []
    limit = (6 / (neuron + weights)) ** 0.5
    for i in range(neuron):
        temp = []
        for j in range(weights):
            temp.append(random.uniform(-limit, limit))
        weight.append(temp)
    return weight

# Initialize bias with zeros
def bias_initalization(weights):
    bias_weight = []
    bias_weights = []
    for i in range(weights):
        bias_weight.append(0)
    bias_weights.append(bias_weight)
    return bias_weights


# Forward pass of LSTM cell
def lstm_forward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by):
    # Layer 1
    fg1 = sigmoid(add_matrices(matmul(x, Wf1), add_matrices(matmul(h1, Uf1), bf1)))
    ig1 = sigmoid(add_matrices(matmul(x, Wi1), add_matrices(matmul(h1, Ui1), bi1)))
    cg1 = tanh(add_matrices(matmul(x, Wc1), add_matrices(matmul(h1, Uc1), bc1)))
    og1 = sigmoid(add_matrices(matmul(x, Wo1), add_matrices(matmul(h1, Uo1), bo1)))

    c1 = add_matrices(elem_mul(fg1, c1), elem_mul(ig1, cg1))
    h1 = elem_mul(og1, tanh(c1))

    # Layer 2
    fg2 = sigmoid(add_matrices(matmul(h1, Wf2), add_matrices(matmul(h2, Uf2), bf2)))
    ig2 = sigmoid(add_matrices(matmul(h1, Wi2), add_matrices(matmul(h2, Ui2), bi2)))
    cg2 = tanh(add_matrices(matmul(h1, Wc2), add_matrices(matmul(h2, Uc2), bc2)))
    og2 = sigmoid(add_matrices(matmul(h1, Wo2), add_matrices(matmul(h2, Uo2), bo2)))

    c2 = add_matrices(elem_mul(fg2, c2), elem_mul(ig2, cg2))
    h2 = elem_mul(og2, tanh(c2))

    # Output Layer
    out = add_matrices(matmul(h2, Why), by)
    probs = softmax(out)

    return h1, c1, h2, c2, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2

# Backward pass of LSTM cell
def lstm_backward(xs, h1_init, c1_init, h2_init, c2_init, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, targets, lr):
    T = len(xs)

    h1_hist = [h1_init]; c1_hist = [c1_init]
    h2_hist = [h2_init]; c2_hist = [c2_init]
    fg1_hist, ig1_hist, cg1_hist, og1_hist = [], [], [], []
    fg2_hist, ig2_hist, cg2_hist, og2_hist = [], [], [], []
    probs_hist = []

    h1, c1, h2, c2 = h1_init, c1_init, h2_init, c2_init
    for t in range(T):
        h1_new, c1_new, h2_new, c2_new, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2 = lstm_forward(xs[t], h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by)
        h1_hist.append(h1_new); c1_hist.append(c1_new)
        h2_hist.append(h2_new); c2_hist.append(c2_new)
        fg1_hist.append(fg1); ig1_hist.append(ig1); cg1_hist.append(cg1); og1_hist.append(og1)
        fg2_hist.append(fg2); ig2_hist.append(ig2); cg2_hist.append(cg2); og2_hist.append(og2)
        probs_hist.append(probs)
        h1, c1, h2, c2 = h1_new, c1_new, h2_new, c2_new

    dWf1 = scalar_multiply(Wf1, 0); dWi1 = scalar_multiply(Wi1, 0)
    dWc1 = scalar_multiply(Wc1, 0); dWo1 = scalar_multiply(Wo1, 0)
    dUf1 = scalar_multiply(Uf1, 0); dUi1 = scalar_multiply(Ui1, 0)
    dUc1 = scalar_multiply(Uc1, 0); dUo1 = scalar_multiply(Uo1, 0)
    dbf1 = scalar_multiply(bf1, 0); dbi1 = scalar_multiply(bi1, 0)
    dbc1 = scalar_multiply(bc1, 0); dbo1 = scalar_multiply(bo1, 0)

    dWf2 = scalar_multiply(Wf2, 0); dWi2 = scalar_multiply(Wi2, 0)
    dWc2 = scalar_multiply(Wc2, 0); dWo2 = scalar_multiply(Wo2, 0)
    dUf2 = scalar_multiply(Uf2, 0); dUi2 = scalar_multiply(Ui2, 0)
    dUc2 = scalar_multiply(Uc2, 0); dUo2 = scalar_multiply(Uo2, 0)
    dbf2 = scalar_multiply(bf2, 0); dbi2 = scalar_multiply(bi2, 0)
    dbc2 = scalar_multiply(bc2, 0); dbo2 = scalar_multiply(bo2, 0)

    dWhy = scalar_multiply(Why, 0); dby = scalar_multiply(by, 0)

    dh2_next = scalar_multiply(h2_init, 0); dc2_next = scalar_multiply(c2_init, 0)
    dh1_next = scalar_multiply(h1_init, 0); dc1_next = scalar_multiply(c1_init, 0)

    for t in reversed(range(T)):
        dout = [row[:] for row in probs_hist[t]]
        dout[0][targets[t]] -= 1

        dWhy = add_matrices(dWhy, matmul(transpose(h2_hist[t+1]), dout))
        dby = add_matrices(dby, dout)

        dh2 = add_matrices(matmul(dout, transpose(Why)), dh2_next)

        dog2 = elem_mul(dh2, tanh(c2_hist[t+1]))
        dog2 = sigmoid_derivative_post(og2_hist[t], dog2)

        dc2 = add_matrices(elem_mul(elem_mul(dh2, og2_hist[t]), tanh_derivative(c2_hist[t+1])), dc2_next)

        dfg2 = elem_mul(dc2, c2_hist[t])
        dfg2 = sigmoid_derivative_post(fg2_hist[t], dfg2)
        dig2 = elem_mul(dc2, cg2_hist[t])
        dig2 = sigmoid_derivative_post(ig2_hist[t], dig2)
        dcg2 = elem_mul(dc2, ig2_hist[t])
        dcg2 = tanh_derivative_post(cg2_hist[t], dcg2)

        dWf2 = add_matrices(dWf2, matmul(transpose(h1_hist[t+1]), dfg2))
        dWi2 = add_matrices(dWi2, matmul(transpose(h1_hist[t+1]), dig2))
        dWc2 = add_matrices(dWc2, matmul(transpose(h1_hist[t+1]), dcg2))
        dWo2 = add_matrices(dWo2, matmul(transpose(h1_hist[t+1]), dog2))

        dUf2 = add_matrices(dUf2, matmul(transpose(h2_hist[t]), dfg2))
        dUi2 = add_matrices(dUi2, matmul(transpose(h2_hist[t]), dig2))
        dUc2 = add_matrices(dUc2, matmul(transpose(h2_hist[t]), dcg2))
        dUo2 = add_matrices(dUo2, matmul(transpose(h2_hist[t]), dog2))

        dbf2 = add_matrices(dbf2, dfg2); dbi2 = add_matrices(dbi2, dig2)
        dbc2 = add_matrices(dbc2, dcg2); dbo2 = add_matrices(dbo2, dog2)

        dh2_next = add_matrices(matmul(dfg2, transpose(Uf2)), add_matrices(matmul(dig2, transpose(Ui2)), add_matrices(matmul(dcg2, transpose(Uc2)), matmul(dog2, transpose(Uo2)))))
        dc2_next = elem_mul(dc2, fg2_hist[t])

        dh1 = add_matrices(add_matrices(matmul(dfg2, transpose(Wf2)), add_matrices(matmul(dig2, transpose(Wi2)), add_matrices(matmul(dcg2, transpose(Wc2)), matmul(dog2, transpose(Wo2))))), dh1_next)

        dog1 = elem_mul(dh1, tanh(c1_hist[t+1]))
        dog1 = sigmoid_derivative_post(og1_hist[t], dog1)

        dc1 = add_matrices(elem_mul(elem_mul(dh1, og1_hist[t]), tanh_derivative(c1_hist[t+1])), dc1_next)

        dfg1 = elem_mul(dc1, c1_hist[t])
        dfg1 = sigmoid_derivative_post(fg1_hist[t], dfg1)
        dig1 = elem_mul(dc1, cg1_hist[t])
        dig1 = sigmoid_derivative_post(ig1_hist[t], dig1)
        dcg1 = elem_mul(dc1, ig1_hist[t])
        dcg1 = tanh_derivative_post(cg1_hist[t], dcg1)

        dWf1 = add_matrices(dWf1, matmul(transpose(xs[t]), dfg1))
        dWi1 = add_matrices(dWi1, matmul(transpose(xs[t]), dig1))
        dWc1 = add_matrices(dWc1, matmul(transpose(xs[t]), dcg1))
        dWo1 = add_matrices(dWo1, matmul(transpose(xs[t]), dog1))

        dUf1 = add_matrices(dUf1, matmul(transpose(h1_hist[t]), dfg1))
        dUi1 = add_matrices(dUi1, matmul(transpose(h1_hist[t]), dig1))
        dUc1 = add_matrices(dUc1, matmul(transpose(h1_hist[t]), dcg1))
        dUo1 = add_matrices(dUo1, matmul(transpose(h1_hist[t]), dog1))

        dbf1 = add_matrices(dbf1, dfg1); dbi1 = add_matrices(dbi1, dig1)
        dbc1 = add_matrices(dbc1, dcg1); dbo1 = add_matrices(dbo1, dog1)

        dh1_next = add_matrices(matmul(dfg1, transpose(Uf1)), add_matrices(matmul(dig1, transpose(Ui1)), add_matrices(matmul(dcg1, transpose(Uc1)), matmul(dog1, transpose(Uo1)))))
        dc1_next = elem_mul(dc1, fg1_hist[t])

    dWf1 = clip_matrix(dWf1, -5.0, 5.0); dWi1 = clip_matrix(dWi1, -5.0, 5.0)
    dWc1 = clip_matrix(dWc1, -5.0, 5.0); dWo1 = clip_matrix(dWo1, -5.0, 5.0)
    dUf1 = clip_matrix(dUf1, -5.0, 5.0); dUi1 = clip_matrix(dUi1, -5.0, 5.0)
    dUc1 = clip_matrix(dUc1, -5.0, 5.0); dUo1 = clip_matrix(dUo1, -5.0, 5.0)
    dbf1 = clip_matrix(dbf1, -5.0, 5.0); dbi1 = clip_matrix(dbi1, -5.0, 5.0)
    dbc1 = clip_matrix(dbc1, -5.0, 5.0); dbo1 = clip_matrix(dbo1, -5.0, 5.0)

    dWf2 = clip_matrix(dWf2, -5.0, 5.0); dWi2 = clip_matrix(dWi2, -5.0, 5.0)
    dWc2 = clip_matrix(dWc2, -5.0, 5.0); dWo2 = clip_matrix(dWo2, -5.0, 5.0)
    dUf2 = clip_matrix(dUf2, -5.0, 5.0); dUi2 = clip_matrix(dUi2, -5.0, 5.0)
    dUc2 = clip_matrix(dUc2, -5.0, 5.0); dUo2 = clip_matrix(dUo2, -5.0, 5.0)
    dbf2 = clip_matrix(dbf2, -5.0, 5.0); dbi2 = clip_matrix(dbi2, -5.0, 5.0)
    dbc2 = clip_matrix(dbc2, -5.0, 5.0); dbo2 = clip_matrix(dbo2, -5.0, 5.0)

    dWhy = clip_matrix(dWhy, -5.0, 5.0); dby = clip_matrix(dby, -5.0, 5.0)

    Wf1 = sub_matrices(Wf1, scalar_multiply(dWf1, lr)); Wi1 = sub_matrices(Wi1, scalar_multiply(dWi1, lr))
    Wc1 = sub_matrices(Wc1, scalar_multiply(dWc1, lr)); Wo1 = sub_matrices(Wo1, scalar_multiply(dWo1, lr))
    Uf1 = sub_matrices(Uf1, scalar_multiply(dUf1, lr)); Ui1 = sub_matrices(Ui1, scalar_multiply(dUi1, lr))
    Uc1 = sub_matrices(Uc1, scalar_multiply(dUc1, lr)); Uo1 = sub_matrices(Uo1, scalar_multiply(dUo1, lr))
    bf1 = sub_matrices(bf1, scalar_multiply(dbf1, lr)); bi1 = sub_matrices(bi1, scalar_multiply(dbi1, lr))
    bc1 = sub_matrices(bc1, scalar_multiply(dbc1, lr)); bo1 = sub_matrices(bo1, scalar_multiply(dbo1, lr))

    Wf2 = sub_matrices(Wf2, scalar_multiply(dWf2, lr)); Wi2 = sub_matrices(Wi2, scalar_multiply(dWi2, lr))
    Wc2 = sub_matrices(Wc2, scalar_multiply(dWc2, lr)); Wo2 = sub_matrices(Wo2, scalar_multiply(dWo2, lr))
    Uf2 = sub_matrices(Uf2, scalar_multiply(dUf2, lr)); Ui2 = sub_matrices(Ui2, scalar_multiply(dUi2, lr))
    Uc2 = sub_matrices(Uc2, scalar_multiply(dUc2, lr)); Uo2 = sub_matrices(Uo2, scalar_multiply(dUo2, lr))
    bf2 = sub_matrices(bf2, scalar_multiply(dbf2, lr)); bi2 = sub_matrices(bi2, scalar_multiply(dbi2, lr))
    bc2 = sub_matrices(bc2, scalar_multiply(dbc2, lr)); bo2 = sub_matrices(bo2, scalar_multiply(dbo2, lr))

    Why = sub_matrices(Why, scalar_multiply(dWhy, lr)); by = sub_matrices(by, scalar_multiply(dby, lr))

    return Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, h1_hist[-1], c1_hist[-1], h2_hist[-1], c2_hist[-1], probs_hist
    
# Compute cross entropy loss
def cross_entropy_loss(probs, actual_char):
    return -math.log(probs[0][actual_char] + 1e-9)
