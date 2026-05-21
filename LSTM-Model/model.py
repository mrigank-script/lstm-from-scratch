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
def lstm_backward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, actual_char, lr):
    h1_new, c1_new, h2_new, c2_new, probs, fg1, ig1, cg1, og1, fg2, ig2, cg2, og2 = lstm_forward(x, h1, c1, h2, c2, Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by)
    
    dout = [row[:] for row in probs]
    dout[0][actual_char] -= 1
    
    dWhy = matmul(transpose(h2_new), dout)
    dby = dout
    dh2 = matmul(dout, transpose(Why))

    # Layer 2 Gradients
    dog2 = elem_mul(dh2, tanh(c2_new))
    dog2 = sigmoid_derivative_post(og2, dog2)

    dtanh_c2 = tanh_derivative(c2_new)
    dc2 = elem_mul(elem_mul(dh2, og2), dtanh_c2)

    dfg2 = elem_mul(dc2, c2)
    dfg2 = sigmoid_derivative_post(fg2, dfg2)

    dig2 = elem_mul(dc2, cg2)
    dig2 = sigmoid_derivative_post(ig2, dig2)

    dcg2 = elem_mul(dc2, ig2)
    dcg2 = tanh_derivative_post(cg2, dcg2)

    dWf2 = matmul(transpose(h1_new), dfg2)
    dWi2 = matmul(transpose(h1_new), dig2)
    dWc2 = matmul(transpose(h1_new), dcg2)
    dWo2 = matmul(transpose(h1_new), dog2)

    dUf2 = matmul(transpose(h2), dfg2)
    dUi2 = matmul(transpose(h2), dig2)
    dUc2 = matmul(transpose(h2), dcg2)
    dUo2 = matmul(transpose(h2), dog2)

    dbf2 = dfg2
    dbi2 = dig2
    dbc2 = dcg2
    dbo2 = dog2

    # Flow gradient back to Layer 1
    dh1 = add_matrices(matmul(dfg2, transpose(Wf2)), add_matrices(matmul(dig2, transpose(Wi2)), add_matrices(matmul(dcg2, transpose(Wc2)), matmul(dog2, transpose(Wo2)))))

    # Layer 1 Gradients
    dog1 = elem_mul(dh1, tanh(c1_new))
    dog1 = sigmoid_derivative_post(og1, dog1)

    dtanh_c1 = tanh_derivative(c1_new)
    dc1 = elem_mul(elem_mul(dh1, og1), dtanh_c1)

    dfg1 = elem_mul(dc1, c1)
    dfg1 = sigmoid_derivative_post(fg1, dfg1)

    dig1 = elem_mul(dc1, cg1)
    dig1 = sigmoid_derivative_post(ig1, dig1)

    dcg1 = elem_mul(dc1, ig1)
    dcg1 = tanh_derivative_post(cg1, dcg1)

    dWf1 = matmul(transpose(x), dfg1)
    dWi1 = matmul(transpose(x), dig1)
    dWc1 = matmul(transpose(x), dcg1)
    dWo1 = matmul(transpose(x), dog1)

    dUf1 = matmul(transpose(h1), dfg1)
    dUi1 = matmul(transpose(h1), dig1)
    dUc1 = matmul(transpose(h1), dcg1)
    dUo1 = matmul(transpose(h1), dog1)

    dbf1 = dfg1
    dbi1 = dig1
    dbc1 = dcg1
    dbo1 = dog1

    # Clip gradients
    dWf1 = clip_matrix(dWf1, -5.0, 5.0)
    dWi1 = clip_matrix(dWi1, -5.0, 5.0)
    dWc1 = clip_matrix(dWc1, -5.0, 5.0)
    dWo1 = clip_matrix(dWo1, -5.0, 5.0)
    dUf1 = clip_matrix(dUf1, -5.0, 5.0)
    dUi1 = clip_matrix(dUi1, -5.0, 5.0)
    dUc1 = clip_matrix(dUc1, -5.0, 5.0)
    dUo1 = clip_matrix(dUo1, -5.0, 5.0)
    dbf1 = clip_matrix(dbf1, -5.0, 5.0)
    dbi1 = clip_matrix(dbi1, -5.0, 5.0)
    dbc1 = clip_matrix(dbc1, -5.0, 5.0)
    dbo1 = clip_matrix(dbo1, -5.0, 5.0)

    dWf2 = clip_matrix(dWf2, -5.0, 5.0)
    dWi2 = clip_matrix(dWi2, -5.0, 5.0)
    dWc2 = clip_matrix(dWc2, -5.0, 5.0)
    dWo2 = clip_matrix(dWo2, -5.0, 5.0)
    dUf2 = clip_matrix(dUf2, -5.0, 5.0)
    dUi2 = clip_matrix(dUi2, -5.0, 5.0)
    dUc2 = clip_matrix(dUc2, -5.0, 5.0)
    dUo2 = clip_matrix(dUo2, -5.0, 5.0)
    dbf2 = clip_matrix(dbf2, -5.0, 5.0)
    dbi2 = clip_matrix(dbi2, -5.0, 5.0)
    dbc2 = clip_matrix(dbc2, -5.0, 5.0)
    dbo2 = clip_matrix(dbo2, -5.0, 5.0)

    dWhy = clip_matrix(dWhy, -5.0, 5.0)
    dby = clip_matrix(dby, -5.0, 5.0)

    # Update weights Layer 1
    Wf1 = sub_matrices(Wf1, scalar_multiply(dWf1, lr))
    Wi1 = sub_matrices(Wi1, scalar_multiply(dWi1, lr))
    Wc1 = sub_matrices(Wc1, scalar_multiply(dWc1, lr))
    Wo1 = sub_matrices(Wo1, scalar_multiply(dWo1, lr))

    Uf1 = sub_matrices(Uf1, scalar_multiply(dUf1, lr))
    Ui1 = sub_matrices(Ui1, scalar_multiply(dUi1, lr))
    Uc1 = sub_matrices(Uc1, scalar_multiply(dUc1, lr))
    Uo1 = sub_matrices(Uo1, scalar_multiply(dUo1, lr))

    bf1 = sub_matrices(bf1, scalar_multiply(dbf1, lr))
    bi1 = sub_matrices(bi1, scalar_multiply(dbi1, lr))
    bc1 = sub_matrices(bc1, scalar_multiply(dbc1, lr))
    bo1 = sub_matrices(bo1, scalar_multiply(dbo1, lr))

    # Update weights Layer 2
    Wf2 = sub_matrices(Wf2, scalar_multiply(dWf2, lr))
    Wi2 = sub_matrices(Wi2, scalar_multiply(dWi2, lr))
    Wc2 = sub_matrices(Wc2, scalar_multiply(dWc2, lr))
    Wo2 = sub_matrices(Wo2, scalar_multiply(dWo2, lr))

    Uf2 = sub_matrices(Uf2, scalar_multiply(dUf2, lr))
    Ui2 = sub_matrices(Ui2, scalar_multiply(dUi2, lr))
    Uc2 = sub_matrices(Uc2, scalar_multiply(dUc2, lr))
    Uo2 = sub_matrices(Uo2, scalar_multiply(dUo2, lr))

    bf2 = sub_matrices(bf2, scalar_multiply(dbf2, lr))
    bi2 = sub_matrices(bi2, scalar_multiply(dbi2, lr))
    bc2 = sub_matrices(bc2, scalar_multiply(dbc2, lr))
    bo2 = sub_matrices(bo2, scalar_multiply(dbo2, lr))

    Why = sub_matrices(Why, scalar_multiply(dWhy, lr))
    by = sub_matrices(by, scalar_multiply(dby, lr))

    return Wf1, Wi1, Wc1, Wo1, Uf1, Ui1, Uc1, Uo1, bf1, bi1, bc1, bo1, Wf2, Wi2, Wc2, Wo2, Uf2, Ui2, Uc2, Uo2, bf2, bi2, bc2, bo2, Why, by, h1_new, c1_new, h2_new, c2_new, probs

# Compute cross entropy loss
def cross_entropy_loss(probs, actual_char):
    return -math.log(probs[0][actual_char] + 1e-9)
