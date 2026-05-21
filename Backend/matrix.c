#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// All matrices are flat 1D arrays passed as float*
// Access element [row][col] as: matrix[row * cols + col]

void matmul(float *A, float *B, float *result, int rowsA, int common, int colsB) {
    for (int i = 0; i < rowsA * colsB; i++) result[i] = 0.0f;

    for (int i = 0; i < rowsA; i++) {
        for (int k = 0; k < common; k++) {
            float a = A[i * common + k];
            for (int j = 0; j < colsB; j++) {
                result[i * colsB + j] += a * B[k * colsB + j];
            }
        }
    }
}

// Add two matrices: result = A + B
void add_matrices(float *A, float *B, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] + B[i];
    }
}

// Subtract two matrices: result = A - B
void sub_matrices(float *A, float *B, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] - B[i];
    }
}

// Transpose a matrix: result = A.T
void transpose_matrix(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = A[i * cols + j];
        }
    }
}

// Multiply every element by a scalar: result = A * scalar
void scalar_multiply(float *A, float scalar, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] * scalar;
    }
}

// ===== ACTIVATION FUNCTIONS =====

void apply_tanh(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = tanhf(A[i]);
    }
}

void apply_tanh_derivative(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float t = tanhf(A[i]);
        result[i] = 1.0f - t * t;
    }
}

void apply_sigmoid(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = 1.0f / (1.0f + expf(-A[i]));
    }
}

void apply_sigmoid_derivative(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float s = 1.0f / (1.0f + expf(-A[i]));
        result[i] = s * (1.0f - s);
    }
}

void apply_relu(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] > 0.0f ? A[i] : 0.0f;
    }
}

void apply_relu_derivative(float *A, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void elem_mul(float *A, float *B, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        result[i] = A[i] * B[i];
    }
}

// Sigmoid derivative post-activation: grad * S * (1 - S)
void sigmoid_derivative_post(float *S, float *grad, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float s = S[i];
        result[i] = grad[i] * s * (1.0f - s);
    }
}

// Tanh derivative post-activation: grad * (1 - T^2)
void tanh_derivative_post(float *T, float *grad, float *result, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float t = T[i];
        result[i] = grad[i] * (1.0f - t * t);
    }
}

// Clip matrix values to [min_val, max_val]
void clip_matrix(float *A, float *result, int rows, int cols, float min_val, float max_val) {
    for (int i = 0; i < rows * cols; i++) {
        float v = A[i];
        if      (v < min_val) result[i] = min_val;
        else if (v > max_val) result[i] = max_val;
        else                  result[i] = v;
    }
}

void clip_matrix_inplace(float *A, int rows, int cols, float min_val, float max_val) {
    for (int i = 0; i < rows * cols; i++) {
        float v = A[i];
        if      (v < min_val) A[i] = min_val;
        else if (v > max_val) A[i] = max_val;
    }
}