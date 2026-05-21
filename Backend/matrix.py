import ctypes
import os
import array

_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "matrix.dll")
lib = ctypes.CDLL(_lib_path)

lib.add_matrices.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.add_matrices.restype = None

lib.sub_matrices.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.sub_matrices.restype = None

lib.matmul.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.matmul.restype = None

lib.transpose_matrix.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.transpose_matrix.restype = None

lib.scalar_multiply.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.scalar_multiply.restype = None

lib.apply_tanh.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_tanh.restype = None

lib.apply_tanh_derivative.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_tanh_derivative.restype = None

lib.apply_sigmoid.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_sigmoid.restype = None

lib.apply_sigmoid_derivative.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_sigmoid_derivative.restype = None

lib.apply_relu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_relu.restype = None

lib.apply_relu_derivative.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.apply_relu_derivative.restype = None

lib.softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.softmax.restype = None

lib.elem_mul.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.elem_mul.restype = None

lib.sigmoid_derivative_post.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.sigmoid_derivative_post.restype = None

lib.tanh_derivative_post.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
lib.tanh_derivative_post.restype = None

lib.clip_matrix.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.clip_matrix.restype = None

lib.clip_matrix_inplace.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.clip_matrix_inplace.restype = None

class LazyMatrix(list):
    def __init__(self, rows, cols, ptr=None, data=None):
        self.rows = rows
        self.cols = cols
        self.ptr = ptr
        self._data = data
        if data is not None:
            self.extend(data)

    def _unflatten(self):
        if self._data is None and self.ptr is not None:
            flat = list(self.ptr)
            self._data = [flat[i:i+self.cols] for i in range(0, len(flat), self.cols)]
            self.extend(self._data)

    def __iter__(self):
        self._unflatten()
        return super().__iter__()

    def __getitem__(self, idx):
        self._unflatten()
        return super().__getitem__(idx)

    def __len__(self):
        if self._data is None:
            return self.rows
        return super().__len__()

def _dimensions(A):
    if isinstance(A, LazyMatrix):
        return A.rows, A.cols
    return len(A), len(A[0])

def _flat(matrix):
    if isinstance(matrix, LazyMatrix) and matrix.ptr is not None:
        return matrix.ptr, None
    flat = []
    for row in matrix:
        flat.extend(row)
    arr = array.array('f', flat)
    address, length = arr.buffer_info()
    ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_float))
    return ptr, arr

def _unflatten(result, rows, cols):
    return LazyMatrix(rows, cols, ptr=result)

def softmax(A):
    ptr_A, arr_A = _flat(A)
    _, size = _dimensions(A)
    result = (ctypes.c_float * size)()
    lib.softmax(ptr_A, result, size)
    return [result[:]]

def add_matrices(A, B):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    ptr_B, arr_B = _flat(B)
    lib.add_matrices(ptr_A, ptr_B, result, rows, cols)
    return _unflatten(result, rows, cols)

def sub_matrices(A, B):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    ptr_B, arr_B = _flat(B)
    lib.sub_matrices(ptr_A, ptr_B, result, rows, cols)
    return _unflatten(result, rows, cols)

def matmul(A, B):
    rowsA, common = _dimensions(A)
    _, colsB = _dimensions(B)
    result = (ctypes.c_float * (rowsA * colsB))()
    ptr_A, arr_A = _flat(A)
    ptr_B, arr_B = _flat(B)
    lib.matmul(ptr_A, ptr_B, result, rowsA, common, colsB)
    return _unflatten(result, rowsA, colsB)

def transpose(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.transpose_matrix(ptr_A, result, rows, cols)
    return _unflatten(result, cols, rows)

def scalar_multiply(A, scalar):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.scalar_multiply(ptr_A, ctypes.c_float(scalar), result, rows, cols)
    return _unflatten(result, rows, cols)

def tanh(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_tanh(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def tanh_derivative(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_tanh_derivative(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def sigmoid(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_sigmoid(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def sigmoid_derivative(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_sigmoid_derivative(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def relu(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_relu(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def relu_derivative(A):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.apply_relu_derivative(ptr_A, result, rows, cols)
    return _unflatten(result, rows, cols)

def elem_mul(A, B):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    ptr_B, arr_B = _flat(B)
    lib.elem_mul(ptr_A, ptr_B, result, rows, cols)
    return _unflatten(result, rows, cols)

def sigmoid_derivative_post(S, grad):
    rows, cols = _dimensions(S)
    result = (ctypes.c_float * (rows * cols))()
    ptr_S,    arr_S    = _flat(S)
    ptr_grad, arr_grad = _flat(grad)
    lib.sigmoid_derivative_post(ptr_S, ptr_grad, result, rows, cols)
    return _unflatten(result, rows, cols)

def tanh_derivative_post(T, grad):
    rows, cols = _dimensions(T)
    result = (ctypes.c_float * (rows * cols))()
    ptr_T,    arr_T    = _flat(T)
    ptr_grad, arr_grad = _flat(grad)
    lib.tanh_derivative_post(ptr_T, ptr_grad, result, rows, cols)
    return _unflatten(result, rows, cols)

def clip_matrix(A, min_val, max_val):
    rows, cols = _dimensions(A)
    result = (ctypes.c_float * (rows * cols))()
    ptr_A, arr_A = _flat(A)
    lib.clip_matrix(ptr_A, result, rows, cols, ctypes.c_float(min_val), ctypes.c_float(max_val))
    return _unflatten(result, rows, cols)