import numpy as np


def gradient_f_l(f_var, y):
    """
    Function to calculate the gradient of the loss function l = (f - y)^2

    grad_f_l = 2 * (f - y)

    :param f_var:
    :param y:
    :return:
    """
    # return gradient of l respect to f
    return 2 * (f_var - y)


def delta_f(grad_f_l, vf, a):
    """
    Function that return the delta_f as

    delta_f = gradient_f_l * sigmoid_derivative(vf)

    :param grad_f_l: gradient of the loss respect to f (dimension 1x1)
    :param vf: vf = UZ + C
    :param a: parameter needed for the sigmoid function
    :return: delta_f (dimension 1x1)
    """
    # return delta f
    return grad_f_l * sigmoid_derivative(vf, a)


def gradient_z_l(U, Z, C, Y, a):
    """
    Function to calculate gradient_z_l

    gradient_z_l = 2 * ( sigmoid(UZ + C) - Y ) * sigmoid_derivative(UZ + C) * U^T
    :param U: matrix 1x3
    :param Z: matrix 3x1
    :param C: val 1x1
    :param Y: val 1x1
    :param a: sigmoid parameter
    :return: gradient of the loss function respect to z (dimension 3x1)
    """
    vf = v_f(U, Z, C)
    ret = []
    val = 2 * (sigmoid(vf, a) - Y) * sigmoid_derivative(vf, a)
    for element in U:
        ret.append(element * val)

    # return gradient_z_l
    return np.array(ret).reshape(-1, 1)


def delta_z(grad_z_l, vz, a):
    """
    Function that return the delta_z as

    delta_z = gradient_z_l * sigmoid_derivative(vz)^T

    :param grad_z_l: gradient of the loss function respect to z (dimension 3x1)
    :param vz: dimension 3x1
    :param a: sigmoid parameter
    :return: delta_z (dimension 3x1)
    """
    sig_der = []
    for element in np.array(vz).reshape(1, -1):
        sig_der.append(sigmoid_derivative(element, a))

    # return delta_z
    res = []
    for i in range(len(sig_der)):
        res.append(sig_der[i] * grad_z_l[i][0])

    return np.array(res).reshape(-1, 1)

def v_z(W, X, b):
    """
    Function that computes vz = Wx + b
    :param W: weight matrix k X n (k = biases, n = inputs)
    :param X: input vector n X 1 (n = inputs)
    :param b: biases vector k x 1 (k = biases)
    :return: vz vector k x 1 (k = biases)
    """
    # return vz
    return np.dot(np.array(W), np.array(X)) + np.array(b)


def z(vz, a):
    """
    Function that computes z = sigmoid(vz)
    :param vz: vector k x 1 (k = biases)
    :param a: parameter of sigmoid function
    :return: z vector k x 1 (k = biases)
    """
    # return z
    ret = []
    for element in vz:
        ret.append([sigmoid(element[0], a)])

    return ret


def v_f(U, Z, C):
    """
    Function that computes vf = UZ + C
    :param U: weight matrix 1 x k (k = biases)
    :param Z: Z vector k x 1 (k = biases)
    :param C: bias 1 x 1
    :return: vf value 1 x 1
    """
    # return vf
    return v_z(U, Z, C)


def f(vf, a):
    """
    Function that computes f = sigmoid(vf)
    :param vf: value 1 x 1
    :param a: parameter of sigmoid function
    :return: f value 1 x 1
    """
    # return f
    return sigmoid(vf, a)


def sigmoid(x, a):
    """
    Function to calculate the sigmoid function with parameter a
    :param x: x
    :param a: a parameter
    :return:
    """
    # return the sigmoid function given x and a
    return 1 / (1 + np.exp(-a * x))


def sigmoid_derivative(x, a):
    """
    Function to calculate the derivative of the sigmoid function
    :param x:
    :param a: a parameter
    :return:
    """
    # compute the sigmoid function given x and a
    sig = sigmoid(x, a)
    # return the derivative
    return a * sig * (1 - sig)


def main():
    print("hello")


main()
