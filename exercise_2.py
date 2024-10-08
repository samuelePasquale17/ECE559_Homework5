import numpy as np
import matplotlib.pyplot as plt


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


def gradient_z_l(del_f, U):
    """
    Function to calculate gradient_z_l

    gradient_z_l =  U^T * del_f
    :param del_f: dimension 1x1
    :param U: dimension 1x3
    :return: gradient of the loss function respect to z (dimension 3x1)
    """
    # return gradient_z_l
    return np.array(U).T * del_f


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


def initialize_matrix(n, k, m, v):
    """
    Function that initializes the matrix
    :param n: rows
    :param k: cols
    :param m: avg
    :param v: var
    :return:
    """
    std_dev = np.sqrt(v)
    matrix = np.random.normal(loc=m, scale=std_dev, size=(n, k))
    return matrix


def generate_data_and_plot():
    # Generate 1000 random points for x1 and x2 uniformly between -2 and 2
    x1 = np.random.uniform(-2, 2, 1000)
    x2 = np.random.uniform(-2, 2, 1000)

    # Calculate z1, z2, z3 based on the conditions
    z1 = ((x1 - x2 + 1) > 0) * 1
    z2 = ((-x1 - x2 + 1) > 0) * 1
    z3 = ((-x2 - 1) > 0) * 1

    # Use the Heaviside function to determine the label (y)
    y = np.heaviside(z1 + z2 - z3 - 1.5, 1)

    # Plot the points where y == 1 (red) and y == 0 (blue)
    plt.scatter(x1[np.where(y == 1)], x2[np.where(y == 1)], c='red')
    plt.scatter(x1[np.where(y == 0)], x2[np.where(y == 0)], c='blue')

    # Set axis to be equal
    plt.axis('equal')

    # Show the plot
    plt.show()

    # Return x and y for further use
    return x1, x2, y


def plot_mse_epoch(MSE_results, eta):
    """
    Function to plot the MSE vs. epochs
    """
    y_mse = []
    for mse in MSE_results:
        y_mse.append(mse[0][0])

    plt.plot(range(1, len(y_mse) + 1), y_mse, marker='o', linestyle='-', color='b')
    plt.title(f'MSE vs. Epochs with eta = {eta}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.show()


def training_NN_forward_backward(x1, x2, y, num_epochs, W_rows, W_cols, avg, var, b_rows, b_cols, U_rows, U_cols, c_rows, c_cols, a, eta, eta_dim_factor, eta_mod, initRand, initVal, minibatch_en, minibatch_size):
    """
    Function that given a data set, trains the NN with forward and backward propagation
    :param x1: dataset x1
    :param x2: dataset x2
    :param y: dataset y
    :param num_epochs: number of epochs
    :param W_rows: number of rows in W
    :param W_cols: number of columns in W
    :param avg: average for the random generation of weights and biases
    :param var: deviation standard for the random generation of weights and biases
    :param b_rows: number of rows in b
    :param b_cols: number of columns in b
    :param U_rows: number of rows in U
    :param U_cols: number of columns in U
    :param c_rows: number of rows in c
    :param c_cols: number of columns in c
    :param a: parameter of the sigmoid function
    :param eta: learning rate
    :param eta_dim_factor: factor for updating learning algorithm runtime based on the MSE
    :param eta_mod: flag, if True learning algorithm is updated runtime
    :param initRand: flag, if True the initialization of weights and biases is randomly
    :param initVal: if flag initRand is False, all weights and biases are initialized with this value
    :param minibatch_en: flag, if active minibatch or Proper gradient descend is activated
    :param minibatch_size: size of the minibatch, if Proper gradient descend is needed, set the size equal to the number of epochs
    :return: Weights and biases
    """
    if initRand:
        # random initialization of weights and biases
        np.random.seed(1713)
        W = initialize_matrix(W_rows, W_cols, avg, var)
        b = initialize_matrix(b_rows, b_cols, avg, var)
        U = initialize_matrix(U_rows, U_cols, avg, var)
        c = initialize_matrix(c_rows, c_cols, avg, var)
    else:
        # initialization of weights and biases with the given value
        W = np.full((W_rows, W_cols), initVal)
        b = np.full((b_rows, b_cols), initVal)
        U = np.full((U_rows, U_cols), initVal)
        c = np.full((c_rows, c_cols), initVal)

    # MSE list over epochs
    MSE_results = []
    # init MSE prec to inf
    previous_mse = float('inf')

    # run over data (epochs)
    for epoch in range(num_epochs):
        # init mse per epoch
        mse_epoch = 0
        # zeroed-out accumulators for gradient
        grad_w_l_acc = 0
        grad_b_l_acc = 0
        grad_u_l_acc = 0
        grad_c_l_acc = 0

        # pass over data set
        for i, (x1_val, x2_val, y_val) in enumerate(zip(x1, x2, np.array(y))):

            # Forward (x -> vz -> z -> vf -> f)
            x = [[x1_val], [x2_val]]
            vz = v_z(W, x, b)
            z_vect = z(vz, a)
            vf = v_f(U, z_vect, c)
            f_out = f(vf, a)

            # Backward (grad_b_l <- grad_W_l <- delta_z <- grad_z_l <- grad_c_l <- grad_U_l <- delta_f <- grad_f_l)
            grad_f_l  = gradient_f_l(f_out, y_val)
            del_f = delta_f(grad_f_l, vf, a)
            grad_u_l = np.array(del_f) @ np.array(z_vect).T
            grad_c_l = del_f
            grad_z_l = gradient_z_l(del_f, U)
            del_z = delta_z(grad_z_l, vz, a)
            grad_w_l = np.array(del_z) @ np.array(x).T
            grad_b_l = del_z

            # Check if grad's have to be accumulated
            if minibatch_en:
                # accumulate gradients
                grad_w_l_acc += grad_w_l
                grad_b_l_acc += grad_b_l
                grad_u_l_acc += grad_u_l
                grad_c_l_acc += grad_c_l

            # check minibatch size
            if minibatch_en and (i + 1) % minibatch_size == 0:
                # end of the minibatch, update gradients and zero-out accumulators
                W = W - eta * (grad_w_l_acc/minibatch_size)
                b = b - eta * (grad_b_l_acc/minibatch_size)
                U = U - eta * (grad_u_l_acc/minibatch_size)
                c = c - eta * (grad_c_l_acc/minibatch_size)
                # zeroed-out accumulators for gradient
                grad_w_l_acc = 0
                grad_b_l_acc = 0
                grad_u_l_acc = 0
                grad_c_l_acc = 0

            elif not minibatch_en:
                # gradient descend
                W = W - eta * grad_w_l
                b = b - eta * grad_b_l
                U = U - eta * grad_u_l
                c = c - eta * grad_c_l

            # Accumulate squared error for MSE
            mse_epoch += (f_out - y_val) ** 2

        # Compute MSE for the current epoch and add to the list
        mse_epoch /= len(y)  # Divide by the number of samples to get the average
        MSE_results.append(mse_epoch)

        # Check if the MSE is increasing and eta modulable is active
        if mse_epoch > previous_mse and eta_mod:
            # update eta
            eta = eta * eta_dim_factor
        # Update previous MSE for the next epoch
        previous_mse = mse_epoch
    # Plot MSE here
    plot_mse_epoch(MSE_results, var)
    return W, b, U, c


def plot_3D_decision_boundary(x1, x2, y_pred, title):
    fig = plt.figure(figsize=(7, 10))
    plt.title(title)
    ax = plt.axes(projection="3d")
    ax.scatter3D(x1, x2, y_pred, color="green")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.zaxis._axinfo['juggled'] = (2, 2, 1)
    plt.show()


def neural_network(x, W, b, U, c, a):
    """
    Function that given the input x returns the prediction based on its weights and biases
    :param x: x (dimension 2x1)
    :param W: weight matrix (dimension 3x2)
    :param b: bias vector (dimension 3x1)
    :param U: weight matrix (dimension 1x3)
    :param c: bias vector (dimension 1x1)
    :return: y (dimension 1x1)
    """
    return sigmoid(np.array(U) @ np.array(sigmoid(np.array(W) @ np.array(x) + np.array(b), a)) + np.array(c), a)


def main():
    # data set
    x1, x2, y = generate_data_and_plot()

    # training with forward and backward propagation
    W, b, U, c = training_NN_forward_backward(
        x1,  # x1
        x2,  # x2
        y,  # y
        100,  # number of epochs
        3,  # W rows
        2,  # W cols
        0,  # average random generation
        0.1,  # standard deviation random generation
        3,  # b rows
        1,  # b cols
        1,  # U rows
        3,  # U cols
        1,  # c rows
        1,  # c cols
        5,  # sigmoid function parameter
        0.01,  # eta gradient descend
        0.95,  # change factor for eta if MSE is positive and eta_mod flag activated
        False,  # enable eta value modulable based on MSE
        True,  # random initialization of weights and biases
        0,  # if random initialization deactivated, all weights and biases are initialized to initVal
        False,  # enable minibatch, must be true is we want either minibatches or gradient descend with
                            # cumulative gradients, updated after each epoch
        0  # minibatch size, if we want gradient descend with cumulative gradients,
                        # updated after each epoch, set it to the size of the data set
    )

    # compute y_pred over the data set
    y_pred_NN = []
    for x1_val, x2_val in zip(x1, x2):
        y_pred_NN.append(neural_network(np.array([[x1_val], [x2_val]]), W, b, U, c, 5))

    # Visualize the decision boundary in 3D
    plot_3D_decision_boundary(x1, x2, y_pred_NN, "3D decision boundary")

    ###################################################
    ##################### Point f #####################
    ###################################################
    # Possible improvements shown in the report for simplicity
    # Different η values
    # Variable η based on the MSE value
    # Different a parameters of the sigmoid function
    # Different number of epochs
    # Proper gradient descent
    # Minibatches
    # Different architecture
    # Change in the standard deviation for the random generation of weights and biases


main()