import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


def risk(w):
    """
    Function that given the vector w, returns the risk
    R(w) = 13 w1^2 - 10 w1 w2 + 4 w1 + 2 w2^2 - 2 w2 + 1

    :param w: weights vector
    :return: risk
    """
    # return risk
    return 13 * w[0]**2 - 10 * w[0] * w[1] + 4 * w[0] + 2 * w[1]**2 - 2 * w[1] + 1


def gradient_w_R(w):
    """
    Function that given the vector w, returns the gradient of the risk

    :param w: weights vector
    :return: gradient respect to w of the risk
    """
    # return gradient or R in w
    return 26 * w[0] - 10 * w[1] + 4, 4 * w[1] - 2 - 10 * w[0]


def gradient_descend(eta, w_opt, iterations, initRand=True, init_w_val=0):
    """
    Function that implements the gradient descent algorithm, given different parameters
    :param eta: Eta
    :param init_w_val: Initialization value for the weight vector
    :param w_opt: weight vector of the optimality condition
    :param iterations: number of iterations
    :return: final weights, and list with the distance between the current iteration and the optimality condition
    """
    # Initialization
    if initRand:
        w_old = np.random.rand(len(w_opt))
    else:
        w_old = [init_w_val]*len(w_opt)
    # Distances vector
    distances = []

    # run algorithm
    for i in range(iterations):
        # gradient descend formula
        w_new = np.array(w_old) - eta * np.array(gradient_w_R(w_old))
        # compute and save distance with Euclidean distance
        distances.append(np.linalg.norm(np.array(w_new) - np.array(w_old)))
        # update weights
        w_old = w_new

    # Return weights and distances
    return w_old, distances


def plot_distances(x, distances, eta, color, maxY):
    """
    Function that plots the distances over the iterations
    :param x: iterations
    :param ys: distances for a specific eta
    :param etas: correspondingly eta
    :return:
    """
    plt.plot(x, distances, color=color)

    # labels
    plt.xlabel('Iterations')
    plt.ylabel('Distances')

    # title
    plt.title(f"Distance vs. Iterations with eta = {eta}")    # Show the grid for better readability
    plt.grid(True)
    plt.ylim(0, maxY)
    # Display the plot
    plt.show()


def main():
    #############################################################
    ########################## Point a ##########################
    #############################################################
    # optimality condition
    w_opt = [1, 3]
    print("### Point a ### \nThe gradient of R is ", gradient_w_R(w_opt), " when w = (",w_opt[0],",",w_opt[1],")")
    #############################################################
    #############################################################
    #############################################################

    #############################################################
    ########################## Point b ##########################
    #############################################################
    # eta's
    etas = [0.02, 0.05, 0.1]
    # number of iterations
    iterations = 500
    # list of distances per each eta
    distances = []

    # run gradient descend algorithm for different eta's
    for eta in etas:
        # gradient descend algorithm
        w_new, dist = gradient_descend(eta, w_opt, iterations)
        # save distances
        distances.append(dist)

    # Plot distances
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']
    for i in range(len(distances)):
        plot_distances(list(range(1, iterations + 1)), distances[i], etas[i], colors[i], max(distances[i]) * 1.1 )

    #############################################################
    #############################################################
    #############################################################


main()