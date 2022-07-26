import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***

    softmax_matrix = np.zeros((x.shape[0], x.shape[1]))

    # m = np.max(x, axis = 1)

    for row_index in range(x.shape[0]):

        m = np.max(x[row_index])
        denom = np.sum(np.exp(x[row_index] - m))

        for col_index in range(x.shape[1]):

            numerator = np.exp(x[row_index, col_index] - m)
            softmax_matrix[row_index, col_index] = numerator / denom

    return softmax_matrix

    # return np.exp(x - np.max(x, axis=1)) / np.sum(np.exp(x - np.max(x, axis=1)), axis=0)


    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***

    # return np.exp(x/2) / (np.exp(x/2) + np.exp(-x/2))
    return 1 / (1 + np.exp(-x))

    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***

    W1 = np.random.randn(input_size, num_hidden)
    b1 = np.array([0.] * num_hidden)
    # b1 = np.array([0.] * num_hidden).reshape((num_hidden, 1))

    W2 = np.random.randn(num_hidden, num_output)
    b2 = np.array([0.] * num_output)
    # b2 = np.array([0.] * num_output).reshape((num_output, 1))

    return {'W2': W2, 'W1': W1, 'b1': b1, 'b2': b2}

    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***

    z_1 = data @ params['W1'] + params['b1']
    a_1 = sigmoid(z_1)

    z_2 = a_1 @ params['W2'] + params['b2']
    output = softmax(z_2)

    CE = 0
    for i in range(len(labels)):
        CE -= labels[i].dot(np.log(output[i])) / len(labels)

    return a_1, output, CE

    # *** END CODE HERE ***

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***

    a_1, output, cost = forward_prop_func(data, labels, params)
    batch_size_norm = 1/len(data)

    d_Z2 = (output - labels)
    d_W2 = batch_size_norm * (a_1.T @ d_Z2)
    d_b2 = batch_size_norm * (np.sum(d_Z2, axis=0, keepdims=True))

    d_Z1 = d_Z2 @ params['W2'].T * (a_1 * (1 - a_1))
    d_W1 = batch_size_norm * (data.T @ d_Z1)
    d_b1 = batch_size_norm * np.sum(d_Z1, axis=0, keepdims=True)

    return {'W2': d_W2, 'W1': d_W1, 'b1': d_b1.reshape(300,), 'b2': d_b2.reshape(10,)}

    # *** END CODE HERE ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***

    a_1, output, cost = forward_prop_func(data, labels, params)
    batch_size_norm = 1/len(data)

    d_Z2 = (output - labels)
    d_W2 = batch_size_norm * (a_1.T @ d_Z2) + 2 * reg * params['W2']
    d_b2 = batch_size_norm * (np.sum(d_Z2, axis=0, keepdims=True))

    d_Z1 = d_Z2 @ params['W2'].T * (a_1 * (1 - a_1))
    d_W1 = batch_size_norm * (data.T @ d_Z1) + 2 * reg * params['W1']
    d_b1 = batch_size_norm * np.sum(d_Z1, axis=0, keepdims=True)

    return {'W2': d_W2, 'W1': d_W1, 'b1': d_b1.reshape(300,), 'b2': d_b2.reshape(10,)}


    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***

    params['W2'] -= learning_rate * backward_prop_func(train_data, train_labels, params, forward_prop_func)['W2']
    params['b2'] -= learning_rate * backward_prop_func(train_data, train_labels, params, forward_prop_func)['b2']
    params['W1'] -= learning_rate * backward_prop_func(train_data, train_labels, params, forward_prop_func)['W1']
    params['b1'] -= learning_rate * backward_prop_func(train_data, train_labels, params, forward_prop_func)['b1']

    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    """
    Train model using gradient descent for specified number of epochs.
    
    Evaluates cost and accuracy on training and dev set at the end of each epoch.

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        dev_data: A numpy array containing the dev data
        dev_labels: A numpy array containing the dev labels
        get_initial_params_func: A function to initialize model parameters
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API
        num_hidden: Number of hidden layers
        learning_rate: The learning rate
        num_epochs: Number of epochs to train for
        batch_size: The amount of items to process in each batch

    Returns: 
        params: A dict of parameter names to parameter values for the trained model
        cost_train: An array of training costs at the end of each training epoch
        cost_dev: An array of dev set costs at the end of each training epoch
        accuracy_train: An array of training accuracies at the end of each training epoch
        accuracy_dev: An array of dev set accuracies at the end of each training epoch
    """

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    """Predict labels and compute accuracy for held-out test data"""
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    """Convert labels from integers to one hot encoding"""
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    """Load images and labels"""
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    """Trains model, applies model to test data, and (optionally) plots loss"""
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):

    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    # p = np.random.permutation(10000)
    train_data = train_data[p, :]
    train_labels = train_labels[p, :]

    dev_data = train_data[0:10000, :]
    dev_labels = train_labels[0:10000, :]
    train_data = train_data[10000:, :]
    train_labels = train_labels[10000:, :]

    '''
    a = 10000
    dev_data = train_data[0:a,:]
    dev_labels = train_labels[0:a,:]
    train_data = train_data[a:,:]
    train_labels = train_labels[a:,:]
    '''

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        functools.partial(backward_prop_regularized, reg=0.0001),
        args.num_epochs, plot)
        
    # return baseline_acc, reg_acc

def test():

    m = 8000
    num_input = 784
    num_hidden = 300
    num_output = 10
    np.random.seed(100)

    labels = np.zeros((m, 10))
    for i in range(m):

        a = np.random.randint(0, 10)
        labels[i, a] = 1

    data = np.random.randn(m, 784)
    params = get_initial_params(num_input, num_hidden, num_output)

    # a_1, output, cost = forward_prop(data, labels, params)

    reg = 0.1
    learning_rate = 5
    costs = []

    for i in range(10):

        costs.append(forward_prop(data, labels, params)[2])
        print(costs[-1])

        gradients = backward_prop(data, labels, params, forward_prop)
        for param in params:
            params[param] -= learning_rate * gradients[param]

    return

if __name__ == '__main__':
    main()
    # test()
