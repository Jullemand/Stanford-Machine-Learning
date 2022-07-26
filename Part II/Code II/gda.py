import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***

    clf = GDA()
    clf.fit(x_train, y_train)

    if valid_path == "ds1_valid.csv": clf.plot(x_valid, y_valid, save_path = "gda_plot_ds1.png")
    if valid_path == "ds2_valid.csv": clf.plot(x_valid, y_valid, save_path = "gda_plot_ds2.png")

    predictions = clf.predict(x_valid)
    np.savetxt(save_path, predictions, delimiter=', ')

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """

        if not theta_0:
            self.theta = np.zeros((3, 1))
        else:
            self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def plot(self, x, y, save_path):
        util.plot(x, y, self.theta, save_path)

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        indicator_y_1 = 0
        indicator_y_0 = 0
        mu_1 = np.zeros((2, 1))
        mu_0 = np.zeros((2, 1))

        for cur_x, cur_y in zip(x, y):

            cur_x = np.array([cur_x]).T
            if cur_y == 1:
                indicator_y_1 += 1
                mu_1 += cur_x

            elif cur_y == 0:
                indicator_y_0 += 1
                mu_0 += cur_x

        phi = indicator_y_1 / x.shape[0]
        mu_0 = mu_0 / indicator_y_0
        mu_1 = mu_1 / indicator_y_1

        sigma = np.zeros((2, 2))
        for cur_x, cur_y in zip(x, y):

            cur_x = np.array([cur_x]).T
            if cur_y == 1: sigma = sigma + np.matmul((cur_x - mu_1), (cur_x - mu_1).T)
            else: sigma = sigma + np.matmul((cur_x - mu_0), (cur_x - mu_0).T)
            # else: sigma += (cur_x - mu_0).dot((cur_x - mu_0).T)

        sigma /= x.shape[0]
        sigma_inv = np.linalg.inv(sigma)

        # self.theta_0 = np.log((1-phi)/phi) + 0.5 * (mu_1.T.dot(sigma_inv).dot(mu_1) - mu_0.T.dot(sigma_inv).dot(mu_0))
        self.theta_0 = - np.log((1-phi)/phi) + 0.5 * (mu_0.T.dot(sigma_inv).dot(mu_0) - mu_1.T.dot(sigma_inv).dot(mu_1))
        # self.theta = (mu_0 - mu_1).T.dot(sigma_inv)
        self.theta = (mu_1 - mu_0).T.dot(sigma_inv)
        # self.theta = np.matmul((mu_0 - mu_1).T, sigma_inv)
        self.theta = np.array([np.stack((self.theta_0[0][0], self.theta[0][0], self.theta[0][1]))]).T

        # self.theta = np.array([[2.13511208], [-1.51635702], [-0.01279913]])

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        # *** START CODE HERE ***

        probabilities = np.array([0])

        def sigmoid(z): return 1 / (1 + np.exp(-z))
        # def hypothesis(x): return sigmoid(self.theta.T.dot(x))[0][0]
        def hypothesis(x): return sigmoid(self.theta.T.dot(x))

        for cur_x in x:

            cur_x = np.array([cur_x]).T
            prob = hypothesis(cur_x)
            # probabilities = np.vstack((probabilities, np.array([prob])))
            probabilities = np.vstack((probabilities, prob))

        #return np.squeeze(probabilities[1:], 1)
        return probabilities[1:]

        # *** END CODE HERE

if __name__ == '__main__':
    '''
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')
    '''

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')