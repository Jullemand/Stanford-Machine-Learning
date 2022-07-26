import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***

    clf = PoissonRegression()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_valid)
    clf.plot(y_valid, predictions, save_path = "plot.png")

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """

        self.theta = theta_0
        self.theta = np.zeros((4, 1))
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def plot(self, truth, pred, save_path):

        plt.scatter(truth, pred, alpha=.5, s=30)
        plt.show()
        # plt.savefig(save_path)

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        for epoch in range(self.max_iter):

            print(epoch)
            ascent_rule = np.zeros((4, 1))

            for cur_x, cur_y in zip(x, y):

                cur_x = np.array([cur_x]).T
                ascent_rule += (cur_y - np.exp(self.theta.T.dot(cur_x))[0][0]) * cur_x

            theta_temp = self.theta
            self.theta = self.theta + self.step_size * ascent_rule

            dist = np.linalg.norm(theta_temp - self.theta)
            print(dist)
            if dist < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***

        def hypothesis(x): return np.exp(self.theta.T.dot(x))[0][0]

        probabilities = np.array([0])

        for cur_x in x:

            cur_x = np.array([cur_x]).T
            prob = hypothesis(cur_x)
            probabilities = np.vstack((probabilities, np.array([prob])))

        return probabilities[1:]


        # *** END CODE HERE ***

if __name__ == '__main__':
    '''
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
    '''

    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
