import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***

    logReg = LogisticRegression()
    logReg.fit(x_train, y_train)

    if valid_path == "ds1_valid.csv": logReg.plot(x_test, y_test, save_path = "logreg_plot_ds1.png")
    if valid_path == "ds2_valid.csv": logReg.plot(x_test, y_test, save_path = "logreg_plot_ds2.png")

    logReg.plot(x_test, y_test, save_path="plot.png")
    predictions = logReg.predict(x_test)
    np.savetxt(save_path, predictions, delimiter=', ')

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        def sigmoid(z): return 1 / (1 + np.exp(-z))
        def hypothesis(x): return sigmoid(self.theta.T.dot(x))[0][0]

        errors = []
        # runs = 800

        for i in range(self.max_iter):

            grad = np.zeros((3, 1))
            hessian = np.zeros((3, 3))

            error = 0
            for cur_x, cur_y in zip(x, y):

                cur_x = np.array([cur_x]).T

                # errors.append(y - hypothesis(x))
                # error = error - ((cur_y * np.log(hypothesis(cur_x))) + (1-cur_y)*np.log(1-hypothesis(cur_x)))
                error = 0
                # errors.append( -((cur_y * np.log(hypothesis(cur_x))) + (1-cur_y)*np.log(1-hypothesis(cur_x))))

                grad += (cur_y - hypothesis(cur_x)) * cur_x
                hessian += hypothesis(cur_x)*(1-hypothesis(cur_x))* (cur_x.dot(cur_x.T))

                # grad = y * (hypothesis(x) - 1) * x + hypothesis(x)*x*(1-y)
                # self.theta =- self.step_size * grad

            errors.append(error / x.shape[0])
            if self.verbose:
                print(error / x.shape[0])

            grad = grad * (1/x.shape[0])
            hessian = hessian * (1/x.shape[0])
            hessian_inv = np.linalg.inv(hessian)

            theta_temp = self.theta
            self.theta = self.theta + hessian_inv.dot(grad)

            dist = np.linalg.norm(theta_temp - self.theta)
            if dist < self.eps:
                break

        # plt.plot(errors)
        # plt.show()

        1

        # *** END CODE HERE ***
    def plot(self, x, y, save_path):
        util.plot(x, y, self.theta, save_path)

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # self.theta = np.array([[-2.41],[1.034],[0.2446]])

        def sigmoid(z): return 1 / (1 + np.exp(-z))
        def hypothesis(x): return sigmoid(self.theta.T.dot(x))[0][0]

        probabilities = np.array([0])

        for cur_x in x:

            cur_x = np.array([cur_x]).T
            prob = hypothesis(cur_x)
            probabilities = np.vstack((probabilities, np.array([prob])))

        return np.squeeze(probabilities[1:], 1)

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    # main(train_path='ds2_train.csv',
    #      valid_path='ds2_valid.csv',
    #      save_path='logreg_pred_2.txt')
