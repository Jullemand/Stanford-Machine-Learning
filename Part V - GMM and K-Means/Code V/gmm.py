import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)

    # *** START CODE HERE ***

    n, dim = x.shape

    x_copy = np.copy(x)
    np.random.shuffle(x_copy)
    x_copy = np.array_split(x_copy, K)

    mu = [np.mean(x_copy[i], axis=0) for i in range(0, K)]
    sigma = [np.cov(x_copy[i].T) for i in range(0, K)]
    phi = np.array([1 / K for _ in range(0, K)])

    w = np.array([[1 / K for _ in range(0, K)]] * len(x))

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):

        # (1) E-step: Update your estimates in w

        # (2) M-step: Update the model parameters phi, mu, and sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # *** START CODE HERE ***

        # Such that autograder avoids timeout
        if it > 50:
            break

        n = len(x)
        K = w.shape[1]

        prev_ll = ll
        ll = 0
        for i, x_iter in enumerate(x):

            inner_sum = 0
            for j in range(K):
                inner_sum += multi_variate_gaussian(x=x_iter, mu=mu[j], sigma=sigma[j]) * phi[j]
            ll += np.log(inner_sum)

        # E-STEP

        for i, x_iter in enumerate(x):

            denom = 0
            for l in range(K):
                denom += multi_variate_gaussian(x=x_iter, mu=mu[l], sigma=sigma[l]) * phi[l]

            for j in range(K):
                numerator = multi_variate_gaussian(x=x_iter, mu=mu[j], sigma=sigma[j]) * phi[j]
                w[i, j] = numerator / denom

        # M-STEP

        # PHI

        phi = np.sum(w, axis=0) / len(x)

        '''
        for j in range(K):

            sum = 0
            for i, x_iter in enumerate(x):
                sum += w[i, j] / n
            phi[j] = sum
        '''

        # MU
        for j in range(K):

            numerator = np.zeros(x[0].shape)
            # denom = np.zeros(mu[j].shape)
            # denom = 0
            denom = np.sum(w, axis=0)
            numerator = w.T[j].dot(x)

            '''
            for i, x_iter in enumerate(x):
                numerator += w[i, j] * x_iter
                # denom += w[i, j]
            '''

            mu[j] = numerator / denom[j]

        # SIGMA
        for j in range(K):

            numerator = np.zeros(sigma[j].shape)
            # denom = 0
            denom = np.sum(w, axis=0)

            for i, x_iter in enumerate(x):
                x_vec_diff = np.array([x_iter]).T - np.array([mu[j]]).T

                numerator += w[i, j] * (x_vec_diff @ x_vec_diff.T)
                # denom += w[i, j]

            sigma[j] = numerator / denom[j]

        if it > 0 and (it - 1) % 5 == 0:
            # print("Iteration:", it, "Likelihood diff:", abs(ll-prev_ll))
            print("Iteration:", it - 1, "Likelihood:", ll)

        it += 1

        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):

        # (1) E-step: Update your estimates in w

        # (2) M-step: Update the model parameters phi, mu, and sigma

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

        # *** START CODE HERE ***

        if it > 150:
            break

        n = len(x)
        n_tilde = len(x_tilde)
        K = w.shape[1]

        prev_ll = ll
        ll = 0
        for i, x_iter in enumerate(x):

            # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.

            inner_sum = 0
            for j in range(K):
                inner_sum += multi_variate_gaussian(x=x_iter, mu=mu[j], sigma=sigma[j]) * phi[j]
            ll += np.log(inner_sum)

        for i, x_iter in enumerate(x_tilde):

            inner_sum = 0
            for j in range(K):
                if int(z_tilde[i]) == j:
                    inner_sum += multi_variate_gaussian(x=x_iter, mu=mu[j], sigma=sigma[j]) * phi[j]

            ll += alpha * np.log(inner_sum)

        # E-STEP

        for i, x_iter in enumerate(x):

            denom = 0
            for l in range(K):
                denom += multi_variate_gaussian(x=x_iter, mu=mu[l], sigma=sigma[l]) * phi[l]

            for j in range(K):
                numerator = multi_variate_gaussian(x=x_iter, mu=mu[j], sigma=sigma[j]) * phi[j]
                w[i, j] = numerator / denom

        # PHI

        for j in range(K):

            left_sum = 0
            right_sum = 0
            for i, x_iter in enumerate(x):
                left_sum += w[i, j] / (n + alpha * n_tilde)

            for i, x_iter in enumerate(x_tilde):
                if int(z_tilde[i]) == j:
                    right_sum += alpha / (n + alpha * n_tilde)

            phi[j] = left_sum + right_sum

        # MU
        for j in range(K):
            numerator = np.zeros(x[0].shape)
            # denom = np.zeros(mu[j].shape)
            # denom = 0
            # denom = np.sum(w, axis=0)
            # numerator = w.T[j].dot(x)

            for i, x_iter in enumerate(x):
                numerator += w[i, j] * x_iter
                denom += w[i, j]

            for i, x_iter in enumerate(x_tilde):
                if int(z_tilde[i]) == j:
                    numerator += alpha * x_iter
                    denom += alpha

            mu[j] = numerator / denom

        # SIGMA
        for j in range(K):

            numerator = np.zeros(sigma[j].shape)
            denom = 0
            # denom = np.sum(w, axis=0)

            for i, x_iter in enumerate(x):
                x_vec_diff = np.array([x_iter]).T - np.array([mu[j]]).T

                numerator += w[i, j] * (x_vec_diff @ x_vec_diff.T)
                denom += w[i, j]

            for i, x_iter in enumerate(x_tilde):
                if int(z_tilde[i]) == j:
                    x_vec_diff = np.array([x_iter]).T - np.array([mu[j]]).T

                    numerator += alpha * (x_vec_diff @ x_vec_diff.T)
                    denom += alpha

            sigma[j] = numerator / denom

        if it > 0 and (it - 1) % 1 == 0:
            # print("Iteration:", it, "Likelihood diff:", abs(ll-prev_ll))
            print("Iteration:", it - 1, "Likelihood:", ll)

        it += 1


        # *** END CODE HERE ***

    return w

# Helper functions
# *** START CODE HERE ***

def multi_variate_gaussian(x, mu, sigma):
    x = np.array([x]).T
    mu = np.array([mu]).T

    size = len(x)
    det = np.linalg.det(sigma)
    norm_const = 1.0 / (((2 * np.pi) ** (float(size) / 2)) * (det ** (1.0 / 2)))
    x_mu = x - mu
    inv = np.linalg.pinv(sigma)

    result = np.exp(-0.5 * (x_mu.T @ inv) @ x_mu)
    return (norm_const * result)[0][0]

# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # Once you've implemented the semi-supervised version,
        # uncomment the following line:

        main(is_semi_supervised=True, trial_num=t)
