from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START CODE HERE ***

    centroids_init = np.zeros((num_clusters, image.shape[-1]))

    for i in range(num_clusters):

        random.seed(i)
        rand_pixel_row = random.randint(0, image.shape[0])
        rand_pixel_col = random.randint(0, image.shape[1])

        pixel_rgb = image[rand_pixel_row, rand_pixel_col, :]
        centroids_init[i] = pixel_rgb

    # *** END CODE HERE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START CODE HERE ***

    # for iteration in range(max_iter):

    changes_centroid = []
    for iteration in range(30):

        import time

        print("="*25)
        print("Iteration", iteration)
        print("=" * 25)

        time.sleep(2)

        centroid_matrix = np.zeros((image.shape[0], image.shape[1]))

        for i in range(image.shape[0]):

            if i % 20 == 0:
                print("Row:", i)

            for j in range(image.shape[1]):

                closest_cluster = 0
                closest_cluster_dist = 10 ** 10
                for index, centroid in enumerate(centroids):
                    dist = np.linalg.norm(image[i, j, :] - centroid)
                    if dist < closest_cluster_dist:
                        closest_cluster_dist = dist
                        closest_cluster = index

                centroid_matrix[i, j] = closest_cluster

        sum_change_centroid = 0
        for centroid_index in range(len(centroids)):

            centroid_pixel_count = 0
            rgb_sum = np.zeros((3,))

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):

                    if centroid_matrix[i, j] == centroid_index:
                        centroid_pixel_count += 1
                        rgb_sum += image[i, j, :]

            new_centroid = rgb_sum / centroid_pixel_count
            sum_change_centroid = np.linalg.norm(centroids[centroid_index] - new_centroid)

            centroids[centroid_index] = new_centroid

        changes_centroid.append(sum_change_centroid)
        print("\nTotal Change:", changes_centroid[-1])

    new_centroids = centroids

    # *** END CODE HERE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START CODE HERE ***

    updated_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

    for i in range(image.shape[0]):

        if i % 10 == 0:
            print("Row:", i)

        for j in range(image.shape[1]):

            closest_cluster = 0
            closest_cluster_dist = 10 ** 10
            for index, centroid in enumerate(centroids):
                dist = np.linalg.norm(image[i, j, :] - centroid)
                if dist < closest_cluster_dist:
                    closest_cluster_dist = dist
                    closest_cluster = index

            updated_image[i, j] = centroids[closest_cluster]

    # *** END CODE HERE ***

    return updated_image.astype(int)


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
