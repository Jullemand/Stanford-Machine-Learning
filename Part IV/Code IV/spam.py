import collections

import numpy as np

import util
import svm

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For simplicity, you should split on whitespace, not
    punctuation or any other character. For normalization, you should convert
    everything to lowercase.  Please do not consider the empty string (" ") to be a word.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # return 0

    words = message.lower().split(' ')
    # words = list(map(lambda x: x.lower(), words))
    return words

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # return 0

    dictionary = {}
    a = {}

    all_words = []
    for message in messages:
        all_words += get_words(message)

    unique_words = list(np.unique(np.array(all_words)))

    dict_index_counter = 0

    for word in unique_words:

        # if not word.isalnum(): continue

        print(word)

        count = 0
        # for message in messages_lower:
        for message in messages:

            if word in get_words(message):
                count += 1

            if count == 5:
                dictionary[word] = dict_index_counter
                dict_index_counter += 1
                break

        # if word.replace(" ", "") == '': continue

        # occurrences = list(filter(lambda x: word in x, messages))

        #if len(occurrences) > 5:
        #    a[word] = len(occurrences)

    return dictionary

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # return 0

    # columns = []
    matrix = np.zeros((len(messages), len(word_dictionary)), dtype=int)

    for i in range(len(messages)):

        words_message = get_words(messages[i])

        for word, index in word_dictionary.items():

            word_count = words_message.count(word)
            matrix[i, index] = word_count

            # columns.append(list(map(lambda x: x.count(word), get_words(messages))))

    # df = np.column_stack([columns])
    # return df.T
    return matrix

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # return 0

    # count_y_1 = len(labels[np.where(labels == 1)])
    # count_y_0 = len(labels[np.where(labels == 0)])

    count_y_0_words = 0
    count_y_1_words = 0

    numerator_y_0 = np.array([0.] * matrix.shape[1])
    numerator_y_1 = np.array([0.] * matrix.shape[1])

    # Rows (messages)
    for i in range(matrix.shape[0]):

        if labels[i] == 1:
            count_y_1_words += int(matrix[i].sum())
            numerator_y_1 += matrix[i]

        elif labels[i] == 0:
            count_y_0_words += int(matrix[i].sum())
            numerator_y_0 += matrix[i]

    phi_k_y_1 = (np.array([1.] * matrix.shape[1]) + numerator_y_1) / (matrix.shape[1] + count_y_1_words)
    #phi_k_y_1 = (0 + numerator_y_1) / (0 + count_y_1_words)
    phi_k_y_0 = (np.array([1.] * matrix.shape[1]) + numerator_y_0) / (matrix.shape[1] + count_y_0_words)
    #phi_k_y_0 = (0 + numerator_y_0) / (0 + count_y_0_words)
    phi_y = len(labels[np.where(labels == 1)]) / matrix.shape[0]

    return [np.array([phi_k_y_1]).T, np.array([phi_k_y_0]).T, phi_y]

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    # return 0

    phi_k_y_1 = model[0]
    phi_k_y_0 = model[1]
    phi_y = model[2]

    predictions = np.array([0.] * matrix.shape[0])

    for i in range(matrix.shape[0]):

        message = matrix[i]

        '''
        numerator = 0
        denom_right = 0
        for j in range(len(message)):

            if message[j] == 0:
                continue

            numerator *= phi_k_y_1[j] ** message[j]
            # numerator += np.log(phi_k_y_1[j]) * message[j]
            denom_right *= phi_k_y_0[j] ** message[j]
            # denom_right += np.log(phi_k_y_0[j]) * message[j]

        numerator *= phi_y
        denom_right *= (1-phi_y)

        # numerator += np.log(phi_y)
        # denom_right += np.log((1-phi_y))

#       numerator = phi_y * phi_k_y_1.T.dot(message)
#       denom_left = phi_y * phi_k_y_1.T.dot(message)
#       denom_right = (1-phi_y) * phi_k_y_0.T.dot(message)

        # prob = np.log(numerator) / np.log((numerator + denom_right))
        prob = numerator / (numerator + denom_right)
        prob = prob[0]
        '''

        prob_1 = np.log(phi_y)
        prob_0 = np.log(1 - phi_y)

        for j in range(len(message)):

            if message[j] == 0:
                continue

            prob_1 += np.log(phi_k_y_1[j]) * message[j]
            prob_0 += np.log(phi_k_y_0[j]) * message[j]

        predictions[i] = prob_1 > prob_0

    return predictions

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***

    phi_k_y_1 = model[0]
    phi_k_y_0 = model[1]
    phi_y = model[2]

    log_prob = {}

    for i in range(phi_k_y_1.shape[0]):

        pred = np.log(phi_k_y_1[i] / phi_k_y_0[i])
        log_prob[i] = list(pred)[0]

    sorted_list = sorted(log_prob.items(), key=lambda x: x[1], reverse=True)
    sorted_list = list(map(lambda x: x[0], sorted_list))[:5]

    five_words = []
    for index in sorted_list:
        for word, dict_index in dictionary.items():
            if dict_index == index:
                # five_words.append({ word : dict_index })
                five_words.append(word)

    return five_words

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***

    acc_max = 0
    radius_max = 0

    for radius in radius_to_consider:

        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        acc = np.mean(predictions == val_labels)

        if acc > acc_max:
            acc_max = acc
            radius_max = radius

    return radius_max

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

#    dictionary = create_dictionary(train_messages)
    dictionary = util.read_json("spam_dictionary")

    print('Size of dictionary: ', len(dictionary))

#    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()
