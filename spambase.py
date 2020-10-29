
"""
Oliver Stappas, 1730124
Tuesday, April 13
R. Vincent, instructor
Assignment 4
"""

from extra_trees import extra_trees # suggestion!
from classifier import data_item, normalize_dataset
from random import shuffle
from knnclassifier import knnclassifier

fp = open('spambase.data')
dataset = []
for line in fp:
    fields = line.split(',')
    data = [float(x) for x in fields[:-1]]
    label = int(fields[-1])
    dataset.append(data_item(label, data))

print("Read {} items.".format(len(dataset)))
print("{} features per item.".format(len(dataset[0].data)))


# Add your code here...
GOOD_LABEL = 0
SPAM_LABEL = 1

def evaluate(dataset, cls, n_folds, **kwargs):
    '''Training and evaluating a classifier. Returns the confusion matrix entries
    to be used in later methods.'''
    n_features = len(dataset[0].data)  # Number of features per email
    test_size = len(dataset) // n_folds  # Number of emails for testing
    n_true_positive = n_true_negative = n_false_positive = n_false_negative = 0
    for fold in range(n_folds):
        shuffle(dataset)  # Shuffle to maximize testing data
        train_data, test_data = dataset[test_size:], dataset[:test_size]
        c = cls(**kwargs)  # Create an instance of classifier class with given arguments
        c.train(train_data)
        for test_email in test_data:
            predicted_label = c.predict(test_email.data)
            if predicted_label == test_email.label:
                if predicted_label == GOOD_LABEL:
                    n_true_negative += 1
                elif predicted_label == SPAM_LABEL:
                    n_true_positive += 1
            else:
                if predicted_label == GOOD_LABEL:
                    n_false_negative += 1
                elif predicted_label == SPAM_LABEL:
                    n_false_positive += 1
    return n_true_positive, n_true_negative, n_false_positive, n_false_negative

def print_confusion_matrix(n_true_positive, n_true_negative, n_false_positive, n_false_negative):
    '''Prints the confusion matrix'''
    print(n_true_negative, n_false_negative)
    print(n_false_positive, n_true_positive)

def calculate_TPR(n_true_positive, n_false_negative):
    '''Returns the true positive rate'''
    return (n_true_positive / (n_true_positive + n_false_negative))

def calculate_FPR(n_false_positive, n_true_negative):
    '''Returns the false positive rate'''
    return (n_false_positive / (n_false_positive + n_true_negative))

def print_results(n_true_positive, n_true_negative, n_false_positive, n_false_negative):
    '''Prints all test results of the classifier experiment'''
    print_confusion_matrix(n_true_positive, n_true_negative, n_false_positive, n_false_negative)
    print(calculate_TPR(n_true_positive, n_false_negative))
    print(calculate_FPR(n_false_positive, n_true_negative)) 

def experiment(dataset, cls, **kwargs):
    '''Tests the classifier with given parameters'''
    n_true_positive, n_true_negative, n_false_positive, n_false_negative = evaluate(dataset, cls, n_folds = 5, **kwargs)
    print("Results from {} with arguments {}".format(cls.__name__, kwargs))
    print_results(n_true_positive, n_true_negative, n_false_positive, n_false_negative)

# Experimenting with extra_trees classifier
experiment(dataset, extra_trees, M = 15, K = 10, Nmin = 2)
experiment(dataset, extra_trees, M = 15, K = 10, Nmin = 1)
experiment(dataset, extra_trees, M = 15, K = 10, Nmin = 3)
experiment(dataset, extra_trees, M = 15, K = 10, Nmin = 4)
experiment(dataset, extra_trees, M = 15, K =  5, Nmin = 2)
experiment(dataset, extra_trees, M = 15, K = 15, Nmin = 2)
experiment(dataset, extra_trees, M = 15, K = 25, Nmin = 2)
experiment(dataset, extra_trees, M = 10, K = 10, Nmin = 2)
experiment(dataset, extra_trees, M = 20, K = 10, Nmin = 2)
experiment(dataset, extra_trees, M = 25, K = 10, Nmin = 2)

# Experimenting with kNN classifier
experiment(dataset, knnclassifier, K = 1)
experiment(dataset, knnclassifier, K = 2)
experiment(dataset, knnclassifier, K = 3)
experiment(dataset, knnclassifier, K = 4)
experiment(dataset, knnclassifier, K = 5)
experiment(dataset, knnclassifier, K = 6)
experiment(dataset, knnclassifier, K = 7)
experiment(dataset, knnclassifier, K = 8)
experiment(dataset, knnclassifier, K = 9)
experiment(dataset, knnclassifier, K = 10)
