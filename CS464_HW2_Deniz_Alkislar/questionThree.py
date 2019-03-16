import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from math import pow

# k-fold function
def k_fold(X_Y, linear, C=1.0, gamma='auto'):
    k = 5
    fold = int(len(X_Y) / k)
    accuracy = []
    np.random.shuffle(X_Y)
    x, y = (X_Y[:, :-1], X_Y[:, [-1]])
    start, end = (0, fold)

    # continue until end of the last fold reaches end of the data
    while end < len(X_Y):
        # separate train and validation data
        x_validation = x[start:end]
        x_train = np.concatenate((x[:start], x[end:]), axis=0)
        y_validation = y[start:end]
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # train the model and use validation data to find accuracies
        if linear:
            svm = LinearSVC(C=C)
            svm.fit(x_train, np.squeeze(np.asarray(y_train)))
            predict = svm.predict(x_validation)
            accuracy.append(accuracy_score(np.squeeze(np.asarray(y_validation)), predict))
        else:
            svm = SVC(gamma=gamma)
            svm.fit(x_train, np.squeeze(np.asarray(y_train)))
            predict = svm.predict(x_validation)
            accuracy.append(accuracy_score(np.squeeze(np.asarray(y_validation)), predict))

        # move on to next fold
        end += fold
        start += fold
    return sum(accuracy) / len(accuracy)

def find_optimum_param(X_Y):
    # find best C
    limit = 10
    max, optimum_c, optimum_g = 0, 0, 0

    C, accuracies = [pow(10, c) for c in range(-limit, limit + 1)], []
    for c in C:
        accuracy = k_fold(X_Y, True, C=c)
        accuracies.append(accuracy)
        if accuracy > max:
            max = accuracy
            optimum_c = c

    plt.plot(C, accuracies)
    plt.xlabel('C')
    plt.ylabel('Mean accuracy')
    plt.xscale('log', basex=10)
    plt.show()

    max = 0
    accuracies.clear()
    G = [pow(2, g) for g in range(-limit, limit + 1)]
    for g in G:
        accuracy = k_fold(X_Y, False, gamma=g)
        accuracies.append(accuracy)
        if accuracy > max:
            max = accuracy
            optimum_g = g

    plt.plot(G, accuracies)
    plt.xlabel('Gamma')
    plt.ylabel('Mean accuracy')
    plt.xscale('log', basex=2)
    plt.show()

    return optimum_c, optimum_g

# read features and labels
x_y = np.genfromtxt('UCI_Breast_Cancer.csv', delimiter=',')
x_train = x_y[:500][:, [x for x in range(1, 10)]]
x_test = x_y[500:][:, [x for x in range(1, 10)]]
y_train = x_y[:500][:, [-1]]
y_test = x_y[500:][:, [-1]]

X_Y = np.hstack((x_train, y_train))
params = find_optimum_param(X_Y)

# Run the model with optimum c
svm = LinearSVC(C=params[0])
svm.fit(x_train, np.squeeze(np.asarray(y_train)))
predict = svm.predict(x_test)
print(confusion_matrix(np.squeeze(np.asarray(y_test)), predict))
print(params)

# Run the model with optimum gamma
svm = SVC(gamma=params[1])
svm.fit(x_train, np.squeeze(np.asarray(y_train)))
predict = svm.predict(x_test)
print(confusion_matrix(np.squeeze(np.asarray(y_test)), predict))