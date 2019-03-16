from numpy import genfromtxt, dot, hstack, ones, matrix, zeros, concatenate, exp, ndarray, lexsort
from numpy.random import shuffle
import matplotlib.pyplot as plt

IT_COUNT = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
L_RATE = [0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

# hypothesis function
def hypo(X, beta):
    z = dot(X, beta)
    return 1 / (1 + exp(-z))

# finding gradients
def find_gradients(X, Y, beta):
    n = len(X)
    gradients = dot(X.T, (hypo(X, beta) - Y))
    return -gradients / n

# gradient ascent
def gradient_ascent(X, Y, it_count, l_rate):
    beta = zeros((X.shape[1], 1))
    for i in range(it_count):
        beta += l_rate * find_gradients(X, Y, beta)
    return beta

# function for finding TP, FP, TN, FN
def confusion_matrix(X, Y, beta, show=False):
    predict = exp(dot(X, beta)) > 1
    comparison = hstack((predict, Y))

    TP, FP, TN, FN = (0, 0, 0, 0)
    for i in range(len(comparison)):
        if comparison.item(i, 0) + comparison.item(i, 1) == 2:
            TP += 1
        elif comparison.item(i, 0) + comparison.item(i, 1) == 0:
            TN += 1
        elif comparison.item(i, 0) == 1 and comparison.item(i, 1) == 0:
            FP += 1
        elif comparison.item(i, 0) == 0 and comparison.item(i, 1) == 1:
            FN += 1

    if show:
        print("  \t", "(+)", "(-)")
        print("(+)\t", TP, " ", FP)
        print("(-)\t", FN, " ", TN)

    return TP, FP, FN, TN

# function for evaluating the performance
def accuracy_rate(X, Y, beta):
    predict = exp(dot(X, beta)) > 1
    comparison = hstack((predict, Y))
    wrong_predicts = 0
    for i in range(len(predict)):
        if comparison.item(i, 0) != comparison.item(i, 1):
            wrong_predicts += 1
    return 1 - (wrong_predicts / len(Y))

# k-fold function
def k_fold(X_Y, it_count, l_rate):
    k = 5
    fold = int(len(X_Y) / k)
    accuracy = []
    shuffle(X_Y)
    x, y = (X_Y[:, :-1], X_Y[:, [-1]])
    start, end = (0, fold)

    # continue until end of the last fold reaches end of the data
    while end < len(X_Y):
        # separate train and validation data
        x_validation = x[start:end]
        x_train = concatenate((x[:start], x[end:]), axis=0)
        y_validation = y[start:end]
        y_train = concatenate((y[:start], y[end:]), axis=0)

        # train the model and use validation data to find accuracies
        new_beta = gradient_ascent(x_train, y_train, it_count, l_rate)
        accuracy.append(accuracy_rate(x_validation, y_validation, new_beta))

        # move on to next fold
        end += fold
        start += fold
    return sum(accuracy) / len(accuracy)

def find_optimum_param(X_Y):
    accuracies = {}
    for i in range(len(IT_COUNT)):
        for j in range(len(L_RATE)):
            accuracies[(IT_COUNT[i], L_RATE[j])] = k_fold(X_Y, IT_COUNT[i], L_RATE[j])
            print((IT_COUNT[i], L_RATE[j]), accuracies[(IT_COUNT[i], L_RATE[j])])
    return accuracies

def forward_elimination(X_Y, it_count, l_rate):
    selected = [0]
    current_accuracy = 0
    best_set, remaining = ([], [])
    for i in range(1, X_Y.shape[1] - 1):
        remaining.append(i)

    while True:
        prev_accuracy = current_accuracy
        best_set.clear()
        for i in range(len(remaining)):
            if remaining[i] not in selected:
                current_features = selected + [remaining[i]]
                local_max = k_fold(X_Y[:, current_features + [X_Y.shape[1] - 1]], it_count, l_rate)
                if local_max > current_accuracy:
                    best_set = current_features
                    current_accuracy = local_max
                    print("Local max:", current_features, local_max)

        remaining = [x for x in remaining if x not in best_set]
        selected += [x for x in best_set if x not in selected]
        print("Selected:", selected)
        print("Max accuracy:", current_accuracy)
        if current_accuracy <= prev_accuracy:
            break
    return selected

def backward_elimination(X_Y, it_count, l_rate):
    current_accuracy = 0
    best_set, selected = ([], [])
    for i in range(1, X_Y.shape[1] - 1):
        selected.append(i)

    while True:
        prev_accuracy = current_accuracy
        best_set.clear()
        for i in range(len(selected)):
            print("Feature:", selected[i])
            tested_set = [x for x in selected if x not in [selected[i]]]
            local_max = k_fold(X_Y[:, [0] + tested_set + [X_Y.shape[1] - 1]], it_count, l_rate)
            if local_max > current_accuracy:
                best_set = tested_set
                current_accuracy = local_max
                print("Removed:", selected[i], "Max:", local_max)

        selected = best_set
        print("Current accuracy:", current_accuracy)
        if current_accuracy <= prev_accuracy:
            break
    return selected

# read feature set
ovariancancer = genfromtxt('ovariancancer.csv', delimiter=',')
ovariancancer = hstack((ones((len(ovariancancer), 1)), ovariancancer))
x_train = concatenate((ovariancancer[20:121], ovariancancer[141:]))
x_test = concatenate((ovariancancer[:20], ovariancancer[121:141]))

# read labels
ovariancancer_labels = genfromtxt('ovariancancer_labels.csv')
ovariancancer_labels = (matrix(ovariancancer_labels)).T
y_train = concatenate((ovariancancer_labels[20:121], ovariancancer_labels[141:]))
y_test = concatenate((ovariancancer_labels[:20], ovariancancer_labels[121:141]))
X_Y = hstack((x_train, y_train))

# find optimum parameters
accuracies = find_optimum_param(X_Y)
max = 0
opt = 0
for key, value in accuracies.items():
    if value > max:
        max = value
        opt = key
print(opt)

# forward elimination and testing
forward_selected = forward_elimination(X_Y, opt[0], opt[1])
beta = gradient_ascent(x_train[:, forward_selected], y_train, opt[0], opt[1])
confusion_matrix(x_test[:, forward_selected], y_test, beta, show=True)

# backward elimination and testing
backward_selected = forward_elimination(X_Y, opt[0], opt[1])
beta = gradient_ascent(x_train[:, backward_selected], y_train, opt[0], opt[1])
confusion_matrix(x_test[:, backward_selected], y_test, beta, show=True)

# curve drawing process
X_Y_test = hstack((hypo(x_test[:, forward_selected], beta), y_test))
temp = X_Y_test.view(ndarray)
X_Y_test = temp[lexsort((temp[:, 0],))]

knob = 0
roc_x, roc_y, pr_x, pr_y = ([], [], [], [])
for m in range(len(X_Y_test)):
    predict = []
    for j in range(len(X_Y_test)):
        if X_Y_test.item(j, 0) >= knob:
            predict.append(1)
        else:
            predict.append(0)
    predict = matrix(predict).T
    comparison = hstack((predict, X_Y_test[:, [-1]]))
    TP, FP, TN, FN = (0, 0, 0, 0)
    for i in range(len(comparison)):
        if comparison.item(i, 0) + comparison.item(i, 1) == 2:
            TP += 1
        elif comparison.item(i, 0) + comparison.item(i, 1) == 0:
            TN += 1
        elif comparison.item(i, 0) == 1 and comparison.item(i, 1) == 0:
            FP += 1
        elif comparison.item(i, 0) == 0 and comparison.item(i, 1) == 1:
            FN += 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    roc_x.append(FPR)
    roc_y.append(TPR)
    pr_y.append(PPV)
    knob = X_Y_test.item(m, 0)

plt.plot(roc_x, roc_y)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()

plt.plot(roc_y, pr_y, color='red')
plt.xlabel('Recall (TPR)')
plt.ylabel('Precision')
plt.show()