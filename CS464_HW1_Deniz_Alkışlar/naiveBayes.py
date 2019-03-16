# DENIZ ALKISLAR / 21502930

import csv
import math

# constants for the data
VOCAB_SIZE = 26507
TRAIN_SIZE = 1600
EACH_TRAIN_SIZE = TRAIN_SIZE / 2
TEST_SIZE = 400
MIN_INIT = 0.000000001

def calculate_accuracy(test_features, space_mail_probs, med_mail_probs, test_labels, valid_indices):
    # init probability array
    space_prob = [math.log(0.5)] * len(test_features)
    med_prob = [math.log(0.5)] * len(test_features)

    # testing by using the formula and test-features
    for j in range(len(test_features)):
        for i in range(len(space_mail_probs)):
            if valid_indices[i] == 1:
                if float(test_features[j][i]) != 0:
                    if space_mail_probs[i] == 0:
                        space_prob[j] += float("-inf")
                    else:
                        space_prob[j] += float(test_features[j][i]) * math.log(space_mail_probs[i])

                    if med_mail_probs[i] == 0:
                        med_prob[j] += float("-inf")
                    else:
                        med_prob[j] += float(test_features[j][i]) * math.log(med_mail_probs[i])

    # creating the results
    results = [""] * len(test_features)
    for j in range(len(test_features)):
        if space_prob[j] > med_prob[j]:
            results[j] = "1"
        elif space_prob[j] <= med_prob[j]:
            results[j] = "0"

    # calculating the accuracy
    wrong_count = 0
    for j in range(len(test_features)):
        if results[j] != test_labels[j][0]:
            wrong_count += 1

    return ((len(test_features) - wrong_count) / len(test_features))

# extracting train features and labels
train_features = list(csv.reader(open('train-features.csv')))
train_labels = list(csv.reader(open('train-labels.csv')))

# extracting test features and labels
test_features = list(csv.reader(open('test-features.csv')))
test_labels = list(csv.reader(open('test-labels.csv')))

# space = 1, med = 0
space_feature_mails = list()
med_feature_mails = list()

# separating mail types
for i in range(TRAIN_SIZE):
    if train_labels[i][0] == '1':
        space_feature_mails.append(train_features[i])
    elif train_labels[i][0] == '0':
        med_feature_mails.append(train_features[i])

# init prob and occurrence arrays
space_mail_probs = [MIN_INIT] * VOCAB_SIZE
med_mail_probs = [MIN_INIT] * VOCAB_SIZE
space_mail_occurrence = [MIN_INIT] * VOCAB_SIZE
med_mail_occurrence = [MIN_INIT] * VOCAB_SIZE

# accumulating the frequencies
for i in range(int(EACH_TRAIN_SIZE)):
    for j in range(VOCAB_SIZE):
        space_mail_probs[j] += int(space_feature_mails[i][j])
        med_mail_probs[j] += int(med_feature_mails[i][j])

        # summing the occurrences
        if int(space_feature_mails[i][j]) >= 1:
            space_mail_occurrence[j] += 1
        if int(med_feature_mails[i][j]) >= 1:
            med_mail_occurrence[j] += 1

# finding the probabilities to plug into the formula
space_mail_probs_sum = sum(space_mail_probs)
med_mail_probs_sum = sum(med_mail_probs)
for i in range(VOCAB_SIZE):
    space_mail_probs[i] = (1 + space_mail_probs[i]) / (space_mail_probs_sum + VOCAB_SIZE)
    med_mail_probs[i] = (1 + med_mail_probs[i]) / (med_mail_probs_sum + VOCAB_SIZE)

# prints MAP accuracy
valid_indices = [1] * VOCAB_SIZE
print(calculate_accuracy(test_features, space_mail_probs, med_mail_probs, test_labels, valid_indices))

# calculating the scores for space mails
space_scores = list()
for i in range(VOCAB_SIZE):
    n00 = EACH_TRAIN_SIZE - med_mail_occurrence[i]
    n01 = EACH_TRAIN_SIZE - space_mail_occurrence[i]
    n10 = med_mail_occurrence[i]
    n11 = space_mail_occurrence[i]
    n = n00 + n01 + n10 + n11

    score1 = (n11 / n) * math.log2((n * n11) / ((n11 + n10) * (n11 + n01)))
    score2 = (n01 / n) * math.log2((n * n01) / ((n01 + n00) * (n11 + n01)))
    score3 = (n10 / n) * math.log2((n * n10) / ((n11 + n10) * (n10 + n00)))
    score4 = (n11 / n) * math.log2((n * n00) / ((n01 + n00) * (n10 + n00)))

    score = score1 + score2 + score3 + score4
    space_scores.append(score)

# calculating the scores for med mails
med_scores = list()
for i in range(VOCAB_SIZE):
    n00 = EACH_TRAIN_SIZE - space_mail_occurrence[i]
    n01 = EACH_TRAIN_SIZE - med_mail_occurrence[i]
    n10 = space_mail_occurrence[i]
    n11 = med_mail_occurrence[i]
    n = n00 + n01 + n10 + n11

    score1 = (n11 / n) * math.log2((n * n11) / ((n11 + n10) * (n11 + n01)))
    score2 = (n01 / n) * math.log2((n * n01) / ((n01 + n00) * (n11 + n01)))
    score3 = (n10 / n) * math.log2((n * n10) / ((n11 + n10) * (n10 + n00)))
    score4 = (n11 / n) * math.log2((n * n00) / ((n01 + n00) * (n10 + n00)))

    score = score1 + score2 + score3 + score4
    med_scores.append(score)

# calculating the final scores and sorting them
scores = list()
for i in range(VOCAB_SIZE):
    scores.append((i, ((space_scores[i] + med_scores[i]) / 2)))
scores.sort(key = lambda tup: tup[1], reverse = True)

# printing top 10 features
for p in range(10):
    print(scores[p])

# feature selection
valid_index = [1] * VOCAB_SIZE
step_size = 1
i = VOCAB_SIZE - step_size
j = VOCAB_SIZE
while i > 0:
    for k in range(i, j):
        valid_index[scores[k][0]] = 0
    print(calculate_accuracy(test_features, space_mail_probs, med_mail_probs, test_labels, valid_index))
    i -= step_size
    j -= step_size



