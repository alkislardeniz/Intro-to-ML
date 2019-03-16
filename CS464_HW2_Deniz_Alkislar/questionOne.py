from numpy import genfromtxt, matmul, hstack, ones, mean, std, power, squeeze, asarray
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt

def find_beta(X, Y):
    beta = inv(matmul(X.T, X))
    beta = matmul(beta, X.T)
    beta = matmul(beta, Y)
    return beta

def find_mse(X, Y, beta):
    mse = Y - matmul(X, beta)
    mse = matmul(mse.T, mse) / len(X)
    return mse

def centralize_matrix(M):
    means = mean(M, axis=0)
    stds = std(M, axis=0)
    for i in range(len(M)):
        for j in range(1, M.shape[1]):
                M.itemset((i, j), (M.item(i, j) - means[j]) / stds[j])

# create matrices and find rank
car_big_csv = genfromtxt('carbig.csv', delimiter=',')
car_big = hstack((ones((len(car_big_csv), 1)), car_big_csv))
car_big_t = car_big[:, :-1].T
mult = matmul(car_big_t, car_big[:, :-1])
print(matrix_rank(mult))

# separate X and y matrices as train and test
car_big_train = car_big[:300][:, :-1]
car_big_test = car_big[300:][:, :-1]
y_train = car_big[:300][:, [-1]]
y_test = car_big[300:][:, [-1]]

# find beta coefficients
beta = find_beta(car_big_train, y_train)
print(beta)

# calculate mean square error for train and test data
mse = find_mse(car_big_train, y_train, beta)
print("Trn mse:", mse)
mse = find_mse(car_big_test, y_test, beta)
print("Tst mse:", mse)

# plot MPG vs. HP
P = 5
MPG = car_big[:, 7]
HP = car_big[:, 3]
plt.plot(HP, MPG, 'ro')
plt.xlabel('Horsepower (HP)')
plt.ylabel('Miles per gallon (MPG)')
plt.show()

# create polynomial HP matrix
HP = car_big[:, [0, 3]]
for i in range(2, P + 1):
    a = power(HP[:, [1]], i)
    HP = hstack((HP, a))

# plot for non-centralized rank values
a, rank = ([], [])
for i in range(P + 1):
    a.append(i)
    X = HP[:, a]
    rank.append(matrix_rank(matmul(X.T, X)))
plt.plot(range(0, P + 1), rank, 'ro', label='Non-centralized')
plt.xlabel('Polynomial degree (P)')
plt.ylabel('Rank')

# centralize the HP matrix
centralize_matrix(HP)

# plot for centralized rank values
a, rank = ([], [])
for i in range(P + 1):
    a.append(i)
    X = HP[:, a]
    rank.append(matrix_rank(matmul(X.T, X)))
plt.plot(range(0, P + 1), rank, 'bs', label='Centralized')
plt.show()

# find mse for centralized HP for each p value
HP_train = HP[:300]
HP_test = HP[300:]
a = []
print("HP ONLY")
plt.plot(car_big[:, 3], car_big[:, 7], 'ro')
for i in range(P + 1):
    a.append(i)
    beta = find_beta(HP_train[:, a], y_train)
    beta_plot = squeeze(asarray(matmul(HP_train[:, a], beta)))

    plt.plot(car_big[:300][:, 3], beta_plot, 'o')
    plt.xlabel('Horsepower (HP)')
    plt.ylabel('Miles per gallon (MPG)')

    mse = find_mse(HP_train[:, a], y_train, beta)
    print("P(trn) =", i, mse)
    mse = find_mse(HP_test[:, a], y_test, beta)
    print("P(tst) =", i, mse)
plt.show()

# create polynomial MY matrix and centralize it
P = 3
MY = car_big[:, [6]]
for i in range(2, P + 1):
    a = power(MY[:, [0]], i)
    MY = hstack((MY, a))
centralize_matrix(MY)

# create matrix with HP and MY
HP_MY = hstack((HP[:, [x for x in range(P + 1)]], MY))

# find mse errors for HP_MY matrix
HP_MY_train = HP_MY[:300]
HP_MY_test = HP_MY[300:]
a = [0]
print("HP + MODEL YEAR")
for i in range(1, P + 1):
    a.append(i)
    a.append(i + P)
    beta = find_beta(HP_MY_train[:, a], y_train)
    mse = find_mse(HP_MY_train[:, a], y_train, beta)
    print("P(trn) =", i, mse)
    mse = find_mse(HP_MY_test[:, a], y_test, beta)
    print("P(tst) =", i, mse)