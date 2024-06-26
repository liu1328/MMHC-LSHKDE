import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from MMHC-LSHKDE import MMHC
from LSH_KDE import FastLaplacianKDE
data = pd.read_csv('C:\\Users\\kkk\\Downloads\\dataset\\iris.csv')
data = np.array(data)
x = data[:, :-1]
y = data[:, -1]

# np.random.seed(33)

kf = KFold(n_splits=10)

accuracy_scores = []

# Perform a ten-fold cross-validation
for train_index, test_index in kf.split(x):
    print(train_index,test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_train_reshaped = y_train.reshape(-1, 1)
    data1 = np.concatenate((X_train, y_train_reshaped), axis=1)
    DAG, dag = MMHC(data1)

    p_1 = []
    # Calculate conditional probabilities
    for class_label in np.unique(y_train):
        X_class = X_train[y_train == class_label]
        pc = len(X_class) / len(y_train)
        px = []
        for i in dag:
            for j in dag[i].values():
                if len(j) == 0:
                    # conditional probability P（x|C)
                    x0 = []
                    x_train = np.array([X_class[:, int(i)]]).T
                    x_test = np.array([X_test[:, int(i)]]).T
                    hash = FastLaplacianKDE(x_train, bandwidth=(4 / (3 * len(x_train))) ** (1 / 5),
                                            L=int(len(x_train) * 0.5))
                    for i in range(len(x_test)):
                        x_kde = hash.kde(x_test[i, :], bandwidth=(4 / (3 * len(x_train))) ** (1 / 5))
                        x0.append(x_kde)
                    px.append(x0)

                if len(j) == 1:
                    # conditional probability P（x2|x0，C）
                    x02_1 = []
                    xy_train = np.array([X_class[:, int(i)], X_class[:, int(j[0])]]).T
                    xy_test = np.array([X_test[:, int(i)], X_test[:, int(j[0])]]).T
                    hash = FastLaplacianKDE(xy_train, bandwidth=(4 / (4 * len(xy_train))) ** (1 / 6),
                                            L=int(len(xy_train) * 0.5))
                    xy_kde1 = []
                    for i in range(len(xy_test)):
                        xy_kde = hash.kde(xy_test[i, :], bandwidth=(4 / (4 * len(xy_train))) ** (1 / 6))
                        x02_1.append(xy_kde)
                    x0_1 = []
                    x1_train = np.array([X_class[:, int(j[0])]]).T
                    x1_test = np.array([X_test[:, int(j[0])]]).T
                    hash = FastLaplacianKDE(x1_train, bandwidth=(4 / (3 * len(x1_train))) ** (1 / 5),
                                            L=int(len(x1_train) * 0.5))
                    for i in range(len(x1_test)):
                        x_kde = hash.kde(x1_test[i, :], bandwidth=(4 / (3 * len(x1_train))) ** (1 / 5))
                        x0_1.append(x_kde)
                    x02c = np.divide(x02_1, x0_1)
                    px.append(list(x02c))

        p = np.prod(px, axis=0)
        p_c = np.multiply(p, pc)
        p_1.append(list(p_c))

    y_test1 = []
    for y1 in y_test:
        if y1 == "Iris-setosa":
            a = 0
            y_test1.append(a)
        if y1 == "Iris-versicolor":
            a = 1
            y_test1.append(a)
        if y1 == "Iris-virginica":
            a = 2
            y_test1.append(a)

    b = []
    for i in range(len(p_1[0])):
        max = 0
        for j in range(len(p_1)):
            if p_1[j][i] > max:
                max = p_1[j][i]
                a = j
        b.append(a)

    correct = 0
    incorrect = 0
    for i in range(len(y_test1)):
        if y_test1[i] == b[i]:
            correct += 1
        else:
            incorrect += 1
    currentFoldAccuracy = correct / (correct + incorrect)
    accuracy_scores.append(currentFoldAccuracy)

average_accuracy = np.mean(accuracy_scores)
print(average_accuracy)
