from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
import warnings
warnings.filterwarnings("error", category=UserWarning)

# import SGD classifier

# import standard scalar transformer
y_train = []
X_train = []

i = 0
for i, target in enumerate(os.listdir('archive/training/')):
    for pic in os.listdir('archive/training/'+target):
        try:
            X_train.append(
                resize(imread('archive/training/'+target+'/'+pic), (64, 64)))
            y_train.append(i)
            print(f'running {target} in training')
        except:
            pass

X_train = np.array(X_train)
y_train = np.array(y_train)
y_valid = []
X_valid = []
for i, target in enumerate(os.listdir('archive/validation/')):
    for pic in os.listdir('archive/validation/'+target):
        try:
            X_valid.append(
                resize(imread('archive/validation/'+target+'/'+pic), (64, 64)))
            y_valid.append(i)
            print(f'running {target} in valid')

        except:
            pass

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)
y_test = []
X_test = []
for i, target in enumerate(os.listdir('archive/evaluation/')):
    for pic in os.listdir('archive/evaluation/'+target):
        try:
            X_test.append(
                resize(imread('archive/evaluation/'+target+'/'+pic), (64, 64)))
            y_test.append(i)
            print(f'running {target} in eval')
        except:
            pass

X_test = np.array(X_test)
y_test = np.array(y_test)

from GreyifyHogify import RGB2GrayTransformer, HogTransformer

grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(4, 4),
    orientations=8,
    block_norm='L2-Hys',
)

scalify = StandardScaler()


def setup_training(X_train):
    X_train_gray = grayify.fit_transform(X_train)
    # hogify all the grey images in a loop
    X_train_hog = []
    for i in range(len(X_train_gray)):
        X_train_hog.append(hog(X_train_gray[i], pixels_per_cell=(14, 14), cells_per_block=(
            4, 4), orientations=8, block_norm='L2-Hys'))

    # scale the hog features
    X_train_prepared = scalify.fit_transform(X_train_hog)

    return X_train_prepared

def sgd_2000(X_train_prepared):
    print(X_train_prepared.shape)
    print('running sgd 2000')
    sgd_clf = SGDClassifier(random_state=1, max_iter=2000, tol=1e-3, dual=False)
    sgd_clf.fit(X_train_prepared, y_train)
    print('done')
    return sgd_clf

# train using svm
from sklearn.svm import SVC

def svm_1000_rbf(X_train_prepared):
    svm_clf = SVC(C=1000, kernel='rbf', gamma=0.001, random_state=1)
    svm_clf.fit(X_train_prepared, y_train)
    return svm_clf

def setup_testing(X_test):
    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)
    return X_test_prepared

def measure_accuracy(model, test_data, model_name):
    print("Measuring the accuracy for " + model_name)
    y_pred = model.predict(test_data)
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    return 100*np.sum(y_pred == y_test)/len(y_test)

# deep copy the data
import copy as cp
 
# deep copy the data
X_train_copy = cp.deepcopy(X_train)
X_valid_copy = cp.deepcopy(X_valid)
X_test_copy = cp.deepcopy(X_test)

# decreasing loop from 127 to 1
for i in range(127, 58, -3):
    this_train = []
    print(f'running {i} in train')
    for image in X_train_copy:
        this_train.append(resize(image, (i, i)))

    this_test = []
    print(f'running {i} in test')
    for image in X_test_copy:
        this_test.append(resize(image, (i, i)))

    training_data = setup_training(this_train)
    svm_clf = svm_1000_rbf(training_data)
    test_data = setup_training(this_test)
    acc = measure_accuracy(svm_clf, test_data, 'SVM, C=1000, rbf')

    # write the acc number to a file
    with open('svm_1000_rbf.txt', 'a') as f:
        f.write(str(i) + 'x' + str(i) + ', ' + str(acc) + '\n')

    # close the file
    f.close()
    