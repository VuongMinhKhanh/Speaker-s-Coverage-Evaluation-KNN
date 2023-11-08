import numpy as np
import pandas as pd
import operator
from collections import Counter
import re
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def find_space_keyword(content, keywords):
    # Sample text for demonstration
    #text = "Python is a great language. It is used for various purposes. Many developers prefer Python due to its simplicity and versatility. I love using Python for data analysis."

    # Keyword to search for
    kw = "không gian"
    freq = {}
    # Split the text into sentences using regex
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
    for keyword in keywords:
        # Count the number of sentences containing the keyword
        count = sum(1 for sentence in sentences if keyword in sentence and kw in sentence)
        freq.update({keyword: count})
    return (max(freq))

def output_excel(data, path):
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


def remove_str(arr, keywords):
    for i in range (0, len(arr)):
        for kw in keywords:
            arr[i] = arr[i].replace(kw, "").strip()
        if arr[i].find("-") > 0:
            arr[i] = cal_means_power(arr[i])
        arr[i] = float(arr[i].replace(",", ".")) # converts type of a value
    return arr.astype(np.float) # converts type of a np

def cal_means_power(power):
    arr = []
    arr = power.split("-")
    arr[0] = float(arr[0])
    arr[1] = float(arr[1])
    return str(round((arr[0] + arr[1])/2))


def testKNN():
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
         np_array[:,0:4], np_array[:,4], test_size=50)

        #print ("Training size: ", len(y_train))
        #print ("Test size    : ", len(y_test))
    
        clf = neighbors.KNeighborsClassifier(n_neighbors = i*10+4, p = 2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        #print ("Print results for 20 test data points:")
        #for i in range(0, 40):
         #   print(y_pred[i], " - ", y_test[i])
    
        print ("Accuracy of NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


def user_input():
    length = (input("Nhập chiều dài: "))
    wid = (input("Nhập chiều rộng: "))
    hei = (input("Nhập chiều cao: "))
    power = (input("Nhập công suất: "))
    n = (input("Nhập số n: "))
    p = (input("Nhập số p: "))


    return length, wid, hei, pow, n, p


def get_user_input(msg):
    return input(msg)


# Driver code
if __name__ == '__main__':
    name = "speaker_dataset"
    data_frame = pd.read_excel(f'{name}.xlsx')
    np_array = data_frame.to_numpy()
    training = np_array[:,0:4]
    coverage = np_array[:,4]
    """_n = 10; _p = 2"""

    
    length = float(input("Nhập chiều dài: "))
    width = float(input("Nhập chiều rộng: "))
    height = float(input("Nhập chiều cao: "))
    power = float(input("Nhập công suất: "))
    _n = int(input("Nhập số n: "))
    _p = int(input("Nhập số p: "))
    

    #test = np.array([[108.6, 10.86, 31.4, 225]])
    test = np.array([[length, width, height, power]])
    #assert test.shape[0] == coverage.shape[0]
    #print(test.shape, coverage.shape)
    clf = neighbors.KNeighborsClassifier(n_neighbors = _n, p = _p)
    clf.fit(training, coverage)
    predict = clf.predict(test)

    print("Không gian sử dụng của sản phẩm là:", predict[0])
