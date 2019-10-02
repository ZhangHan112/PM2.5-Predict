from __future__ import absolute_import,division,unicode_literals
import os
import csv
import pandas as pd
import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt


# gradient decent
def GD(X, Y, W, eta, Iteration, lambdaL2):
    """
    使用gradient decent learning rate 要調很小，不然很容易爆炸
    """
    listCost = []
    for itera in range(Iteration):
        arrayYHat = X.dot(W)
        arrayLoss = arrayYHat - Y
        arrayCost = (np.sum(arrayLoss**2) / X.shape[0])
        listCost.append(arrayCost)

        arrayGradient = (X.T.dot(arrayLoss) / X.shape[0]) + (lambdaL2 * W)
        W -= eta * arrayGradient
        if itera % 1000 == 0:
            print("iteration:{}, cost:{} ".format(itera, arrayCost))
    return W, listCost

def Adagrad(X, Y, W, eta, Iteration, lambdaL2):
    listCost = []
    arrayGradientSum = np.zeros(X.shape[1])
    for itera in range(Iteration):
        arrayYHat = np.dot(X, W)
        arrayLoss = arrayYHat -Y
        arrayCost = np.sum(arrayLoss**2) / X.shape[0]

        # save cost function value in process
        listCost.append(arrayCost)

        arrayGradient = (np.dot(np.transpose(X), arrayLoss) / X.shape[0]) + (lambdaL2 * W)
        arrayGradientSum += arrayGradient**2
        arraySigma = np.sqrt(arrayGradientSum)
        W -= eta * arrayGradient / arraySigma

        if itera % 1000 == 0:
            print("iteration:{}, cost:{} ".format(itera, arrayCost))
    return W, listCost



def read_train_file():
    # read csv file
    Train_Datapath = "./given/train.csv"
    train_data_file = open(Train_Datapath, "r", encoding="Big5")
    raw_train_data = csv.reader(train_data_file)

    # decalre an empty array (with 18 labels)
    Train_Data = []
    for i in range(18):
        Train_Data.append([])

    # load train data into listTrainData
    n_row = 0
    for r in raw_train_data:
        if n_row != 0:
            for i in range(3, 27):
                if r[i] != "NR":
                    Train_Data[(n_row-1) % 18].append(float(r[i]))
                else:
                    Train_Data[(n_row-1) % 18].append(float(0))   
        n_row += 1    
    train_data_file.close()
    return Train_Data
    
def raw_train_data():
    #initialize trainning data
    Train_Data = []
    for i in range(18):
        Train_Data.append([])
    Train_Data = read_train_file()

    x_train = []
    y_train = []
    # 將資料拆成 x 和 y
    for m in range(12):
        # 一個月每10小時算一筆資料，會有471筆 (csv 資料只有1-20號的資料,ex: 1/1-1/20 , 2/1-2/20 , etc)
        for i in range(471):
            x_train.append([])
            y_train.append(Train_Data[9][480*m + i + 9]) # 10小時之後的結果
            # 18個標籤
            for p in range(18):
            # 收集9小時的資料
                for t in range(9):
                    x_train[471*m + i].append(Train_Data[p][480*m + i + t])
    return x_train , y_train

def read_test_file():
    Test_Datapath = "./given/test.csv"
    test_data_file = open(Test_Datapath, "r", encoding="Big5")
    raw_test_data = csv.reader(test_data_file)

    x_test = []

    n_row = 0
    for r in raw_test_data:
        if n_row % 18 == 0:
            x_test.append([])
            for i in range(2, 11):
                x_test[n_row // 18].append(float(r[i]))
        else:
            for i in range(2, 11):
                if r[i] == "NR":
                    x_test[ n_row//18 ].append(float(0))
                else:
                    x_test[ n_row//18 ].append(float(r[i]))
        n_row += 1

    test_data_file.close()
    
    return x_test

def plot(listCost_gd,listCost_gd_1,listCost_ada):
    ###---Visualization---###
    plt.plot(np.arange(len(listCost_gd[3:])), listCost_gd[3:], "b--", label="GD_0")
    plt.plot(np.arange(len(listCost_gd_1[3:])), listCost_gd_1[3:], "r--", label="GD_100")
    plt.plot(np.arange(len(listCost_ada[3:])), listCost_ada[3:], "g--", label="Adagrad")
    plt.title("Train Process")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function (MSE)")
    plt.legend()
    plt.savefig(os.path.join("./TrainProcess"))
    plt.show()

def predict(arrayPredictY_ada,arrayPredictY_cf,arrayPredictY_gd):
    # compare predict value with different methods
    
    dcitD = {"Adagrad":arrayPredictY_ada, "CloseForm":arrayPredictY_cf, "GD":arrayPredictY_gd}
    pdResult = pd.DataFrame(dcitD)
    pdResult.to_csv("./Predict")
    print(pdResult)

def compare(arrayPredictY_ada,arrayPredictY_cf,arrayPredictY_gd):
    # visualize predict value with different methods
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(np.arange(len(arrayPredictY_ada)), arrayPredictY_ada, "b--")
    plt.title("Adagrad")
    plt.xlabel("Test Data Index")
    plt.ylabel("Predict Result")
    plt.subplot(132)
    plt.plot(np.arange(len(arrayPredictY_cf)), arrayPredictY_cf, "r--")
    plt.title("CloseForm")
    plt.xlabel("Test Data Index")
    plt.ylabel("Predict Result")
    plt.subplot(133)
    plt.plot(np.arange(len(arrayPredictY_gd)), arrayPredictY_gd, "g--")
    plt.title("GD")
    plt.xlabel("Test Data Index")
    plt.ylabel("Predict Result")
    plt.tight_layout()
    plt.savefig(os.path.join("./Compare"))
    plt.show()

def main():
    
    x_train,y_train = raw_train_data()
    x_test = read_test_file()
    # print (x_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)

    # gradient decent
    intLearningRate = 1e-6
    arrayW = np.zeros(x_train.shape[1])  # (163, )
    arrayW_gd, listCost_gd = GD(X=x_train, Y=y_train, W=arrayW, eta=intLearningRate, Iteration=20000, lambdaL2=0)
    arrayW = np.zeros(x_train.shape[1])  # (163, )
    arrayW_gd_1, listCost_gd_1 = GD(X=x_train, Y=y_train, W=arrayW, eta=intLearningRate, Iteration=20000, lambdaL2=100)
    # Adagrad
    intLearningRate = 5
    arrayW = np.zeros(x_train.shape[1])  # (163, )
    arrayW_ada, listCost_ada = Adagrad(X=x_train, Y=y_train, W=arrayW, eta=intLearningRate, Iteration=20000, lambdaL2=0)
    # close form
    arrayW_cf = inv(x_train.T.dot(x_train)).dot(x_train.T.dot(y_train))


    x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1) # (5652, 163)

    # gradient decent
    arrayPredictY_gd = np.dot(x_test, arrayW_gd)
    # Adagrad
    arrayPredictY_ada = np.dot(x_test, arrayW_ada)
    # close form
    arrayPredictY_cf = np.dot(x_test, arrayW_cf)



    plot(listCost_gd,listCost_gd_1,listCost_ada)
    predict(arrayPredictY_ada,arrayPredictY_cf,arrayPredictY_gd)
    compare(arrayPredictY_ada,arrayPredictY_cf,arrayPredictY_gd)
    


if __name__ == "__main__":
    main()








