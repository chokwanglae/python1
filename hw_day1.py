import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF

def zscore_analsys():
# 1. z-score를 활용한, 데이터 산점도 분석
    # 독립변수와 종속변수 정의
    X=np.array(df.drop(columns='BMI'))
    Y=np.array(df.BMI)

    # plt.scatter(X[:,1],Y[:,],marker='o', alpha=0.7,s=10,c='red')        # Density
    # plt.scatter(X[:,2],Y[:,]+30,marker='o', alpha=0.7,s=10,c='green')   # Weight
    # plt.scatter(X[:,4],Y[:,]+60,marker='o', alpha=0.7,s=10,c='blue')    # Chest
    # plt.ylabel("BMI")
    # plt.show()

    # 정규화 처리 전.
    fig, subplots = plt.subplots(1, 1)
    subplots.scatter(X[:,0],Y[:,], label='Density', alpha=0.5,s=10)
    subplots.scatter(X[:,1],Y[:,]+30, label='Weight', alpha=0.5,s=10)
    subplots.scatter(X[:,2],Y[:,]+60, label='Chest', alpha=0.5,s=10)
    plt.legend(loc='best')
    plt.ylabel('BMI')
    plt.show()

    # 독립변수들의 z-score 처리: (독립변수-평균) / 표준편차
    X = preprocessing.scale(X)

    # plt.scatter(X[:,0],Y[:,],marker='o', alpha=0.7,s=10,c='red')      # Density
    # plt.scatter(X[:,1],Y[:,]+30,marker='o', alpha=0.7,s=10,c='green') # Weight
    # plt.scatter(X[:,2],Y[:,]+60,marker='o', alpha=0.7,s=10,c='blue')  # Chest
    # plt.ylabel("BMI")
    # plt.show()

    # 정규화 처리 후
    fig, subplots = plt.subplots(1, 1)
    subplots.scatter(X[:,0],Y[:,], label='Density',alpha=0.5,s=10)
    subplots.scatter(X[:,1],Y[:,]+30, label='Weight', alpha=0.5,s=10)
    subplots.scatter(X[:,2],Y[:,]+60, label='Chest', alpha=0.5,s=10)
    plt.legend(loc='best')
    plt.ylabel('BMI')
    plt.show()

#2. PCA의 차원 축소
def showText(X):
    plt.imshow(X, cmap='gray')
    plt.show()

def reducePCA(X,nPC):
    pca = PCA(n_components=nPC)
    X_pca = pca.fit_transform(X)
    rs = pca.inverse_transform(X_pca)
    return rs

if __name__ == '__main__':

    func = 'pca'
    if func == 'zscore':
        # 258*8 의 데이터 셋.
        # 컬럼: ['Body_Fat', 'Density', 'Weight', 'BMI', 'Fat_Free_Weight', 'Chest','Thigh', 'Forearm']
        df = pd.read_csv('..\data\data_male_physical_measurements.csv', header='infer', encoding='latin1')
        zscore_analsys()
    elif func=='pca':
        # shape: 23*23, col: x1~x23
        df = pd.read_csv('..\data\data_number_nine.csv', header='infer', encoding='latin1')
        X = np.array(1 - df)
        for nPC in [23, 10, 5, 3]:
            showText(reducePCA(X, nPC))