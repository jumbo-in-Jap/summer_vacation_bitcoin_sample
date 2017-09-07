# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tflearn

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class Prediction :

    def __init__(self):
        self.dataset = None
        # 算出する値たち
        self.model = None
        self.train_predict = None
        self.test_predict = None

        # データセットのパラメータ設定
        self.steps_of_history = 3
        self.steps_in_future = 1
        self.csv_path = './csv/bitcoin_log.csv'

    def load_dataset(self):
        # データ準備
        dataframe = pd.read_csv(self.csv_path,
                usecols=['終値'],
                engine='python').sort_values('終値', ascending=True)
        self.dataset = dataframe.values
        self.dataset = self.dataset.astype('float32')

        # 標準化
        self.dataset -= np.min(np.abs(self.dataset))
        self.dataset /= np.max(np.abs(self.dataset))


    def create_dataset(self):
        X, Y = [], []
        for i in range(0, len(self.dataset) - self.steps_of_history, self.steps_in_future):
            X.append(self.dataset[i:i + self.steps_of_history])
            Y.append(self.dataset[i + self.steps_of_history])

        X = np.reshape(np.array(X), [-1, self.steps_of_history, 1])
        Y = np.reshape(np.array(Y), [-1, 1])
        return X, Y

    def setup(self):
        self.load_dataset()
        X, Y = self.create_dataset()

        # Build neural network
        net = tflearn.input_data(shape=[None, self.steps_of_history, 1])

        # LSTMは時間かかるのでGRU
        # http://dhero.hatenablog.com/entry/2016/12/02/%E6%9C%80%E5%BC%B1SE%E3%81%A7%E3%82%82%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%A7%E3%81%8A%E9%87%91%E3%81%8C%E7%A8%BC%E3%81%8E%E3%81%9F%E3%81%84%E3%80%905%E6%97%A5%E7%9B%AE%E3%83%BBTFLearn%E3%81%A8
        net = tflearn.gru(net, n_units=6)
        net = tflearn.fully_connected(net, 1, activation='linear')

        # 回帰の設定
        # Adam法で測定
        # http://qiita.com/TomokIshii/items/f355d8e87d23ee8e0c7a
        # 時系列分析での予測精度の指標にmean_squareを使っている
        # mapeが一般的なようだ
        # categorical_crossentropy
        # mean_square : 二乗平均平方根
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                loss='mean_square')

        # Define model
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

        # 今回は80%を訓練データセット、20%をテストデータセットとして扱う。
        pos = round(len(X) * (1 - 0.2))
        # trainX = 0.8, trainY = 0.2, testX = 0.8, testY = 0.2
        trainX, trainY = X[:pos], Y[:pos]
        testX, testY   = X[pos:], Y[pos:]
        
        return trainX, trainY, testX, testY

    def executePredict(self, trainX, trainY, testX, testY):
        # Start training (apply gradient descent algorithm)
        # 正規化した訓練、テストデータをそれぞれ入れてfittingする
        self.model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=1, n_epoch=50, run_id='btc')

        # predict
        self.train_predict = self.model.predict(trainX)
        self.test_predict = self.model.predict(testX)

        print('Accuracy: {0:.3f}'.format(self.model.evaluate(testX, testY)[0]))

    def showResult(self):
        # plot train data
        train_predict_plot = np.empty_like(self.dataset)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.steps_of_history:len(self.train_predict) + self.steps_of_history, :] = \
                self.train_predict

        # plot test dat
        test_predict_plot = np.empty_like(self.dataset)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(self.train_predict) + self.steps_of_history:len(self.dataset), :] = \
                self.test_predict

        # plot show res
        plt.figure(figsize=(8, 8))
        plt.title('History={} Future={}'.format(self.steps_of_history, self.steps_in_future))
        plt.plot(self.dataset, label="actual", color="k")
        plt.plot(train_predict_plot, label="train", color="r")
        plt.plot(test_predict_plot, label="test", color="b")
        plt.savefig('result.png')
        plt.show()


if __name__ == "__main__":

    prediction = Prediction()
    trainX, trainY, testX, testY = prediction.setup()
    prediction.executePredict(trainX, trainY, testX, testY)
    prediction.showResult()


