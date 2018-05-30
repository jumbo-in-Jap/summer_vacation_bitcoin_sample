# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tflearn

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class PredictResult :    
    def __init__(self, accurancy: float, benefit: int):
        self.accurancy = accurancy
        self.benefit   = benefit


class Prediction :

    def __init__(self):
        self.pre_dataset = None
        self.dataset = None
        self.normalize_val = None

        # 算出する値たち
        self.model = None
        self.train_predict = None
        self.test_predict = None

        # データセットのパラメータ設定
        self.steps_of_history = 3
        self.steps_in_future = 1
        self.csv_path = './csv/bitcoin_log.csv'

        # 結果
        self.result = None

    def load_dataset(self):
        # データ準備
        dataframe = pd.read_csv(self.csv_path,
                usecols=['終値'],
                engine='python')
        self.dataset = dataframe.values[::-1]
        self.dataset = self.dataset.astype('float32')

        # 標準化
        self.pre_dataset = dataframe.values[::-1]
        self.normalize_val = np.max(np.abs(self.dataset))
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

        net = tflearn.gru(net, n_units=6)
        net = tflearn.fully_connected(net, 1, activation='linear')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                loss='mean_square')

        # Define model
        self.model = tflearn.DNN(net, tensorboard_verbose=1)

        pos = round(len(X) * (1 - 0.2))
        # 8割を訓練データ
        trainX, trainY = X[:pos], Y[:pos]
        # 2割をテストデータにする
        testX, testY   = X[pos:], Y[pos:]
        
        return trainX, trainY, testX, testY

    def execute_predict(self, trainX, trainY, testX, testY):
        # Start training (apply gradient descent algorithm)
        # 正規化した訓練、テストデータをそれぞれ入れてfittingする
        self.model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=32, n_epoch=150, run_id='btc')

        # predict
        # 訓練データに基づく再予測
        self.train_predict = self.model.predict(trainX) * self.normalize_val
        # テストデータに基づく実予測
        self.test_predict = self.model.predict(testX) * self.normalize_val
        print('Accuracy: {0:.3f}'.format(self.model.evaluate(testX, testY)[0]))

    def plot_graph(self):
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
        plt.plot(self.pre_dataset, label="actual", color="k")
        plt.plot(train_predict_plot, label="train", color="r")
        plt.plot(test_predict_plot, label="test", color="b")
        # 凡例を表示
        plt.legend()
        plt.savefig('result.png')
    
    def calc_result(self):
        # 最後の30日で計算する
        span_date = 30
        # 1日後の価格と比較する
        time_span = 1
        predict_last = self.test_predict[-30:]
        actual_last = self.pre_dataset[-30:]
        total_benefit = 0
        
        for i in range(len(predict_last)):
            predict_price = predict_last[i][0]
            actual_price = actual_last[i][0]

            compare_predict_index = i + time_span
            if compare_predict_index < len(predict_last):
                compare_predict_price = predict_last[compare_predict_index][0]
                is_up = compare_predict_price - predict_price > 0
                if is_up:
                    margin_price = actual_last[compare_predict_index][0] - actual_last[i][0]
                    total_benefit += margin_price
                    print("上がるので買います、翌日に売ると利益は%d円です" % margin_price)

        print("結果として %d円儲かりました" % total_benefit)

    def execute(self):
        trainX, trainY, testX, testY = self.setup()
        self.execute_predict(trainX, trainY, testX, testY)
        self.plot_graph()
        self.calc_result()
        

if __name__ == "__main__":

    prediction = Prediction()
    prediction.execute()


