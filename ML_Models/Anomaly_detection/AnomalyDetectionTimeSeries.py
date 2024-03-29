from datetime import timedelta

import pandas as pd
import torch as torch
from pandas import np
from sklearn.preprocessing import MinMaxScaler

from DataFrameCalender import DataFrameCalender
from ML_Models.Anomaly_detection.DeepAnT import DeepAnTModel
from Definition import DEEP_ANT


class AnomalyDetectionTimeSeries():

    def __init__(self, data):
        self.__data = data
        self.__X = None
        self.__Y = None
        self.__T = None

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data_df):
        self.__data = data_df

    @property
    def X(self):
        return self.__X

    @X.setter
    def X(self, x_var):
        self.__X = x_var

    @property
    def Y(self):
        return self.__Y

    @Y.setter
    def Y(self, y_var):
        self.__Y = y_var

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, t_var):
        self.__T = t_var

    def read_modulate_data(self):
        """
            Data ingestion : Function to read and formulate the data,
            making timestamp as index of data
        """
        self.__data.fillna(self.__data.mean(), inplace=True)
        df = self.__data.copy()
        df = df.reset_index(level=0, drop=True).reset_index()
        df.drop('index', axis='columns', inplace=True)
        self.__data.drop('date', axis='columns', inplace=True)

    def data_pre_processing(self):
        """ Data pre-processing : Function to create data for ML_Models """
        try:
            scaled_data = MinMaxScaler(feature_range=(0, 1))
            data_scaled_ = scaled_data.fit_transform(self.__data)
            self.__data.loc[:, :] = data_scaled_
            _data_ = self.__data.to_numpy(copy=True)
            x = np.zeros(
                shape=(self.__data.shape[0] - DEEP_ANT.LOOKBACK_SIZE, DEEP_ANT.LOOKBACK_SIZE, self.__data.shape[1]))
            y = np.zeros(shape=(self.__data.shape[0] - DEEP_ANT.LOOKBACK_SIZE, self.__data.shape[1]))
            timesteps = []
            for i in range(DEEP_ANT.LOOKBACK_SIZE - 1, self.__data.shape[0] - 1):
                timesteps.append(self.__data.index[i])
                y[i - DEEP_ANT.LOOKBACK_SIZE + 1] = _data_[i + 1]
                for j in range(i - DEEP_ANT.LOOKBACK_SIZE + 1, i + 1):
                    x[i - DEEP_ANT.LOOKBACK_SIZE + 1][DEEP_ANT.LOOKBACK_SIZE - 1 - i + j] = _data_[j]
            self.__X = x
            self.__Y = y
            self.__T = timesteps
        except Exception as e:
            print("Error while performing data pre-processing : {0}".format(e))

    @staticmethod
    def make_train_step(model, loss_fn, optimizer):
        """Computation : Function to make batch size data iterator"""

        def train_step(x, y):
            model.train()
            yhat = model(x)
            loss = loss_fn(y, yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()

        return train_step

    def compute(self):
        """Computation : Find Anomaly using model based computation"""
        model = DeepAnTModel(DEEP_ANT.LOOKBACK_SIZE, DEEP_ANT.FEATURE_NUM)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
        train_data = torch.utils.data.TensorDataset(torch.tensor(self.__X.astype(np.float32)),
                                                    torch.tensor(self.__Y.astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
        train_step = AnomalyDetectionTimeSeries.make_train_step(model, criterion, optimizer)
        for epoch in range(DEEP_ANT.EPOCH):
            loss_sum = 0.0
            ctr = 0
            for x_batch, y_batch in train_loader:
                loss_train = train_step(x_batch, y_batch)
                loss_sum += loss_train
                ctr += 1
            print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum / ctr), epoch + 1))
        hypothesis = model(torch.tensor(self.__X.astype(np.float32))).detach().numpy()
        loss = np.linalg.norm(hypothesis - self.__Y, axis=1)
        return loss.reshape(len(loss), 1)

    def get_score(self):

        self.read_modulate_data()
        self.data_pre_processing()
        loss = self.compute()

        loss_df = pd.DataFrame(self.__T, columns=["date"])
        loss_df.index = self.__T
        loss_df.index = pd.to_datetime(loss_df.index)
        loss_df["loss"] = loss
        loss_df["date"] = pd.to_datetime(loss_df["date"])
        return loss_df

    def update(self, loss_df, start_wave, end_wave):
        delta = timedelta(days=1)
        while start_wave <= end_wave:
            loss_df.at[start_wave, "class"] = 1
            start_wave += delta
        return loss_df

    def find_waves_date(self, loss_df):
        loss_df["class"] = 0
        start_waves = []
        end_waves = []
        DataFrameCalender.set_date_time_index(loss_df, "date", loss_df["date"])
        in_wave = False
        start_counr_end = False
        counter = 0
        start_wave = None
        end_wave = None
        for day in loss_df.iterrows():
            loss = day[1][1]
            timestamp = day[1][0]
            if (loss > 0.1):
                # start wave if didnt start
                if not in_wave:
                    in_wave = True
                    start_wave = timestamp

                else:
                    # stop counting 30 days of decrease if in wave
                    if start_counr_end:
                        counter = 0
                        start_counr_end = False
                        end_wave = None
            else:
                if in_wave:
                    # start count 30 days if didnt start
                    if not start_counr_end:
                        end_wave = timestamp
                        start_counr_end = True
                    counter += 1
                    # if 30 days of decrease passed end wave
                    if counter >= 30:
                        if start_wave < end_wave - timedelta(days=21):
                            loss_df = self.update(loss_df, start_wave, end_wave)
                            print(str(start_wave.date()) + " - " + str(end_wave.date()))
                            start_waves.append(start_wave.date())
                            end_waves.append(end_wave.date())
                        in_wave = False
                        end_wave = None
                        counter = 0
                        start_counr_end = False
        days_df = pd.DataFrame(data=start_waves, columns=["start_wave"])
        days_df["end_wave"] = end_waves
        return days_df, loss_df
