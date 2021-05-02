import pandas as pd
import torch as torch
from pandas import DataFrame, np
from sklearn.preprocessing import MinMaxScaler
from Definition import DEEP_ANT, GTD
from TerrorData import TerrorAttackData


def read_modulate_data(data):
    """
        Data ingestion : Function to read and formulate the data,
        making timestamp as index of data
    """
    data.fillna(data.mean(), inplace=True)
    df = data.copy()
    df = df.reset_index(level=0, drop=True).reset_index()
    df.drop('index', axis='columns', inplace=True)
    data.drop('date', axis='columns', inplace=True)
    return data, df


def data_pre_processing(df):
    """
        Data pre-processing : Function to create data for Model
    """
    try:
        scaled_data = MinMaxScaler(feature_range=(0, 1))
        data_scaled_ = scaled_data.fit_transform(df)
        df.loc[:, :] = data_scaled_
        _data_ = df.to_numpy(copy=True)
        X = np.zeros(shape=(df.shape[0] - DEEP_ANT.LOOKBACK_SIZE, DEEP_ANT.LOOKBACK_SIZE, df.shape[1]))
        Y = np.zeros(shape=(df.shape[0] - DEEP_ANT.LOOKBACK_SIZE, df.shape[1]))
        timesteps = []
        for i in range(DEEP_ANT.LOOKBACK_SIZE - 1, df.shape[0] - 1):
            timesteps.append(df.index[i])
            Y[i - DEEP_ANT.LOOKBACK_SIZE + 1] = _data_[i + 1]
            for j in range(i - DEEP_ANT.LOOKBACK_SIZE + 1, i + 1):
                X[i - DEEP_ANT.LOOKBACK_SIZE + 1][DEEP_ANT.LOOKBACK_SIZE - 1 - i + j] = _data_[j]
        return X, Y, timesteps
    except Exception as e:
        print("Error while performing data pre-processing : {0}".format(e))
        return None, None, None


class DeepAnT(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """

    def __init__(self, lookback_size, dim):
        super(DeepAnT, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=lookback_size, out_channels=16,
                                              kernel_size=DEEP_ANT.KERNEL_CONV)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=DEEP_ANT.KERNEL_POOL)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=DEEP_ANT.KERNEL_CONV)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=DEEP_ANT.KERNEL_POOL)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(16, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, dim)

    def forward(self, x):
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.dense_2_layer(x)


def make_train_step(model, loss_fn, optimizer):
    """
        Computation : Function to make batch size data iterator
    """

    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


def compute(X, Y):
    """
        Computation : Find Anomaly using model based computation
    """
    model = DeepAnT(DEEP_ANT.LOOKBACK_SIZE, DEEP_ANT.FEATURE_NUM)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-5)
    train_data = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)),
                                                torch.tensor(Y.astype(np.float32)))
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    train_step = make_train_step(model, criterion, optimizer)
    for epoch in range(DEEP_ANT.EPOCH):
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch in train_loader:
            loss_train = train_step(x_batch, y_batch)
            loss_sum += loss_train
            ctr += 1
        print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum / ctr), epoch + 1))
    hypothesis = model(torch.tensor(X.astype(np.float32))).detach().numpy()
    loss = np.linalg.norm(hypothesis - Y, axis=1)
    return loss.reshape(len(loss), 1)


if __name__ == "__main__":
    terror_data = TerrorAttackData("gtd1970-2019_clean.csv", 1970, 2019, GTD.SELECTED_COUNTRY)
    data_file = terror_data.load_data()
    data, _data = read_modulate_data(data_file)
    X, Y, T = data_pre_processing(data)
    loss = compute(X, Y)

    loss_df = pd.DataFrame(loss, columns=["loss"])
    loss_df.index = T
    loss_df.index = pd.to_datetime(loss_df.index)
    loss_df["timestamp"] = T
    loss_df["timestamp"] = pd.to_datetime(loss_df["timestamp"])

    loss_df.to_csv("df.csv", index=False)
