import torch

from Definition import DEEP_ANT


class DeepAnTModel(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """

    def __init__(self, lookback_size, dim):
        super(DeepAnTModel, self).__init__()
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
