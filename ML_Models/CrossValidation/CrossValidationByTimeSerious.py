import zope
from sklearn.model_selection import TimeSeriesSplit

from ML_Models.CrossValidation.CrossValidationSplitting_I import CrossValidationSplittingI


@zope.interface.implementer(CrossValidationSplittingI)
class CrossValidationByTimeSerious():

    def __init__(self, num_split):
        self.__num_split_cross_validation = num_split

    def split(self, train_data_x, train_data_y):
        time_split = TimeSeriesSplit(max_train_size=None, n_splits=self.__num_split_cross_validation)
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
        for train_index, test_index in time_split.split(train_data_x):
            x_train_list.append(train_data_x.iloc[train_index, :])
            x_test_list.append(train_data_x.iloc[test_index, :])
            y_train_list.append(train_data_y.iloc[train_index])
            y_test_list.append(train_data_y.iloc[test_index])
        assert (len(x_train_list) == len(x_test_list)) == (len(y_train_list) == len(y_test_list))
        return x_train_list, x_test_list, y_train_list, y_test_list
