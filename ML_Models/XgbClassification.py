import calendar
from datetime import date

import zope
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np

from DataFrameCalender import DataFrameCalender
from Definition import XGB
from ML_Models.ClassificationModelI import ClassificationModelI
from ML_Models.CrossValidation.CrossValidationByTimeSerious import CrossValidationByTimeSerious
from ML_Models.Tuning.XGBGridSearchCV import XGBGridSearchCV
import matplotlib.pyplot as plt


@zope.interface.implementer(ClassificationModelI)
class XgbClassification():
    def __init__(self, type, mode, look_back_size, num_split_cross_validation, feature_names, year_test):
        self.__num_split_cross_validation = num_split_cross_validation
        self.__cross_validation = CrossValidationByTimeSerious(num_split_cross_validation)
        self.__tuning_hyperparameters = XGBGridSearchCV()
        self.__train_data_model = None
        self.__test_model = None
        self.__type = type
        self.__feature_names = feature_names
        self.__look_back_size = look_back_size
        self.__num_split_cross_validation = num_split_cross_validation
        self.__metric = None
        self.__score_function = None
        self.__objective = None
        self.__train_data_model = None
        self.__train_model = None
        self.__mode = mode
        self.__max_depth = None
        self.__min_child_weight = None
        self.__gamma = None
        self.__sub_sample = None
        self.__col_sample = None
        self.__eta = None
        self.__reg_alpha = None
        self.__n_estimators = None
        self.__max_delta_step = None
        self.__colsample_bylevel = None
        self.__reg_lambda = None
        self.__scale_pos_weight = None
        self.__confusion_matrix = None
        self.__year_test = year_test
        self.set_mode_xgb_parameters()

    def get_confusion_matrix_model(self, predictions):
        """get confusion matrix"""
        test_y = self.__test_model.iloc[:, -1].values
        confusion_matrix_normalize = pd.crosstab(test_y, predictions, rownames=['original'], colnames=['Predicted'],
                                                 normalize=True)
        confusion_matrix_with_type = pd.DataFrame(columns=["year", "TP", "FP", "FN", "TN"], data=[[self.__year_test, 0, 0, 0, 0]], index=[0])
        dtype0 = {"year" : np.int64, "TP" : np.float64, "FP" : np.float64, "FN" : np.float64, "TN" : np.float64}
        confusion_matrix_with_type = confusion_matrix_with_type.astype(dtype0)
        try:
            confusion_matrix_with_type.at[0, "TP"] = confusion_matrix_normalize.iloc[0, 0]
        except:
            confusion_matrix_with_type.at[0, "TP"] = 0
        try:
            confusion_matrix_with_type.at[0, "FP"] = confusion_matrix_normalize.iloc[0, 1]
        except:
            confusion_matrix_with_type.at[0, "FP"] = 0
        if len(confusion_matrix_normalize) > 1:
            confusion_matrix_with_type.at[0, "FN"] = confusion_matrix_normalize.iloc[1, 0]
            confusion_matrix_with_type.at[0, "TN"] = confusion_matrix_normalize.iloc[1, 1]
        else:
            confusion_matrix_with_type.at[0, "FN"] = 0
            confusion_matrix_with_type.at[0, "TN"] = 0
        return confusion_matrix_with_type

    def create_model(self):
        """create new model"""
        model = xgb.XGBClassifier(nthread=-1, objective=self.__objective, metrics=self.__metric, n_jobs=-1,
                                  early_stopping_rounds=10,
                                  num_boost_round=999, max_depth=self.__max_depth,
                                  min_child_weight=self.__min_child_weight, colsample_bytree=self.__col_sample,
                                  subsample=self.__sub_sample, learning_rate=self.__eta, gamma=self.__gamma,
                                  reg_alpha=self.__reg_alpha, n_estimators=self.__n_estimators,
                                  max_delta_step=self.__max_delta_step,
                                  colsample_bylevel=self.__colsample_bylevel,
                                  reg_lambda=self.__reg_lambda, scale_pos_weight=self.__scale_pos_weight)
        return model

    def train_and_validate_model(self, stack=None):
        """train model and validted it"""
        train_x, train_y = self.__train_data_model.iloc[:, :-1], self.__train_data_model.iloc[:, -1].values
        test_x, test_y = self.__test_model.iloc[:, :-1], self.__test_model.iloc[:, -1].values
        # fit model
        if stack is not None:
            model = stack
            model.fit(train_x, train_y)
            eval_set = [(train_x, train_y), (test_x, test_y)]
            model.fit(train_x, train_y, eval_set=eval_set)
        else:
            model = self.create_model()
            eval_set = [(train_x, train_y), (test_x, test_y)]
            model.fit(train_x, train_y, eval_metric=self.__metric, eval_set=eval_set, early_stopping_rounds=20)
            print("score train {}".format(model.score(train_x, train_y)))

        self.set_train_model(model)
        predictions = model.predict(test_x, ntree_limit=model.best_ntree_limit)
        accuracy = self.__score_function(test_y, predictions)
        confusion_matrix = pd.crosstab(test_y, predictions, rownames=['original'], colnames=['Predicted'])
        confusion_matrix_normalize = pd.crosstab(test_y, predictions, rownames=['original'], colnames=['Predicted'],
                                                 normalize=True)
        self.set_confusion_matrix(matrix=confusion_matrix_normalize)

        return accuracy * 100, model.evals_result(), model.best_ntree_limit, predictions, test_y

    def update_params_model(self, params):
        """update the params model"""
        for param_name, param_value in params.items():
            self.update_param_model(param_name, param_value)

    def tuning_model(self):
        """tuning hyperparams model (like GridSearchCV or BayesSearchCV"""
        hyper_params = self.__tuning_hyperparameters.tune(self)
        self.update_params_model(hyper_params)

    def cross_validation(self):
        """cross validation like by time serious and etc."""
        train_x, train_y = self.__train_data_model.iloc[:, :-1], self.__train_data_model.iloc[:, -1]
        x_train_list, x_test_list, y_train_list, y_test_list = self.__cross_validation.split(train_x, train_y)
        accuracy_mean = 0
        len_split_to_cross = len(x_train_list)
        num_step_from_begin = 3
        predictions_all = []
        for i in range(num_step_from_begin, len_split_to_cross):
            # fit model on history and make a prediction
            print("#### %d ####" % (i))
            model = self.create_model()
            eval_set = [(x_train_list[i].values, y_train_list[i].values),
                        (x_test_list[i].values, y_test_list[i].values)]
            model.fit(x_train_list[i].values, y_train_list[i].values, eval_metric=self.__metric, eval_set=eval_set,
                      early_stopping_rounds=20)
            # self.set_train_model(model)
            # make a one-step prediction
            predictions = model.predict(x_test_list[i].values)
            # calculate model accuracy
            accuracy = self.__score_function(y_test_list[i].values, predictions)  # squared by default
            accuracy_mean += accuracy
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            predictions_all.append(predictions)
        accuracy_mean /= (len_split_to_cross - num_step_from_begin)
        print("Accuracy mean: %.2f%%" % (accuracy_mean * 100.0))
        return accuracy_mean * 100

    def get_features_importance(self):
        """get feature importance"""
        feature_important = self.__train_model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        keys = [k.lower() for k in keys]
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_index(ascending=True)
        prev_feature_index = 0
        for feature in sorted(self.__feature_names.str.lower()):
            for i in range(1, self.__look_back_size + 1):
                feature_day = feature + "(t-%d)" % i
                if feature_day not in keys:
                    line = pd.DataFrame({"score": 0}, index=[feature_day])
                    if prev_feature_index == 0:
                        data = pd.concat([line, data])
                    else:
                        data = pd.concat([data.iloc[:prev_feature_index], line, data.iloc[prev_feature_index:]])
                prev_feature_index += 1
        columns_name = data.index
        data = data.values.reshape(1, -1)
        #data = np.insert(data, self.__year_test, -1)
        data = pd.DataFrame(data=data, index=[self.__year_test for i in range(len(data))], columns=columns_name)
        feature_score = None
        if feature_score is None:
            feature_score = pd.DataFrame(data=data, index=[self.__year_test])
        else:
            feature_score = pd.concat([feature_score, data])
        return feature_score

    def get_prediction_model(self, predictions):
        """get model predictions"""
        prediction_matrix = DataFrameCalender.create_empty_dates(calendar.SUNDAY, self.__year_test, self.__year_test)
        DataFrameCalender.set_date_time_index(prediction_matrix, 'date', prediction_matrix['date'])
        prediction_matrix['prediction'] = predictions
        #prediction_matrix.drop('date', axis='columns', inplace=True)
        return prediction_matrix

    def get_hyperparams_model(self):
        """get hyperparamters"""
        hyper_params = self.get_hyperparams()
        update_hyper_params = {'year' : self.__year_test}
        update_hyper_params.update(hyper_params)
        hyper_params_df = pd.DataFrame(data=update_hyper_params, columns=update_hyper_params.keys(), index=[0])
        return hyper_params_df

    @staticmethod
    def get_trained_model(data, test_year):
        """create model, find hyper-params and train"""
        start_train, end_train, start_test, end_test = XgbClassification.choose_year(test_year)
        xgb_model = XgbClassification(mode=XGB.CLASSIFIER, look_back_size=30, num_split_cross_validation=7,
                                      feature_names=data.columns[:-1], type=XGB.XGB.value, year_test=test_year)
        data_suprived_model = xgb_model.series_to_supervised(data)
        xgb_model.split_model_by_n_last_days(data_suprived_model, (end_train - start_train).days + 1,
                                             (end_test - start_train).days + 1)

        xgb_model.tuning_params_with_GridSearchCV()
        accuracy_mean = xgb_model.cross_validation()
        accuracy, evals_result, get_num_boosting_rounds, predictions, original = xgb_model.train_and_validate_model()
        return accuracy, accuracy_mean, xgb_model, predictions


    def get_accuracy_result(self, accuracy_mean, accuracy):
        """get train and test results"""
        result = pd.DataFrame(index=[0], columns=["year", "model_accuracy", "test_accuracy"], data=[[self.__year_test, accuracy_mean, accuracy]])
        return result

    def set_mode_xgb_parameters(self):
        self.__objective = 'binary:logistic'
        self.__metric = 'logloss'
        self.__score_function = accuracy_score
        self.__min_child_weight = 4
        self.__max_depth = 7
        self.__sub_sample = 0.6591391650743395
        self.__gamma = 0.16029711661454313
        self.__eta = 0.00082586160160568
        self.__col_sample = 0.6924121359262365
        self.__reg_alpha = 0.5
        self.__colsample_bylevel = 0.37895025816685444
        self.__max_delta_step = 7
        self.__n_estimators = 339
        self.__reg_lambda = 30
        self.__scale_pos_weight = 5

    def get_confusion_matrix(self):
        return self.__confusion_matrix

    def set_confusion_matrix(self, matrix):
        self.__confusion_matrix = matrix

    def set_train_model(self, model):
        self.__train_model = model

    def series_to_supervised(self, data_model):
        """
            Frame a time series as a supervised learning dataset.
            Arguments:
                data: Sequence of observations as a list or NumPy array.
                n_in: Number of lag observations as input (X).
                n_out: Number of observations as output (y).
            Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        num_columns = len(data_model.columns)
        columns_names = data_model.columns.values
        assert num_columns == len(columns_names)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.__look_back_size, 0, -1):
            cols.append(data_model.iloc[:, :-1].shift(i))
            names += [('%s(t-%d)' % (columns_names[j], i)) for j in range(num_columns - 1)]
        # forecast sequence (t, t+1, ... t+n)
        cols.append(data_model[columns_names[num_columns - 1]])
        names += [('%s(t)' % (columns_names[num_columns - 1]))]
        # put it all together
        order_data = pd.concat(cols, axis=1)
        order_data.columns = names
        # drop rows with NaN values
        return order_data.iloc[self.__look_back_size:, :]

    def split_model_by_n_last_days(self, data, end_train, end_test):
        self.__train_data_model = data.iloc[:end_train, :]
        self.__test_model = data.iloc[end_train: end_test, :]
        self.reset_target_test_feture()

    def reset_target_test_feture(self):
        replace_column = {}
        for i in range(1, self.__look_back_size + 1):
            replace_column["copy_target(t-%d)" % i] = 1
        self.__test_model.replace(replace_column, 0, inplace=True)

    def update_param_model(self, param_name, param_value):
        if param_name == XGB.MIN_CHILD_WEIGHT.value:
            self.__min_child_weight = param_value
        elif param_name == XGB.MAX_DEPTH.value:
            self.__max_depth = param_value
        elif param_name == XGB.COLSAMPLE.value:
            self.__col_sample = param_value
        elif param_name == XGB.SUBSAMPLE.value:
            self.__sub_sample = param_value
        elif param_name == XGB.ETA.value:
            self.__eta = param_value
        elif param_name == XGB.GAMMA.value:
            self.__gamma = param_value
        elif param_name == XGB.REG_ALPHA.value:
            self.__reg_alpha = param_value
        elif param_name == XGB.N_ESTIMATORS.value:
            self.__n_estimators = param_value
        elif param_name == XGB.MAX_DELTA_STEP.value:
            self.__max_delta_step = param_value
        elif param_name == XGB.COL_SAMPLE_BY_LEVEL.value:
            self.__colsample_bylevel = param_value
        elif param_name == XGB.REG_LAMBDA.value:
            self.__reg_lambda = param_value
        elif param_name == XGB.SCALE_POS_WEIGHT.value:
            self.__scale_pos_weight = param_value

    def plot_validation_loss_graph(self, results, split, accuracy):
        epochs = len(results['validation_0'][self.__metric])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        title = "Accuracy = {}".format(accuracy)
        ax.annotate(title, xy=(0.5, 0.2))
        ax.plot(x_axis, results['validation_0'][self.__metric], label='Train')
        ax.plot(x_axis, results['validation_1'][self.__metric], label='Test')
        ax.legend()
        pyplot.ylabel('LOSS')
        title = 'XGBoost loss - SPLIT {}'.format(split)
        pyplot.title(title)
        pyplot.show()

    def plot_tree_by_index(self, index):
        if self.__train_model:
            xgb.plot_tree(self.__train_model, num_trees=index)
            plt.rcParams['figure.figsize'] = [50, 10]
            plt.show()
        else:
            print("model didn't train yet")

    def plot_feature_importance(self):
        feature_important = self.__train_model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        return data

    def get_hyperparams(self):
        hyper_params = {
            XGB.MIN_CHILD_WEIGHT.value: self.__min_child_weight,
            XGB.MAX_DEPTH.value: self.__max_depth,
            XGB.SUBSAMPLE.value: self.__sub_sample,
            XGB.COLSAMPLE.value: self.__col_sample,
            XGB.ETA.value: self.__eta,
            XGB.GAMMA.value: self.__gamma,
            XGB.REG_ALPHA.value: self.__reg_alpha,
            XGB.N_ESTIMATORS.value: self.__n_estimators,
            XGB.MAX_DELTA_STEP.value: self.__max_delta_step, XGB.COL_SAMPLE_BY_LEVEL.value: self.__colsample_bylevel,
            XGB.REG_LAMBDA.value: self.__reg_lambda,
            XGB.SCALE_POS_WEIGHT.value: self.__scale_pos_weight
        }
        return hyper_params

    @staticmethod
    def choose_year(year):
        start_train = date(1970, 1, 31)
        end_train = date(year - 1, 12, 31)
        start_test = date(year, 1, 1)
        end_test = date(year, 12, 31)
        return start_train, end_train, start_test, end_test


