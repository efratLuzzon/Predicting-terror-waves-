import zope
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
import numpy as np
from ML_Models.Tuning.TuningHyperparams import TuningHyperparams
from skopt.space import Real, Integer
import xgboost as xgb
import pandas as pd


@zope.interface.implementer(TuningHyperparams)
class XgbBayesSearchCV():

    @staticmethod
    def tune(model):
        xgb_params = {
            'learning_rate': Real(0.000001, 0.1, prior='log-uniform'),
            'min_child_weight': Integer(1, 7),
            'max_depth': Integer(1, 13),
            'max_delta_step': Integer(1, 8),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.3, 0.9, prior='uniform'),
            'colsample_bylevel': Real(0.1, 0.8, prior='uniform'),
            'reg_lambda': Real(10, 500, prior='log-uniform'),
            'reg_alpha': Real(1.0e-9, 5.0, prior='log-uniform'),
            'gamma': Real(1.0e-9, 5, prior='log-uniform'),
            'n_estimators': Integer(100, 2000),
            'scale_pos_weight': Real(0.5, 5, prior='log-uniform')
        }

        model_xgb = xgb.XGBClassifier(objective=model.__objective, eval_metric=model.__metric, n_jobs=-1,
                                      max_depth=model.__max_depth, min_child_weight=model.__min_child_weight,
                                      colsample_bytree=model.__col_sample,
                                      subsample=model.__sub_sample, learning_rate=model.__eta, gamma=model.__gamma,
                                      reg_alpha=model.__reg_alpha, n_estimators=model.__n_estimators,
                                      max_delta_step=model.__max_delta_step,
                                      colsample_bylevel=model.__colsample_bylevel,
                                      reg_lambda=model.__reg_lambda, scale_pos_weight=model.__scale_pos_weight)
        train_x, train_y = model.__train_data_model.iloc[:, :-1], model.__train_data_model.iloc[:, -1]
        split_time_series_cv = TimeSeriesSplit(max_train_size=None, n_splits=model.__num_split_cross_validation)
        bayes_cv_tuner = BayesSearchCV(n_iter=7, estimator=model_xgb, n_jobs=-1,
                                       search_spaces=xgb_params,
                                       scoring=make_scorer(model.__score_function),
                                       cv=split_time_series_cv
                                       )

        def status_print(optim_result):
            """Status callback durring bayesian hyperparameter search"""
            # Get all the models tested so far in DataFrame format
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)
            print('ML_Models #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
                len(all_models),
                np.round(bayes_cv_tuner.best_score_, 4),
                bayes_cv_tuner.best_params_
            ))

        result = bayes_cv_tuner.fit(train_x, train_y, callback=status_print)
        model.update_params_model(dict(result.best_params_))
        print(dict(result.best_params_))
        return model.get_hyperparams()
