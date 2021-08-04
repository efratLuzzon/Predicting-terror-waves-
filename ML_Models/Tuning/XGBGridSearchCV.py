import zope
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from Definition import XGB
from ML_Models.Tuning.TuningHyperparams import TuningHyperparams


@zope.interface.implementer(TuningHyperparams)
class XGBGridSearchCV():
    @staticmethod
    def tune(model):
        params = [
            {'max_depth': range(3, 13), 'min_child_weight': range(1, 6)},
            {'subsample': [0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             'gamma': [0, 1, 5]},
            {'reg_alpha': [0, 0.1, 1], 'reg_lambda': [5, 30, 100, 200, 400]},
            {'learning_rate': [0.01, 0.001, 0.0001, 0.000000001], 'n_estimators': [250, 500, 750, 1000, 2000]},
            {'scale_pos_weight': range(0, 5), 'colsample_bylevel': [0.1, 0.2, 0.5, 0.8]},
            {'max_delta_step': range(0, 5)}
        ]
        default_value = {
            XGB.MIN_CHILD_WEIGHT.value: 4,
            XGB.MAX_DEPTH.value: 3,
            XGB.SUBSAMPLE.value: 0.9,
            XGB.COLSAMPLE.value: 0.7,
            XGB.ETA.value: 0.01,
            XGB.GAMMA.value: 5,
            XGB.REG_ALPHA.value: 0.1,
            XGB.N_ESTIMATORS.value: 500,
            XGB.MAX_DELTA_STEP.value: 0,
            XGB.COL_SAMPLE_BY_LEVEL.value: 0.2,
            XGB.REG_LAMBDA.value: 5,
            XGB.SCALE_POS_WEIGHT.value: 2
        }

        model.update_params_model(default_value)
        for param in params:
            train_x, train_y = model.__train_data_model.iloc[:, :-1], model.__train_data_model.iloc[:, -1]
            split_time_series = TimeSeriesSplit(max_train_size=None, n_splits=model.__num_split_cross_validation).split(
                train_x)
            hyper_params = model.get_hyperparams()
            gsearch1 = GridSearchCV(
                estimator=xgb.XGBClassifier(nthread=-1, objective=model.__objective, **hyper_params),
                param_grid=param, cv=split_time_series, scoring=make_scorer(model.__score_function), n_jobs=-1)
            gsearch1.fit(train_x, train_y)
            print(gsearch1.best_params_, gsearch1.best_score_)
            model.update_params_model(gsearch1.best_params_)
        return model.get_hyperparams()