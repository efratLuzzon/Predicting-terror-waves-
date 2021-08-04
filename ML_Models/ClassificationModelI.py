from zope.interface import Interface


class ClassificationModelI(Interface):
    def get_confusion_matrix_model(predictions):
        """get confusion matrix"""

    def create_model():
        """create new model"""

    def train_and_validate_model(stack=None):
        """train model and validted it"""

    def update_params_model(params):
        """update the params model"""

    def tuning_model():
        """tuning hyperparams model (like GridSearchCV or BayesSearchCV"""

    def cross_validation():
        """cross validation like by time serious and etc."""

    def get_features_importance():
        """get feature importance"""

    def get_prediction_model(predictions):
        """get model predictions"""

    def get_hyperparams_model():
        """get hyperparamters"""

    def get_accuracy_result(accuracy_mean, accuracy):
        """get train and test results"""

    def get_trained_model(data, test_year):
        """create model, find hyper-params and train"""
