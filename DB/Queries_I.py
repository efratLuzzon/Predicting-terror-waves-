from zope.interface import Interface


class QueriesI(Interface):

    def login(username, password):
        """login user to web"""

    def get_model_data():
        """get data to train the model"""

    def get_model_data_per_date(date):
        """get model data per date"""

    def get_num_attacks_per_day():
        """get num attacks for each day"""

    def get_anomaly_detection():
        """get anomaly score for each day"""

    def load_data(df, table_name):
        """load datafrmae to DB"""

    def get_confusion_matrix(year):
        """get confusion matrix by year"""

    def get_model_date_prediction(year):
        """get accuracy and predictions by year"""

    def get_hyperparameters(year):
        """get hyperparams for trained model by year"""

    def get_features(year):
        """get features score for trained model by year"""
