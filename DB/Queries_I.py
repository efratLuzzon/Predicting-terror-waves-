from zope.interface import Interface


class QueriesI(Interface):

    def login(username, password):
        """login user to web"""

    def get_model_data():
        """get data to train the model"""

    def get_model_data_per_date(date):
        """get model data per date"""

    def get_model_accuracy(year):
        """get model and test accuracy"""

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

    def get_weather_between_dates(self, start_date, end_date):
        """get weather data between two dates"""

    def get_attacks_between_dates(self, start_date, end_date):
        """get attacks data between two dates"""

    def get_google_trends_israel_between_dates(self, start_date, end_date):
        """get google trends israel data between two dates"""

    def get_google_trends_palestine_between_dates(self, start_date, end_date):
        """get google trends palestine data between two dates"""

    def get_elections_date_between_dates(self, start_date, end_date):
        """get elections date between two dates"""

    def get_holidays_between_dates(self, start_date, end_date):
        """get holidays data between two dates"""

    def get_attacks_info_by_date(self, date):
        """get attacks information by date"""

    def get_model_predictions(self):
        """get model predictions"""

    def get_real_result_waves(self):
        """get real classification for each day"""
