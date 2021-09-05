import zope
from zope.interface import implementer
from DB.MysqlDB import MysqlDB
from DB.Queries_I import QueriesI


@zope.interface.implementer(QueriesI)
class XGBModelQueries():

    def __init__(self):
        self.__db_connection = MysqlDB()

    def login(self, username, password):
        """login user to web"""
        query = "SELECT * FROM accounts where user_id = %s"
        result = self.__db_connection.fetch(query, (username,))
        return result

    def get_model_data(self):
        """get data to train the model"""
        query = "SELECT * FROM chris_holidays " \
                "INNER JOIN jewish_holidays using(date) " \
                "INNER JOIN muslim_holidays using(date) " \
                "INNER JOIN elections using(date) " \
                "INNER JOIN (SELECT date,num_attack,succeed,num_deaths,num_wounded,orginaized," \
                "explosion,armed_assault,hijacking,infrastructure_attack,unarmed_assault From gtd) gtd_tb using(date) " \
                "INNER JOIN (SELECT date,max_temp,perciption From israel_weather) weather_tb using(date)"
        result = self.__db_connection.fetch(query)
        return result

    def get_model_data_per_date(self, date):
        """get model data per date"""
        query = "SELECT * FROM chris_holidays " \
                "INNER JOIN jewish_holidays using(date) " \
                "INNER JOIN muslim_holidays using(date) " \
                "INNER JOIN elections using(date) " \
                "INNER JOIN (SELECT date,num_attack,succeed,num_deaths,num_wounded,orginaized," \
                "explosion,armed_assault,hijacking,infrastructure_attack,unarmed_assault From gtd) gtd_tb using(date) " \
                "INNER JOIN (SELECT date,max_temp,perciption From israel_weather) weather_tb using(date) where date = %s"
        result = self.__db_connection.fetch(query, (date,))
        return result

    def get_model_accuracy(self, year):
        query = "SELECT model_accuracy, test_accuracy FROM accuracy WHERE year=%s"
        result = self.__db_connection.fetch(query, (year,))
        return result

    def get_num_attacks_per_day(self):
        """get num attacks for each day"""
        query = "SELECT date, num_attack FROM gtd"
        result = self.__db_connection.fetch(query)
        return result

    def get_anomaly_detection(self):
        """get anomaly score for each day"""
        result = self.__db_connection.fetch("SELECT * FROM annomaly_detection")
        return result

    def load_data(self, df, table_name):
        """load datafrmae to DB"""
        self.__db_connection.load_df(df, table_name)

    def get_confusion_matrix(self, year):
        """get confusion matrix by year"""
        query = "SELECT * FROM confusion_matrix WHERE year=%s"
        result = self.__db_connection.fetch(query, (year,))
        return result

    def get_model_date_prediction(self, year):
        """get accuracy and predictions by year"""
        result = self.__db_connection.fetch("SELECT * FROM accuracy INNER JOIN prediction using(date)")
        return result

    def get_hyperparameters(self, year):
        """get hyperparams for trained model by year"""
        query = "SELECT * FROM hyperparameters WHERE year=%s"
        result = self.__db_connection.fetch(query, (year,))
        return result

    def get_features(self, year):
        """get features score for trained model by year"""
        query = "SELECT * FROM features WHERE year=%s"
        result = self.__db_connection.fetch(query, (year,))
        return result

    def get_weather_between_dates(self, start_date, end_date):
        """get weather data between two dates"""
        query = "SELECT * FROM israel_weather WHERE date between %s and %s"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_attacks_between_dates(self, start_date, end_date):
        """get attacks data between two dates"""
        query = "SELECT date, num_attack, num_wounded, num_deaths FROM gtd WHERE date between %s and %s"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_google_trends_israel_between_dates(self, start_date, end_date):
        """get google trends israel data between two dates"""
        query = "SELECT * FROM google_trends_israel WHERE date between %s and %s"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_google_trends_palestine_between_dates(self, start_date, end_date):
        """get google trends palestine data between two dates"""
        query = "SELECT * FROM google_trends_palestine WHERE date between %s and %s"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_elections_date_between_dates(self, start_date, end_date):
        """get elections date between two dates"""
        query = "SELECT date FROM elections WHERE date between %s and %s and is_election = 1"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_holidays_between_dates(self, start_date, end_date):
        """get holidays data between two dates"""
        query = "SELECT * FROM muslim_holidays " \
                "INNER JOIN jewish_holidays using(date) " \
                "INNER JOIN chris_holidays using(date) " \
                "WHERE date between %s and %s"
        result = self.__db_connection.fetch(query, (start_date, end_date,))
        return result

    def get_attacks_info_by_date(self, date):
        """get attacks information by date"""
        query = "SELECT city, nwound, ndeath, summary FROM attacks_info WHERE date = %s"
        result = self.__db_connection.fetch(query, (date,))
        return result

    # No table yet
    def get_model_predictions(self):
        """get model predictions"""
        query = "SELECT date FROM prediction WHERE prediction = 1"
        result = self.__db_connection.fetch(query)
        return result

    def get_real_result_waves(self):
        query = "SELECT class FROM annomaly_detection"
        result = self.__db_connection.fetch(query)
        return result
