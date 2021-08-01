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
        result = self.__db_connection.fetch("SELECT * FROM accounts where user_id = \'{user}\'".format(user=username))
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
        result = self.__db_connection.fetch("SELECT * FROM confusion_matrix")
        return result

    def get_model_date_prediction(self, year):
        """get accuracy and predictions by year"""
        result = self.__db_connection.fetch("SELECT * FROM accuracy INNER JOIN prediction using(date)")
        return result

    def get_hyperparameters(self, year):
        """get hyperparams for trained model by year"""
        result = self.__db_connection.fetch("SELECT * FROM hyperparameters")
        return result

    def get_features(self, year):
        """get features score for trained model by year"""
        result = self.__db_connection.fetch("SELECT * FROM features_importance")
        return result


def login(username, password):
    result = mysql.fetch("SELECT * FROM accounts where user_id = \'{user}\'".format(user=username))
    return result


def get_model_data():
    query = "SELECT * FROM chris_holidays " \
            "INNER JOIN jewish_holidays using(date) " \
            "INNER JOIN muslim_holidays using(date) " \
            "INNER JOIN elections using(date) " \
            "INNER JOIN (SELECT date,num_attack,succeed,num_deaths,num_wounded,orginaized," \
            "explosion,armed_assault,hijacking,infrastructure_attack,unarmed_assault From gtd) gtd_tb using(date) " \
            "INNER JOIN (SELECT date,max_temp,perciption From israel_weather) weather_tb using(date)"
    result = mysql.fetch(query)
    return result


def get_anomaly_detection():
    result = mysql.fetch("SELECT * FROM annomaly_detection")
    return result


def load_data(df):
    mysql.load_df(df, "table-name")


def get_confusion_matrix():
    result = mysql.fetch("SELECT * FROM confusion_matrix")
    return result


def get_model_date_prediction(df):
    result = mysql.fetch("SELECT * FROM accuracy INNER JOIN prediction using(date)")
    return result


'''


    


    
def load_test(df):
    mysql.load_df(df, "confusion_matrix")

'''


def get_hyperparameters():
    result = mysql.fetch("SELECT * FROM hyperparameters")
    return result


def get_features():
    result = mysql.fetch("SELECT * FROM features_importance")
    return result
