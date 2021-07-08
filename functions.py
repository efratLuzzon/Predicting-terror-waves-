from DB import DBHelper

mysql = DBHelper()


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

def load_confusion_matrix(df):
    mysql.load_df(df, "confusion_matrix")

def get_model_date_prediction(df):
    result = mysql.fetch("SELECT * FROM accuracy INNER JOIN prediction using(date)")
    return result

def load_model_date_prediction(accuracy_df, prediction_df):
    mysql.load_df(accuracy_df, "accuracy")
    mysql.load_df(prediction_df, "prediction")

def get_hyperparameters():
    result = mysql.fetch("SELECT * FROM hyperparameters")
    return result

def load_hyperparameters(df):
    mysql.load_df(df, "hyperparameters")

def get_features():
    result = mysql.fetch("SELECT * FROM features_importance")
    return result

def load_features(df):
    mysql.load_df(df, "features_importance")

def load_test(df):
    mysql.load_df(df, "confusion_matrix")