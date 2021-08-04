import base64
import os
import pickle
import pandas as pd
from flask import Flask, request, Response, jsonify
from flask_restful import Api, Resource
from ML_Models.Anomaly_detection.AnomalyDetectionTimeSeries import AnomalyDetectionTimeSeries
from DB.XGBModelQueries import XGBModelQueries
from DataFrameCalender import DataFrameCalender
from ML_Models.XgbClassification import XgbClassification

app = Flask(__name__)
api = Api(app)
# DB = my_dql_db()

db_queries = XGBModelQueries()


class LoginApi(Resource):
    def post(self):
        error = ''
        try:
            username = request.json['username']
            password = request.json['password']
            resp = db_queries.login(username, password)
            if (len(resp) > 0):
                if resp[0]["password"] == password:
                    return Response(status=200)
                else:
                    error = "Invalid credentials. Try Again."
                    return Response(status=404)
        except Exception as e:
            return Response(status=404)




class ModelDataApi(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_model_data()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def get(self, date):
        error = ''
        try:
            result = db_queries.get_model_data_per_date(date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)
    def post(self):
        error = ''
        try:
            for i in range(len(request.files)):
                file = request.files[str(i)]
                filename = file.filename
                table_name = self.get_table_name(filename)
                if table_name is "":
                    return Response(status=404)
                destination = "\\".join([os.getcwd(), filename])
                file.save(destination)
                new_data_model = pd.read_csv(destination)
                DataFrameCalender.set_date_time_index(new_data_model, "date", new_data_model["date"])
                db_queries.load_data(new_data_model, table_name)
            return Response(status=200)
        except Exception as e:
            # return render_template("login.html", error=error)
            return Response(status=404)
    def get_table_name(self, name):
        if name.startswith("gtd"):
            return "gtd"
        elif name.startswith("israel_weather"):
            return "israel_weather"
        elif name.startswith("jewish_holidays"):
            return "jewish_holidays"
        elif name.startswith("chris_holidays"):
            return "chris_holidays"
        elif name.startswith("muslim_holidays"):
            return "muslim_holidays"
        elif name.startswith("elections"):
            return "elections"
        elif name.startswith("google_trends_israel"):
            return "google_trends_israel"
        elif name.startswith("google_trends_palestine"):
            return "google_trends_palestine"
        elif name.startswith("terror_wave_details"):
            return "terror_wave_details"
        else:
            return ""




class AnomaliesApi(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_anomaly_detection()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        """re-train anomalies"""
        try:
            result = db_queries.get_num_attacks_per_day()
            result = pd.DataFrame(result)
            if len(result) is not 0:
                DataFrameCalender.set_date_time_index(result, "date", result["date"])
                anomaly_detection = AnomalyDetectionTimeSeries(result)
                loss_df = anomaly_detection.get_score()
                db_queries.load_data(loss_df, "annomaly_detection")
                resp = jsonify("")
                resp.status_code = 200
                return resp
            else:
                return Response(status=420)
        except Exception as e:
            return Response(status=404)


class TrainModelApi(Resource):
    def post(self):
        resp = db_queries.load_data()


class ModelDateResultApi(Resource):
    def get(self):
        try:
            result = db_queries.get_model_date_prediction()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        test_year = request.json['test_year']
        try:
            data = db_queries.get_model_data()
            data = pd.DataFrame(data)
            if len(data) is not 0:
                DataFrameCalender.set_date_time_index(data, "date", data["date"])
                data.drop('date', axis='columns', inplace=True)
            else:
                return Response(status=404)
            accuracy, accuracy_mean, xgb_model, predictions = XgbClassification.get_trained_model(data, test_year)

            predictions_df = xgb_model.get_prediction_model(predictions=predictions)
            hyperparameters_df = xgb_model.get_hyperparams_model()
            confusion_matrix_df = xgb_model.get_confusion_matrix_model(predictions=predictions)
            accuracy_result_df = xgb_model.get_accuracy_result(accuracy_mean, accuracy)
            features_importance_df = xgb_model.get_features_importance()

            db_queries.load_data(predictions_df, "prediction")
            db_queries.load_data(hyperparameters_df, "hyperparams")
            db_queries.load_data(confusion_matrix_df, "confusion_matrix")
            db_queries.load_data(accuracy_result_df, "accuracy")
            db_queries.load_data(features_importance_df, "features")
        except Exception as e:
            return Response(status=404)


class ConfusionMatrix(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_confusion_matrix()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)



class HyperparmetersApi(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_hyperparameters()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)



class FeaturesApi(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_features()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)




class TestApi(Resource):
    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            resp = db_queries.load_data(df, "confusion_matrix")
        except Exception as e:
            return Response(status=404)





class WeatherApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_weather_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class AttacksApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_attacks_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class GoogleTrendsIsraelApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_google_trends_israel_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class GoogleTrendsPalestineApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_google_trends_palestine_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class ElectionsApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_elections_date_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class HolidaysApi(Resource):
    def get(self):
        try:
            start_date = request.args.get('startDate')
            end_date = request.args.get('endDate')
            result = db_queries.get_holidays_between_dates(start_date, end_date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class AttacksInfoApi(Resource):
    def get(self):
        try:
            date = request.args.get('date')
            result = db_queries.get_attacks_info_by_date(date)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


class ModelPredictionsApi(Resource):
    def get(self):
        try:
            result = db_queries.get_model_predictions()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)


# Setup the Api resource routing
api.add_resource(LoginApi, '/Login')
api.add_resource(ModelDataApi, '/ModelData')
api.add_resource(AnomaliesApi, '/Anomalies')
api.add_resource(TrainModelApi, '/TrainModel')
api.add_resource(ModelDateResultApi, '/ModelDateResult')
api.add_resource(ConfusionMatrix, '/ConfusionMatrix')
api.add_resource(HyperparmetersApi, '/Hyperparmeters')
api.add_resource(FeaturesApi, '/Features')
api.add_resource(TestApi, '/Anomalies')
api.add_resource(WeatherApi, '/Weather')
api.add_resource(AttacksApi, '/Attacks')
api.add_resource(GoogleTrendsIsraelApi, '/GoogleTrendsIsrael')
api.add_resource(GoogleTrendsPalestineApi, '/GoogleTrendsPalestine')
api.add_resource(ElectionsApi, '/Elections')
api.add_resource(HolidaysApi, '/Holidays')
api.add_resource(AttacksInfoApi, '/AttacksInfo')
api.add_resource(ModelPredictionsApi, '/ModelPredictions')


if __name__ == "__main__":
    app.run()
