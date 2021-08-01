import base64
import os
import pickle
import pandas as pd
from flask import Flask, request, Response, jsonify
from flask_restful import Api, Resource
from Anomaly_detection.AnomalyDetectionTimeSeries import AnomalyDetectionTimeSeries
from DB.XGBModelQueries import XGBModelQueries
from DataFrameCalender import DataFrameCalender

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
        try:
            requested_file = request.form['pickled_accuracy_df']
            accuracy_df = pickle.loads(base64.b64decode(requested_file.encode()))
            requested_file = request.form['pickled_prediction_df']
            prediction_df = pickle.loads(base64.b64decode(requested_file.encode()))
            db_queries.load_data(accuracy_df, "accuracy")
            db_queries.load_data(prediction_df, "prediction")
        except Exception as e:
            return Response(status=404)


class RecallAndPrecisionApi(Resource):
    def get(self):
        error = ''
        try:
            result = db_queries.get_confusion_matrix()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            db_queries.load_data(df, "confusion_matrix")
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

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            db_queries.load_data(df, "hyperparameters")
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

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            db_queries.load_data(df, "features_importance")
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


class UploadCsv(Resource):
    def post(self):
        error = ''
        try:
            for i in range(len(request.files)):
                print(i)
                file = request.files[str(i)]
                filename = file.filename
                destination = "\\".join([os.getcwd(), filename])
                file.save(destination)
            return Response(status=200)
        except Exception as e:
            # return render_template("login.html", error=error)
            return Response(status=404)


# Setup the Api resource routing
api.add_resource(LoginApi, '/Login')
api.add_resource(ModelDataApi, '/ModelData')
api.add_resource(AnomaliesApi, '/Anomalies')
api.add_resource(TrainModelApi, '/TrainModel')
api.add_resource(ModelDateResultApi, '/ModelDateResult')
api.add_resource(RecallAndPrecisionApi, '/RecallAndPrecision')
api.add_resource(HyperparmetersApi, '/Hyperparmeters')
api.add_resource(FeaturesApi, '/Features')
api.add_resource(UploadCsv, '/Uploadcsv')
api.add_resource(TestApi, '/Anomalies')

if __name__ == "__main__":
    app.run()
