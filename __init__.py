import base64
import pickle

from flask import Flask, render_template, request, Response, jsonify
from flask_restful import Api, Resource
from functions import *
app = Flask(__name__)
api = Api(app)

class LoginApi(Resource):
    def post(self):
        error = ''
        try:
            username = request.form['username']
            password = request.form['password']
            resp = login(username, password)
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
            result = get_model_data()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

class AnomaliesApi(Resource):
        def get(self):
            error = ''
            try:
                result = get_anomaly_detection()
                resp = jsonify(result)
                resp.status_code = 200
                return resp
            except Exception as e:
                return Response(status=404)

class TrainModelApi(Resource):
    def post(self):
        resp = load_data()

class ModelDateResultApi(Resource):
    def get(self):
        try:
            result = get_model_date_prediction()
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
            load_model_date_prediction(accuracy_df, prediction_df)
        except Exception as e:
            return Response(status=404)

class RecallAndPrecisionApi(Resource):
    def get(self):
        error = ''
        try:
            result = get_confusion_matrix()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            load_confusion_matrix(df)
        except Exception as e:
            return Response(status=404)

class HyperparmetersApi(Resource):
    def get(self):
        error = ''
        try:
            result = get_hyperparameters()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            load_hyperparameters(df)
        except Exception as e:
            return Response(status=404)

class FeaturesApi(Resource):
    def get(self):
        error = ''
        try:
            result = get_features()
            resp = jsonify(result)
            resp.status_code = 200
            return resp
        except Exception as e:
            return Response(status=404)

    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            load_features(df)
        except Exception as e:
            return Response(status=404)

class TestApi(Resource):
    def post(self):
        try:
            requested_file = request.form['pickled_df']
            df = pickle.loads(base64.b64decode(requested_file.encode()))
            resp = load_test(df)
        except Exception as e:
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
api.add_resource(TestApi, '/Test')

if __name__ == "__main__":
    app.run()

