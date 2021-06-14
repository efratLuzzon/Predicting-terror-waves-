from flask import Flask, render_template, request, Response, jsonify
from DB import DBHelper

app = Flask(__name__)

@app.route('/login', methods=["GET", "POST"])
def login_page():
    error = ''
    try:
        if request.method == "POST":
            user_name = request.form['username']
            password = request.form['password']
            result = mysql.fetch("SELECT * FROM accounts where user_id = \'{user}\'".format(user=user_name))
            if(len(result) > 0):
                if result[0]["password"] == password:
                    return Response(status=200)
                else:
                    error = "Invalid credentials. Try Again."
                    return Response(status=404)
    except Exception as e:
        #return render_template("login.html", error=error)
        return Response(status=404)

@app.route('/anomalies', methods=["GET"])
def get_anomaly_detection():
    error = ''
    try:
        if request.method == "GET":
            result = mysql.fetch("SELECT * FROM annomaly_detection")
            resp = jsonify(result)
            resp.status_code = 200
            return resp
    except Exception as e:
        return Response(status=404)

@app.route('/modelData', methods=["GET"])
def get_model_data():
    error = ''
    try:
        if request.method == "GET":
            query = "SELECT * FROM chris_holidays " \
                    "INNER JOIN jewish_holidays using(date) " \
                    "INNER JOIN muslim_holidays using(date) " \
                    "INNER JOIN elections using(date) " \
                    "INNER JOIN (SELECT date,max_temp,perciption From israel_weather) weather_tb using(date)"
            result = mysql.fetch(query)
            resp = jsonify(result)
            resp.status_code = 200
            return resp
    except Exception as e:
        return Response(status=404)


if __name__ == "__main__":
    mysql = DBHelper()
    app.run()