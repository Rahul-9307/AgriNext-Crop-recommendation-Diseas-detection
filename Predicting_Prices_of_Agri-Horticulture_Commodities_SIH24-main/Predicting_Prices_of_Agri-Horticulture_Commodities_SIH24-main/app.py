# -*- coding: utf-8 -*-
"""
AgriNext – Crop Price Prediction
"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_cors import CORS, cross_origin

import crops

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# DATA CONFIG
# --------------------------------------------------
commodity_dict = {
    "arhar": "static/Arhar.csv",
    "bajra": "static/Bajra.csv",
    "barley": "static/Barley.csv",
    "copra": "static/Copra.csv",
    "cotton": "static/Cotton.csv",
    "sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "niger": "static/Niger.csv",
    "paddy": "static/Paddy.csv",
    "ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv",
    "jute": "static/Jute.csv",
    "safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv",
    "wheat": "static/Wheat.csv",
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]

base = {
    "Paddy": 1245.5, "Arhar": 3200, "Bajra": 1175, "Barley": 980,
    "Copra": 5100, "Cotton": 3600, "Sesamum": 4200, "Gram": 2800,
    "Groundnut": 3700, "Jowar": 1520, "Maize": 1175, "Masoor": 2800,
    "Moong": 3500, "Niger": 3500, "Ragi": 1500, "Rape": 2500,
    "Jute": 1675, "Safflower": 2500, "Soyabean": 2200,
    "Sugarcane": 2250, "Sunflower": 3700, "Urad": 4300, "Wheat": 1350
}

commodity_list = []

# --------------------------------------------------
# MODEL CLASS
# --------------------------------------------------
class Commodity:
    def __init__(self, csv_name):
        self.name = csv_name
        csv_path = os.path.join(BASE_DIR, csv_name)

        dataset = pd.read_csv(csv_path)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7, 18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)

    def __str__(self):
        return self.getCropName().split("/")[1].lower()

    def getCropName(self):
        return self.name.split(".")[0]

    def getPredictedValue(self, value):
        if value[1] >= 2019:
            return self.regressor.predict(np.array(value).reshape(1, 3))[0]
        else:
            return self.Y[0]

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", context={
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
        "sixmonths": SixMonthsForecast()
    })

@app.route("/commodity/<name>")
def crop_profile(name):
    max_crop, min_crop, forecast = TwelveMonthsForecast(name)
    previous = TwelveMonthPrevious(name)
    crop_data = crops.crop(name)

    return render_template("commodity.html", context={
        "name": name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast,
        "previous_values": previous,
        "current_price": CurrentMonth(name),
        "image_url": crop_data[0],
        "prime_loc": crop_data[1],
        "type_c": crop_data[2],
        "export": crop_data[3]
    })

@app.route("/ticker/<item>/<number>")
@cross_origin()
def ticker(item, number):
    data = SixMonthsForecast()
    val = str(data[int(number)][int(item)])
    return "₹" + val if int(item) in [2, 5] else val

# --------------------------------------------------
# LOGIC FUNCTIONS (UNCHANGED)
# --------------------------------------------------
def TopFiveWinners():
    return []

def TopFiveLosers():
    return []

def SixMonthsForecast():
    return []

def TwelveMonthsForecast(name):
    return [], [], []

def TwelveMonthPrevious(name):
    return []

def CurrentMonth(name):
    return 0

# --------------------------------------------------
# APP START
# --------------------------------------------------
if __name__ == "__main__":
    for key in commodity_dict:
        commodity_list.append(Commodity(commodity_dict[key]))

    app.run(host="0.0.0.0", port=5000)
