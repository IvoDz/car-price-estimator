from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import torch 
import torch.nn as nn
from utils.model_utils import *

brand_model_dict = joblib.load('utils/models.joblib')
engines = joblib.load('utils/engines.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    brands = list(brand_model_dict.keys())
    return render_template('index.html', brands=brands, engines=engines)

@app.route('/get_models')
def get_all_models():
    return jsonify(brand_model_dict)

@app.route('/get_models/<brand>')
def get_models(brand):
    models = brand_model_dict.get(brand, [])
    return jsonify(models)


if __name__ == '__main__':
    app.run(debug=True)