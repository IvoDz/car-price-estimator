from flask import Flask, render_template, jsonify, request, url_for
import joblib
import torch 
from utils.model_utils import transform_raw_input_to_df, initialize_model

models_brands = joblib.load('utils/models.joblib')
engines = joblib.load('utils/engines.joblib')
model = initialize_model()

app = Flask(__name__)

@app.route('/')
def index():
    brands = list(models_brands.keys())
    return render_template('index.html', brands=brands, engines=engines)

@app.route('/get_models')
def get_all_models():
    return jsonify(models_brands)

@app.route('/get_models/<brand>')
def get_models(brand):
    models = models_brands.get(brand, [])
    return jsonify(models)

@app.route('/prediction', methods=['POST'])
def submit_form():
    brand = request.form['brand']
    car_model = request.form['model']
    engine = request.form['engine']
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])

    sample = transform_raw_input_to_df(brand, car_model, engine, year, mileage)

    with torch.no_grad():
        inputs = torch.tensor(sample, dtype=torch.float32)
        predictions = model(inputs).numpy()
        print(predictions)

    return render_template('prediction.html', brand=brand, model=car_model, engine=engine, year=year, mileage=mileage, predictions=predictions[0])  



if __name__ == '__main__':
    app.run(debug=True)