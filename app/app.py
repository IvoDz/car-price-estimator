from flask import Flask, render_template, jsonify, request, url_for, abort
import joblib
import torch 
from utils.model_utils import transform_raw_input_to_df, initialize_model

models_brands = joblib.load('utils/models.joblib')
engines = joblib.load('utils/engines.joblib')
model = initialize_model()

app = Flask(__name__)

@app.route('/')
def index():
    """
    Navigate to homepage.
    """
    brands = list(models_brands.keys())
    return render_template('index.html', brands=brands, engines=engines)

@app.route('/get_models')
def get_all_models():
    """
    Gets all car models and brands from the dataset in JSON format.
    """
    return jsonify(models_brands)

@app.route('/get_models/<brand>')
def get_models(brand:str):
    """
    Gets all car models for given car brand in JSON format.

    Args:
        brand (str): Car brand
    """
    models = models_brands.get(brand, [])
    if not models:
        abort(404, description="Brand not found")   
    return jsonify(models)

@app.route('/prediction', methods=['POST', 'GET'])
def predict():
    """
    Retrieves form data, formats it accordingly and passes it to the model, that makes prediction based on the input.
    Renders the template that displays the prediction.
    """
    if request.method == 'POST':
        brand = request.form['brand']
        car_model = request.form['model']
        engine = request.form['engine']
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])
    elif request.method == 'GET':
        brand = request.args.get('brand', type=str)
        car_model = request.args.get('model', type=str)
        engine = request.args.get('engine', type=str)
        year = request.args.get('year', type=int, default=0)
        mileage = request.args.get('mileage', type=int, default=0)

    if not all([brand, car_model, engine, year, mileage]):
        abort(400, description="Missing or incorrect parameters")

    sample = transform_raw_input_to_df(brand, car_model, engine, year, mileage)

    with torch.no_grad():
        inputs = torch.tensor(sample, dtype=torch.float32)
        predictions = model(inputs).numpy()

    if request.method == 'POST':
        return render_template('prediction.html', brand=brand, model=car_model, engine=engine, year=year, mileage=mileage, predictions=int(predictions[0]))
    else: 
        return jsonify({ 
                "predicted_price": str(int(predictions[0])),
                "brand": brand,
                "model": car_model,
                "engine": engine,
                "year": year,
                "mileage": mileage  
                })



if __name__ == '__main__':
    app.run()

