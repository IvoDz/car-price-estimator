# Tool for Estimating the Price of Used Cars

## The model was trained on a car listing dataset created by me. The dataset and a sample notebook can be found [here](https://www.kaggle.com/datasets/ivodzalbs/car-price-data-latvia/data).

### To use locally:
 
1.Clone the repo

    git clone https://github.com/IvoDz/car-price-estimator.git
2.cd into the working directory
    
    cd car-price-estimator/app    
3.(optional) Create virtual environment

    python3 -m venv venv
4.Install dependencies

    pip install -r requirements.txt
5.Run the server

    flask run

You can now access the app at port 5000 (localhost:5000).

You can either use a web interface to enter the features of your car and get a prediction, or use tools like curl or Postman to make requests by directly passing parameters in the query.

See both examples below:

**Predict Price Through Browser:**
1. Access localhost:5000/ in your browser
2. Enter the features of the car you are interested in:

![image](https://github.com/IvoDz/car-price-estimator/assets/97388815/06726ab7-404f-4774-91fa-aff66fb9cd84)

3. Press 'Predict' to see the model's prediction!

![image](https://github.com/IvoDz/car-price-estimator/assets/97388815/6e9fe7b9-1ade-41bb-85be-f2c9d63599f4)


**Predict via CLI:**
e.g.

    curl "http://localhost:5000/prediction?brand=BMW&model=X6&engine=1.6D&year=2022&mileage=100000"
Get prediction as JSON:

    {...,"predicted_price":"43448",...}.
