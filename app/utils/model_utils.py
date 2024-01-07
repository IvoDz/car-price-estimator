import pandas as pd
import torch 
import torch.nn as nn
import joblib

scaler = joblib.load('utils/scaler.save')
    
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu1 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        return x
    
    
def transform_raw_input_to_df(brand, model, engine, year, mileage, sc=scaler):
    features = pd.read_pickle('utils/sample.pkl')
    features.drop('price', inplace=True)
    
    features['brand_' + brand] = 1
    features['model_' + model] = 1
    features['engine_' + engine] = 1
    
    scaled_nums = scale_numerical_input(year, mileage, sc)
    features["mileage"] = scaled_nums["mileage"]
    features["age"] = scaled_nums["age"]
    
    return features


def scale_numerical_input(year, mileage, scaler):
    user_input = {
        "age" : 2023-year,
        "mileage" : mileage/1000
    }
    
    input_df = pd.DataFrame([user_input])
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=["age", "mileage"])
    res = input_df_scaled.to_numpy()
    user_input["age"], user_input["mileage"] = res[0][0], res[0][1]
    
    return user_input
    
    
def initialize_model():
    input_size = 973
    hidden_size = 128
    output_size = 1  
    model = NeuralNetwork(input_size, hidden_size, output_size)
    state_dict = torch.load('weights/weights_2.pth')
    model.load_state_dict(state_dict)
    return model

