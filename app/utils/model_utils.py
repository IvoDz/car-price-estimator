import pandas as pd
import torch 
import torch.nn as nn

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
    
    
def transform_raw_input_to_df(brand, model, engine, age, mileage, df, sc):
    user_input = df.iloc[0].copy()
    user_input.loc[:] = 0
    user_input.drop('price', inplace=True)
    
    # Handling categoricals
    user_input['brand_' + brand] = 1
    user_input['model_' + model] = 1
    user_input['engine_' + engine] = 1
    
    # Handling numericals 
    scaled_nums = scale_numerical_input(age, mileage, sc)
    user_input["mileage"] = scaled_nums["mileage"]
    user_input["age"] = scaled_nums["age"]
    
    return user_input



def scale_numerical_input(age, mileage, scaler):
    user_input = {
        "age" : age,
        "mileage" : mileage/1000
    }
    
    input_df = pd.DataFrame([user_input])
    input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=["age", "mileage"])
    res = input_df_scaled.to_numpy()
    user_input["age"], user_input["mileage"] = res[0][0], res[0][1]
    return user_input
    
    
def init_model(weights, model_class):
    input_size = 972
    hidden_size = 128
    output_size = 1  
    model_eval = model_class(input_size, hidden_size, output_size)
    state_dict = torch.load(weights)
    model_eval.load_state_dict(state_dict)
    return model_eval

# data = transform_raw_input_to_df("Opel", "Insignia", "2.0D", 13, 239000, df, scaler)
# with torch.no_grad():
#     inputs = torch.tensor(data, dtype=torch.float32)
#     predictions = model.eval(inputs).numpy()