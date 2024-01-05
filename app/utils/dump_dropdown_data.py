import joblib
import pandas as pd

data = pd.read_csv("../../data/data.csv")

brand_model = data.groupby('brand')['model'].unique().apply(list).to_dict()
all_engine_types = sorted(data['engine'].unique())

joblib.dump(all_engine_types, 'engines.joblib')
joblib.dump(brand_model, 'models.joblib')