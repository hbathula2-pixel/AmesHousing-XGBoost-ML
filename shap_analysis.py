import shap
import xgboost as xgb
import pandas as pd

model = xgb.XGBRegressor()
model.load_model("models/xgboost_model.json")

df = pd.read_csv("data/train.csv").drop("SalePrice", axis=1)

explainer = shap.Explainer(model, df)
shap_values = explainer(df)

shap.summary_plot(shap_values, df)
