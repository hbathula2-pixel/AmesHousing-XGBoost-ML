import json
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from preprocess import load_data, split_features_target

df = load_data("data/train.csv")
X_train, X_test, y_train, y_test = split_features_target(df)

model = xgb.XGBRegressor(objective="reg:squarederror")

param_grid = {
    "n_estimators": [300, 500],
    "learning_rate": [0.03, 0.05],
    "max_depth": [3, 5, 7],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("RMSE:", rmse)

best_model.save_model("models/xgboost_model.json")

with open("models/metrics.json", "w") as f:
    json.dump({"rmse": rmse}, f)
