# House Price Prediction — ML Regression Pipeline (Kaggle)

A complete machine-learning pipeline to predict house prices based on the Kaggle “House Prices: Advanced Regression Techniques” dataset.

## 🚀 Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib / Seaborn
- SHAP
- GridSearchCV

## 📊 Project Highlights
- Performed EDA across 80+ features.
- Handled missing values, encoded categorical data, and fixed skewness.
- Built full ML pipeline with imputation, scaling, encoding, and XGBoost.
- Hyperparameter tuning using GridSearchCV.
- Achieved RMSE = 0.132 (log-price).
- SHAP-based interpretability to highlight key price-driving factors.

## 🧠 Pipeline Architecture
1. Data Loading  
2. EDA & Visualization  
3. Feature Engineering  
4. Model Training (XGBoost)  
5. GridSearchCV Hyperparameter Tuning  
6. Evaluation  
7. SHAP Analysis

## 📁 Repository Structure
```
house-price-prediction-ml/
├── data/
├── notebooks/
├── src/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the model
```
python src/train_model.py
```

### 3. Run SHAP analysis
```
python src/shap_analysis.py
```

## 🎯 Results
- Final Model: XGBoost Regressor
- Best RMSE: 0.132
- Top influential features:
  - OverallQual  
  - GrLivArea  
  - TotalBsmtSF  
  - GarageCars  
  - Neighborhood  

## 📜 License
MIT License
