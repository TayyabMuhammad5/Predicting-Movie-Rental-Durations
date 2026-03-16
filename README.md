# Predicting Movie Rental Durations

## Project Overview
A DVD rental company needs to improve their inventory planning by accurately predicting how many days a customer will rent a DVD. This project builds a regression model to predict the `rental_length_days` based on various features of the rental and the movie itself. The ultimate business goal is to achieve a Mean Squared Error (MSE) of 3 or less on the test set.

## Dataset
- **File**: `rental_info.csv`
- **Features**: `rental_date`, `return_date`, `amount`, `release_year`, `rental_rate`, `length`, `replacement_cost`, `special_features`, and MPAA rating dummy variables (`NC-17`, `PG`, `PG-13`, `R`).
- **Target Variable**: `rental_length_days` (Engineered from `return_date` - `rental_date`)

## Technologies Used
- **Python**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Lasso Regression, Linear Regression, RandomForestRegressor, RandomizedSearchCV)

## Methodology
1. **Feature Engineering**: Created the target variable `rental_length_days` and parsed datetime columns. Extracted binary flags from `special_features` (e.g., deleted scenes, behind the scenes).
2. **Feature Selection**: Used Lasso Regression to identify and select features with positive coefficients.
3. **Model Training & Tuning**: 
   - Evaluated a baseline OLS Linear Regression model on the Lasso-selected features.
   - Built a Random Forest Regressor and optimized hyperparameters (`n_estimators`, `max_depth`) using `RandomizedSearchCV`.
4. **Evaluation**: Achieved the target MSE of ≤ 3 using the tuned Random Forest model.
