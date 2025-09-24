# Training a Machine Learning Model to Forecast USD Purchases  

## Overview  
This project forecasts the company’s future USD purchasing needs using commodity price data.  
Since caustic soda is the largest USD expenditure, and historical analysis shows that USD purchases are **positively correlated** with caustic soda prices (with a lag of several days),  
we train a machine learning model to anticipate future USD requirements.  

The goal is to help the treasury team ensure adequate USD reserves are available in advance, avoiding liquidity issues while optimizing transaction timing.  

## Approach  

### Data Sources  
- **USD Buys**: Transaction records (`audusd deals.csv`) containing settlement dates and purchase amounts.  
- **Caustic Soda Prices**: Weekly commodity prices (`Caustic_Soda_Weekly_Price.csv`).  

### Feature Engineering  
- Lagged values of both price and buys.  
- Rolling statistics (mean, standard deviation).  
- Week-over-week changes.  
- Calendar features (week of year, month, quarter, year).  

### Model  
- Uses **Histogram-based Gradient Boosting Regressor** (`sklearn.ensemble.HistGradientBoostingRegressor`).  
- Target variable: log-transformed USD buy amounts.  
- Validation performed with a recent hold-out window of weekly observations.  

### Forecasting  
- Implements **recursive forecasting** to generate multi-step predictions.  
- Produces both weekly and aggregated monthly forecasts.  
- Visualizes historical relationships (buys lagged behind prices) and forecast performance.  

## Key Features of the Script  
- Data cleaning and transformation of raw CSV inputs.  
- Conversion to consistent weekly frequency.  
- Recursive forecast procedure that uses predicted buys as inputs for future steps.  
- Automatic evaluation with MAE, RMSE, and MAPE.  
- Visualization of actual vs. predicted purchases.  
- Customizable hyperparameters for lags, training window, and forecast horizon.  

## File Structure  
- **hgb_recent_month_forecast_recursive.py**  
  Main script containing data preparation, feature engineering, model training, evaluation, and forecasting logic.  

- **Input Data (user-provided)**  
  - `audusd_deals.csv`: Historical USD purchase records.  
  - `Caustic_Soda_Weekly_Price.csv`: Historical caustic soda commodity prices.  

