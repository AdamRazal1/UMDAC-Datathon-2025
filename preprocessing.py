import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller  # Added for stationarity check

# 1. Load Datasets
main_df = pd.read_csv('Datathon Dataset.xlsx - Data - Main.csv').drop(['Assignment'], axis=1)
category_df = pd.read_csv('Datathon Dataset.xlsx - Others - Category Linkage.csv').rename(columns={'Category' : 'Category Flow'})
country_df = pd.read_csv('Datathon Dataset.xlsx - Others - Country Mapping.csv')
balance_df = pd.read_csv('Datathon Dataset.xlsx - Data - Cash Balance.csv')

# 2. Merge and Preprocess
main_df = main_df.dropna(axis=1, how='all')
df = pd.merge(main_df, category_df, left_on='Category', right_on='Category Names', how='left')
df = pd.merge(df, country_df, left_on='Name', right_on='Code', how='left', suffixes=('', '_country'))

# 3. Date and Numeric Conversion
df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
if df['Amount in USD'].dtype == 'object':
    df['Amount in USD'] = df['Amount in USD'].str.replace(',', '')
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce').fillna(0)

# 4. Net Cash Flow Calculation
df["Cash Inflow (USD)"] = df["Amount in USD"].apply(lambda x: x if x > 0 else 0)
df["Cash Outflow (USD)"] = df["Amount in USD"].apply(lambda x: abs(x) if x < 0 else 0)
df['Net Cash Flow (USD)'] = df['Cash Inflow (USD)'] - df['Cash Outflow (USD)']

# metric evaluation using rmse and mape
def evaluate(actual, forecast, title):
    # Match the dates between actual data and forecast
    common_dates = actual.index.intersection(forecast.index)
    actual_segment = actual.loc[common_dates]
    forecast_segment = forecast.loc[common_dates]
    
    rmse = np.sqrt(mean_squared_error(actual_segment, forecast_segment))
    mape = mean_absolute_percentage_error(actual_segment, forecast_segment)
    
    print(f"--- {title} ---")
    print(f"RMSE: {rmse:,.2f} USD")
    print(f"MAPE: {mape:.2%}\n")

# Actual Forecasting 

# Net Cash Flow

# Aggregate to Weekly Time Series
ts_data = df.set_index('Pstng Date')['Net Cash Flow (USD)'].resample('W').sum()

# creating model
model = SARIMAX(ts_data, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False, 
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Weekly Net Cash Flow Forecast for One Month - Short Term

forecast_steps = 4
forecast_result = model_fit.get_prediction(start="2025-10-05", end="2025-11-30", dynamic = True)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Evaluation
mse = mean_squared_error(ts_data[-5:], forecast_mean[:5])
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(ts_data[-5:], forecast_mean[:5])
print(f"\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"Mean Absolute Percentage Error: {mape:,.2f}")

plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data, label = 'Actual', color = 'blue')
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Weekly Net Cash Flow Forecast (1 Month)')
plt.xlabel('Date')
plt.ylabel('Net Cash Flow (USD)')
plt.legend()
plt.grid(True)
plt.savefig("Weekly Net Cash Flow Forecast (1 Month)")
plt.show()

# Weekly Net Cash Flow Forecast for Six Month - Long Term
forecast_steps = 26
forecast_result = model_fit.get_prediction(start="2025-05-18", end="2026-04-19", dynamic = True)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Evaluation
mse = mean_squared_error(ts_data[19:], forecast_mean[:25])
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(ts_data[19:], forecast_mean[:25])
print(f"\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"Mean Absolute Percentage Error: {mape:,.2f}")

plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data, label = 'Actual', color = 'blue')
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Weekly Net Cash Flow Forecast (6 Month)')
plt.xlabel('Date')
plt.ylabel('Net Cash Flow (USD)')
plt.legend()
plt.grid(True)
plt.savefig("Weekly Net Cash Flow Forecast (6 Month) ")
plt.show()


# Ending Cash Balance

initial_total_balance = balance_df['Carryforward Balance (USD)'].str.replace(',', '').astype('float').sum()
historical_ending_balance = initial_total_balance + ts_data.cumsum()

# creating model
model = SARIMAX(historical_ending_balance, 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False, 
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Weekly Ending Cash Balance for One Month - Short Term

forecast_steps = 4
forecast_result = model_fit.get_prediction(start="2025-10-05", end="2025-11-30", dynamic = True)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Evaluation
mse = mean_squared_error(historical_ending_balance[-5:], forecast_mean[:5])
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(historical_ending_balance[-5:], forecast_mean[:5])
print(f"\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"Mean Absolute Percentage Error: {mape:,.2f}")

    
plt.figure(figsize=(12, 6))
plt.plot(historical_ending_balance.index, historical_ending_balance, 
             label='Actual Ending Balance', color='blue', linewidth=2)
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Weekly Ending Cash Balance Forecast (1 Month)')
plt.xlabel('Date')
plt.ylabel('Ending Balance (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Weekly Ending Cash Balance Forecast (1 Month)")
plt.show()


# Weekly Ending Cash Balance for Six Month - Long Term

forecast_result = model_fit.get_prediction(start="2025-05-18", end="2026-04-19", dynamic = True)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Evaluation
mse = mean_squared_error(historical_ending_balance[19:], forecast_mean[:25])
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(historical_ending_balance[19:], forecast_mean[:25])
print(f"\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"Mean Absolute Percentage Error: {mape:,.2f}")


plt.figure(figsize=(12, 6))
plt.plot(historical_ending_balance.index, historical_ending_balance, 
             label='Actual Ending Balance', color='blue', linewidth=2)
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Weekly Ending Cash Balance Forecast (6 Month)')
plt.xlabel('Date')
plt.ylabel('Ending Balance (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("Weekly Ending Cash Balance Forecast (6 Month)")
plt.show()


# Categories Forecasting

# Select the top 2 categories to analyze
top_categories = df.groupby('Category')['Net Cash Flow (USD)'].sum().abs().nlargest(2).index.tolist()

plt.figure(figsize=(16, 10))

results = []

# Define forecast window
eval_start = "2025-10-05"
future_end = "2025-11-30"

for cat in top_categories:
    # Prepare weekly time series
    cat_ts_net = df[df['Category'] == cat].set_index('Pstng Date')['Amount in USD'].resample('W').sum().fillna(0)
    cat_ts_bal = cat_ts_net.cumsum()
    
    # Create a safe filename prefix (replace spaces with underscores)
    prefix = cat.replace(' ', '_')
    
    # --- A. Net Cash Flow Plot & Export ---
    model_net = SARIMAX(cat_ts_net, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4), enforce_stationarity=False).fit(disp=False)
    pred_net_res = model_net.get_prediction(start=eval_start, end=future_end)
    pred_net = pred_net_res.predicted_mean
    conf_net = pred_net_res.conf_int()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cat_ts_net.index, cat_ts_net, label='Actual Net Flow', color='blue', marker='o', markersize=4)
    plt.plot(pred_net.index, pred_net, label='SARIMA Forecast', color='red', linestyle='--')
    plt.fill_between(pred_net.index, conf_net.iloc[:, 0], conf_net.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f'{cat}: Weekly Net Cash Flow Forecast')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{prefix}_Net_Flow_Forecast.png') # Saves the file
    plt.show()
    plt.close() # Closes current plot to save memory
    
    # --- B. Balance Contribution Plot & Export ---
    model_bal = SARIMAX(cat_ts_bal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4), enforce_stationarity=False).fit(disp=False)
    pred_bal_res = model_bal.get_prediction(start=eval_start, end=future_end)
    pred_bal = pred_bal_res.predicted_mean
    conf_bal = pred_bal_res.conf_int()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cat_ts_bal.index, cat_ts_bal, label='Actual Balance Contribution', color='green', marker='o', markersize=4)
    plt.plot(pred_bal.index, pred_bal, label='SARIMA Forecast', color='orange', linestyle='--')
    plt.fill_between(pred_bal.index, conf_bal.iloc[:, 0], conf_bal.iloc[:, 1], color='yellow', alpha=0.2)
    plt.title(f'{cat}: Weekly Balance Contribution Forecast')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{prefix}_Balance_Contribution_Forecast.png') # Saves the file
    plt.show()
    plt.close()

# Create a dataframe to hold multiple category forecasts
all_forecasts = pd.DataFrame()

for cat in top_categories:
    cat_ts = df[df['Category'] == cat].set_index('Pstng Date')['Net Cash Flow (USD)'].resample('W').sum().fillna(0)
    model = SARIMAX(cat_ts, order=(1,1,1), seasonal_order=(1,1,1,4), enforce_stationarity=False).fit(disp=False)
    # Get 4 weeks future forecast
    f_res = model.get_forecast(steps=4)
    all_forecasts[cat] = f_res.predicted_mean

# Plotting the stacked future forecast
all_forecasts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Forecasted Net Cash Flow Contribution by Category (Next 4 Weeks)")
plt.ylabel("USD")
plt.xlabel("Week Start Date")
plt.xticks(rotation=45)
plt.show()