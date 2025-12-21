import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller  # Added for stationarity check

# AstraZeneca Colour Palette Definitions 
AZ_MULBERRY = "#830051"   # Color 1 (Primary)
AZ_LIME_GREEN = "#C4D600" # Color 2 (Accent)
AZ_NAVY = "#003865"       # Color 3 (Support)
AZ_GRAPHITE = "#3F4444"   # Color 4 (Support)
AZ_LIGHT_BLUE = "#68D2DF" # Color 5 (Support)
AZ_MAGENTA = "#D0006F"    # Color 6 (Support)
AZ_PURPLE = "#3C1053"     # Color 7 (Support)
AZ_GOLD = "#F0AB00"       # Color 8 (Support)

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

# 1. Weekly Net Cash Flow Forecast (1 Month)
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data, label='Actual', color=AZ_NAVY, linewidth=2) # Using Navy for actuals 
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color=AZ_MULBERRY, linestyle='--') # Using Mulberry for forecast 
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color=AZ_LIGHT_BLUE, alpha=0.3) # Light Blue for confidence 
plt.title('Weekly Net Cash Flow Forecast (1 Month)', color=AZ_GRAPHITE, fontweight='bold')
plt.xlabel('Date', color=AZ_GRAPHITE)
plt.ylabel('Net Cash Flow (USD)', color=AZ_GRAPHITE)
plt.legend()
plt.grid(True, color=AZ_GRAPHITE, alpha=0.2) # Graphite for grid 
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

# 2. Weekly Net Cash Flow Forecast (6 Month)
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data, label='Actual', color=AZ_NAVY, linewidth=2)
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color=AZ_MULBERRY, linestyle='--')
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color=AZ_LIGHT_BLUE, alpha=0.3)
plt.title('Weekly Net Cash Flow Forecast (6 Month)', color=AZ_GRAPHITE, fontweight='bold')
plt.xlabel('Date', color=AZ_GRAPHITE)
plt.ylabel('Net Cash Flow (USD)', color=AZ_GRAPHITE)
plt.legend()
plt.grid(True, color=AZ_GRAPHITE, alpha=0.2)
plt.savefig("Weekly Net Cash Flow Forecast (6 Month) ")
plt.show()


# Ending Cash Balance

initial_total_balance = balance_df['Carryforward Balance (USD)'].str.replace(',', '').astype('float').sum()
historical_ending_balance = initial_total_balance + ts_data.cumsum()

# creating model

from statsmodels.tsa.statespace.sarimax import SARIMAX

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

    
# --- 1 Month Forecast Plot ---
plt.figure(figsize=(12, 6))
# Using Navy for the historical line
plt.plot(historical_ending_balance.index, historical_ending_balance, 
             label='Actual Ending Balance', color=AZ_NAVY, linewidth=2)
# Using Mulberry for the forecast line
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color=AZ_MULBERRY, linestyle='--')
# Using Light Blue for the confidence interval shading
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color=AZ_LIGHT_BLUE, alpha=0.3)

plt.title('Weekly Ending Cash Balance Forecast (1 Month)', color=AZ_GRAPHITE, fontweight='bold')
plt.xlabel('Date', color=AZ_GRAPHITE)
plt.ylabel('Ending Balance (USD)', color=AZ_GRAPHITE)
plt.legend()
plt.grid(True, alpha=0.3, color=AZ_GRAPHITE)
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


# --- 6 Month Forecast Plot ---
plt.figure(figsize=(12, 6))
# Using Navy for the historical line
plt.plot(historical_ending_balance.index, historical_ending_balance, 
             label='Actual Ending Balance', color=AZ_NAVY, linewidth=2)
# Using Mulberry for the forecast line
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color=AZ_MULBERRY, linestyle='--')
# Using Light Blue for the confidence interval shading
plt.fill_between(forecast_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color=AZ_LIGHT_BLUE, alpha=0.3)

plt.title('Weekly Ending Cash Balance Forecast (6 Month)', color=AZ_GRAPHITE, fontweight='bold')
plt.xlabel('Date', color=AZ_GRAPHITE)
plt.ylabel('Ending Balance (USD)', color=AZ_GRAPHITE)
plt.legend()
plt.grid(True, alpha=0.3, color=AZ_GRAPHITE)
plt.savefig("Weekly Ending Cash Balance Forecast (6 Month)")
plt.show()


# Categories Forecasting

# Select the top 2 categories to analyze
top_categories = df.groupby('Category')['Net Cash Flow (USD)'].sum().abs().nlargest(4).index.tolist()

plt.figure(figsize=(16, 10))

results = []

# Define forecast window
eval_start = "2025-10-05"
future_end = "2026-04-19"

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

# 3. Forecasted Net Cash Flow Contribution by Category (with Percentages)
all_forecasts = pd.DataFrame()
for cat in top_categories:
    cat_ts = df[df['Category'] == cat].set_index('Pstng Date')['Net Cash Flow (USD)'].resample('W').sum().fillna(0)
    model = SARIMAX(cat_ts, order=(1,1,1), seasonal_order=(1,1,1,4), enforce_stationarity=False).fit(disp=False)
    f_res = model.get_forecast(steps=26)
    all_forecasts[cat] = f_res.predicted_mean

az_palette = [AZ_MULBERRY, AZ_LIME_GREEN, AZ_NAVY, AZ_GRAPHITE, AZ_LIGHT_BLUE, AZ_MAGENTA, AZ_PURPLE, AZ_GOLD]

# Plotting the stacked bar chart
ax = all_forecasts.plot(kind='bar', stacked=True, figsize=(10, 6), color=az_palette)

# Calculate total for each bar to determine percentages
totals = all_forecasts.sum(axis=1)

# Adding percentage labels inside the bars
for container in ax.containers:
    # Calculate the percentage for each segment in the current category
    labels = []
    for i, v in enumerate(container):
        val = v.get_height()
        # Calculate percentage; handle division by zero if total is 0
        pct = (val / totals.iloc[i]) * 100 if totals.iloc[i] != 0 else 0
        # Only show labels for segments large enough to read (e.g., > 1%)
        labels.append(f'{pct:.1f}%' if abs(pct) > 1 else "")
    
    # Place labels in the center of the segments
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=9, fontweight='bold')

az_palette = [AZ_MULBERRY, AZ_LIME_GREEN, AZ_NAVY, AZ_GRAPHITE, AZ_LIGHT_BLUE, AZ_MAGENTA, AZ_PURPLE, AZ_GOLD]
all_forecasts.plot(kind='bar', stacked=True, figsize=(10, 6), color=az_palette)
plt.title("Forecasted Net Cash Flow Contribution by Category (Next 26 Weeks)", color=AZ_GRAPHITE, fontweight='bold')
plt.ylabel("USD", color=AZ_GRAPHITE)
plt.xlabel("Week Start Date", color=AZ_GRAPHITE)
plt.xticks(rotation=45)
plt.grid(axis='y', color=AZ_GRAPHITE, alpha=0.2)
plt.savefig("Forecasted Net Cash Flow Contribution by Category (Next 26 Weeks)")
plt.show()

# --- POWER BI EXPORT PREPARATION ---

# 1. Prepare Historical Data
historical_combined = pd.DataFrame({
    'Date': ts_data.index,
    'Net_Cash_Flow': ts_data.values,
    'Ending_Balance': historical_ending_balance.values,
    'Type': 'Actual'
})

# 2. Prepare Forecast Data (using the 6-month forecast results)
# Note: We take the mean predictions from your existing model_fit
forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(weeks=1), periods=26, freq='W')
future_res = model_fit.get_forecast(steps=26)
future_mean = future_res.predicted_mean
future_conf = future_res.conf_int()

forecast_combined = pd.DataFrame({
    'Date': future_mean.index,
    'Net_Cash_Flow': future_mean.values,
    'Ending_Balance': np.nan, # Balance needs to be calculated cumulatively
    'Type': 'Forecast',
    'Lower_Bound': future_conf.iloc[:, 0].values,
    'Upper_Bound': future_conf.iloc[:, 1].values
})

# Calculate forecasted Ending Balance based on last known actual
last_bal = historical_ending_balance.iloc[-1]
forecast_combined['Ending_Balance'] = last_bal + forecast_combined['Net_Cash_Flow'].cumsum()

# 3. Combine Actuals and Forecasts
final_export_df = pd.concat([historical_combined, forecast_combined], ignore_index=True)

# 4. (Optional) Add Category-level Forecasts for Power BI Drilldown
# We pivot your all_forecasts variable into a long format
category_export = all_forecasts.reset_index().melt(id_vars='index', var_name='Category', value_name='Forecasted_Flow')
category_export.rename(columns={'index': 'Date'}, inplace=True)

# 5. Export to CSV
final_export_df.to_csv('PowerBI_CashFlow_Main.csv', index=False)
category_export.to_csv('PowerBI_Category_Forecasts.csv', index=False)

print("Files 'PowerBI_CashFlow_Main.csv' and 'PowerBI_Category_Forecasts.csv' are ready for Power BI!")