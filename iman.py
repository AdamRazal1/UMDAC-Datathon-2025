import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.patches as mpatches

# ==========================================
# 0. HELPER: SMART LABELING FUNCTIONS
# ==========================================
def add_bar_labels(ax, format_str='{:.0f}', offset=5):
    """Adds text labels to the end of bars in a horizontal bar chart."""
    for p in ax.patches:
        width = p.get_width()
        if width > 0: # Only label if bar exists
            ax.annotate(format_str.format(width),
                        (width, p.get_y() + p.get_height() / 2),
                        xytext=(offset, 0), 
                        textcoords='offset points',
                        va='center', fontweight='bold', fontsize=14, color='black') 

def add_vertical_bar_labels(ax, format_str='{:.0f}'):
    """Adds text labels to the top of bars in a vertical bar chart."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format_str.format(height),
                        (p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=14) 
# ==========================================
# 1. SETUP & DATA PREP
# ==========================================
color_normal = '#A0E7E5' 
color_anomaly = '#FFAEBC' 
color_weekend = '#FBE7C6' 
sns.set_theme(style="whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color_normal, color_anomaly])

df_main = pd.read_csv("Datathon Dataset.xlsx - Data - Main.csv", low_memory=False)
df_country = pd.read_csv("Datathon Dataset.xlsx - Others - Country Mapping.csv")
df_rates = pd.read_csv("Datathon Dataset.xlsx - Others - Exchange Rate.csv")
df_cat_link = pd.read_csv("Datathon Dataset.xlsx - Others - Category Linkage.csv")

# Clean numerical strings
df_main['Amt in loc.cur.'] = df_main['Amt in loc.cur.'].astype(str).str.replace(',', '').astype(float)
df_main['Pstng Date'] = pd.to_datetime(df_main['Pstng Date'], errors='coerce')

df_corrected = df_main.merge(df_country, left_on='Name', right_on='Code', how='left')
# Merge to get correct Exchange Rate
df_corrected = df_corrected.merge(df_rates, left_on='Currency', right_on='Code', how='left', suffixes=('', '_rate'))

# Recalculate USD Amount
def calculate_usd(row):
    rate = row['Rate (USD)']
    local_amt = row['Amt in loc.cur.']
    if pd.notnull(rate) and rate != 0:
        return local_amt / rate
    return row['Amount in USD']
df_corrected['Amount_USD_Corrected'] = df_corrected.apply(calculate_usd, axis=1)

le = LabelEncoder()
df_corrected['Category_Enc'] = le.fit_transform(df_corrected['Category'].astype(str))
df_corrected['Entity_Enc'] = le.fit_transform(df_corrected['Name'].astype(str))
df_corrected['Type_Enc'] = le.fit_transform(df_corrected['Type'].astype(str))
df_corrected['PK_Enc'] = le.fit_transform(df_corrected['PK'].astype(str))
df_corrected['DayOfWeek'] = df_corrected['Pstng Date'].dt.dayofweek
df_corrected['DayOfMonth'] = df_corrected['Pstng Date'].dt.day

features = ['Amount_USD_Corrected', 'Category_Enc', 'Entity_Enc', 'Type_Enc', 'PK_Enc', 'DayOfWeek', 'DayOfMonth']
X = df_corrected[features].fillna(0)

# A. Isolation Forest
iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X)
df_corrected['Is_ML_Anomaly'] = iso.predict(X) == -1
df_corrected['Anomaly_Score'] = iso.decision_function(X)

# B. Sign Mismatch Rule
df_corrected = df_corrected.merge(df_cat_link, left_on='Category', right_on='Category Names', how='left')
df_corrected.rename(columns={'Category_y': 'Flow_Direction', 'Category_x': 'Category_Main'}, inplace=True)
df_corrected['Is_Sign_Anomaly'] = np.where(
    ((df_corrected['Flow_Direction'] == 'Outflow') & (df_corrected['Amount_USD_Corrected'] > 0)) |
    ((df_corrected['Flow_Direction'] == 'Inflow') & (df_corrected['Amount_USD_Corrected'] < 0)),
    True, False
)

# C. Final Flag
df_corrected['Is_Anomaly'] = df_corrected['Is_ML_Anomaly'] | df_corrected['Is_Sign_Anomaly']
print(f"   -> Total Anomalies Found: {df_corrected['Is_Anomaly'].sum()}")
print(f"        -> Total ML Anomalies Found: {df_corrected['Is_ML_Anomaly'].sum()}")
print(f"        -> Total Sign Mismatch Anomalies Found: {df_corrected['Is_Sign_Anomaly'].sum()}")

# Feature Importance
clf_explain = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
clf_explain.fit(X, df_corrected['Is_Anomaly'])
label_map = {'Amount_USD_Corrected': 'Transaction Amount','Category_Enc': 'Category','Entity_Enc': 'Entity (Country)','Type_Enc': 'Document Type','PK_Enc': 'Posting Key','DayOfWeek': 'Day of Week','DayOfMonth': 'Day of Month'}
importances = pd.DataFrame({'Feature': features, 'Importance': clf_explain.feature_importances_}).sort_values('Importance', ascending=False)
importances['Feature'] = importances['Feature'].map(label_map)

# ==========================================
# 6. GENERATE VISUALIZATIONS
# ==========================================
# --- EDA - Weekly Net Cash Flow ---
plt.figure(figsize=(14, 6))
df_weekly = df_corrected.set_index('Pstng Date').resample('W')['Amount_USD_Corrected'].sum().reset_index()
sns.lineplot(data=df_weekly, x='Pstng Date', y='Amount_USD_Corrected', marker='o', color='#2A9D8F', linewidth=2)
plt.title('EDA: Weekly Net Cash Flow Trend (2025)', fontsize=14, fontweight='bold')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.tight_layout()
plt.savefig('eda_cashflow.png')

# --- EDA - Volume by Day of Week ---
plt.figure(figsize=(10, 5))
day_counts = df_corrected['DayOfWeek'].value_counts().sort_index()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
colors = [color_normal]*5 + [color_weekend]*2
ax = plt.bar(days, day_counts.values, color=colors, edgecolor='black')
add_vertical_bar_labels(plt.gca()) 
plt.title('EDA: Transaction Volume by Day (Weekend Highlight)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_dayofweek.png')

# --- EDA - Total Transaction Volume by Category ---
plt.figure(figsize=(12, 9))
cat_counts = df_corrected['Category_Main'].value_counts()
ax = sns.barplot(
    x=cat_counts.values, 
    y=cat_counts.index, 
    hue=cat_counts.index, 
    legend=False, 
    palette=sns.light_palette(color_normal, reverse=True, n_colors=len(cat_counts))
)
add_bar_labels(ax, '{:.0f}') 
plt.title('EDA: Total Transaction Volume by Category', fontsize=14, fontweight='bold')
plt.xlabel('Number of Transactions')
plt.tight_layout()
plt.savefig('eda_category_volume.png')

# --- EDA - Total Transaction Volume by Entity Country ---
plt.figure(figsize=(12, 9))
country_col = 'Country' if 'Country' in df_corrected.columns else 'Name' 
country_counts = df_corrected[country_col].value_counts().head(15) 
ax = sns.barplot(
    x=country_counts.values,
    y=country_counts.index,
    hue=country_counts.index,
    legend=False,
    palette=sns.light_palette(color_normal, reverse=True, n_colors=len(country_counts))
)
add_bar_labels(ax, '{:.0f}')
plt.title('Total Volume by Entity Country', fontsize=14, fontweight='bold')
plt.xlabel('Number of Transactions')
plt.tight_layout()
plt.savefig('eda_entity_country_volume.png')

# --- Sign Mismatch Breakdown ---
plt.figure(figsize=(21, 3))
mismatch_data = df_corrected[df_corrected['Is_Sign_Anomaly']]
if not mismatch_data.empty:
    sns.countplot(
        data=mismatch_data, y='Category_Main', hue='Flow_Direction', palette='Reds_r'
    )
    plt.title('Rule Check: Sign Mismatches by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sign_mismatch_breakdown.png')

# --- ML - Anomalies by Category ---
plt.figure(figsize=(12, 9))
anomaly_counts = df_corrected[df_corrected['Is_Anomaly']]['Category_Main'].value_counts().reset_index()
anomaly_counts.columns = ['Category', 'Anomaly_Count']
ax = sns.barplot(
    data=anomaly_counts, x='Anomaly_Count', y='Category', hue='Category', legend=False,
    palette=sns.light_palette(color_anomaly, reverse=True, n_colors=len(anomaly_counts))
)
add_bar_labels(ax, '{:.0f}') 
plt.title('ML Result: Number of Anomalies by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ml_category_count.png')

# --- ML - Feature Importance ---
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=importances, x='Importance', y='Feature', hue='Feature', legend=False,
    palette=sns.light_palette(color_normal, reverse=True, n_colors=len(importances))
)
add_bar_labels(ax, '{:.2f}') 
plt.title('ML Interpretation: Dominant Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('ml_feature_importance.png')

# --- ML - Score vs Amount ---
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_corrected[~df_corrected['Is_ML_Anomaly']], x='Amount_USD_Corrected', y='Anomaly_Score',
                color=color_normal, alpha=0.5, s=20, label='Normal')
sns.scatterplot(data=df_corrected[df_corrected['Is_ML_Anomaly']], x='Amount_USD_Corrected', y='Anomaly_Score',
                color=color_anomaly, alpha=0.9, s=40, edgecolor='k', label='Anomaly')
plt.title('ML Interpretation: Transaction Amount vs. Anomaly Score', fontsize=14, fontweight='bold')
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig('ml_score_vs_amount.png')

# --- ML - Score Distribution ---
plt.figure(figsize=(12, 5))
sns.histplot(data=df_corrected, x='Anomaly_Score', hue='Is_ML_Anomaly', kde=True, 
             palette={False: color_normal, True: color_anomaly}, bins=50, alpha=0.6)
plt.title('ML Interpretation: Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('ml_score_distribution.png')

# --- Anomaly Percentage by Category (Ratio) ---
plt.figure(figsize=(12, 9))
cat_stats = df_corrected.groupby('Category_Main')['Is_Anomaly'].agg(['count', 'sum']).reset_index()
cat_stats.columns = ['Category', 'Total_Count', 'Anomaly_Count']
cat_stats['Anomaly_Pct'] = (cat_stats['Anomaly_Count'] / cat_stats['Total_Count']) * 100
cat_stats_filtered = cat_stats[cat_stats['Total_Count'] > 20].sort_values('Anomaly_Pct', ascending=False).head(10)


ax = sns.barplot(
    data=cat_stats_filtered,
    x='Anomaly_Pct', 
    y='Category', 
    hue='Category',
    legend=False,
    palette=sns.light_palette(color_anomaly, reverse=True, n_colors=10)
)
add_bar_labels(ax, '{:.1f}%') 
plt.title('Anomaly Rate (%)', fontsize=14, fontweight='bold')
plt.xlabel('Percentage of Transactions Flagged as Anomalies (%)')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('ml_anomaly_percentage.png')

# --- Anomaly Count by Entity Country (ML) ---
plt.figure(figsize=(12, 9))
# Filter for anomalies ONLY
anomaly_country_counts = df_corrected[df_corrected['Is_Anomaly']][country_col].value_counts().head(15)

ax = sns.barplot(
    x=anomaly_country_counts.values,
    y=anomaly_country_counts.index,
    hue=anomaly_country_counts.index,
    legend=False,
    palette=sns.light_palette(color_anomaly, reverse=True, n_colors=len(anomaly_country_counts))
)
add_bar_labels(ax, '{:.0f}')
plt.title('Anomaly Count by Entity Country', fontsize=14, fontweight='bold')
plt.xlabel('Number of Anomalies')
plt.tight_layout()
plt.savefig('ml_entity_country_anomalies.png')


# --- Anomaly Ratio (Category) - 100% Stacked Bar ---
plt.figure(figsize=(12, 9))

# 1. Group & Count
cat_counts = df_corrected.groupby(['Category_Main', 'Is_Anomaly']).size().unstack(fill_value=0)

# 2. Filter for Volume (Keep only categories with > 20 transactions)
cat_counts = cat_counts[cat_counts.sum(axis=1) > 20]

# 3. Calculate Percentages (Normalize to 100%)
cat_pct = cat_counts.div(cat_counts.sum(axis=1), axis=0) * 100

# 4. Sort by Anomaly Percentage (High risk at top)
if True in cat_pct.columns:
    cat_pct = cat_pct.sort_values(by=True, ascending=True) # Ascending for horizontal bar (highest at top visually usually requires tweaking but standard barh builds bottom up)
    # Actually for barh, the last item in dataframe is at the top. So we sort Ascending to put Highest Anomaly at Bottom? 
    # Let's sort Descending so highest anomaly is at the top.
    # Wait, pandas plot barh puts index 0 at bottom. So sort Ascending to put Highest at Top.
    cat_pct = cat_pct.sort_values(by=True, ascending=True)

# 5. Plot 100% Stacked Bar
ax = cat_pct.plot(kind='barh', stacked=True, color=[color_normal, color_anomaly], figsize=(12, 6), width=0.8)

# 6. Customize
plt.title('Risk Concentration: Anomaly vs Normal Ratio (%) (100% Stacked)', fontsize=14, fontweight='bold')
plt.xlabel('Percentage (%)')
plt.ylabel('Category')
plt.legend(title='Transaction Type', labels=['Normal', 'Anomaly'], bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Add Smart Labels (Only label the Anomaly portion)
for c in ax.containers:
    # c is a group of bars (e.g. all the "Normal" bars, then all the "Anomaly" bars)
    # We only want to label the "Anomaly" bars (which correspond to True, usually the second container)
    # Let's check the color or index.
    if c.get_label() == 'True' or c.get_label() == True: # The label comes from the column name
        labels = [f'{w:.1f}%' if w > 0 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=14, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('anomaly_percentage_stacked.png')

# ---  ML - Timeline of Anomalies ---
plt.figure(figsize=(14, 6))
sns.scatterplot(data=df_corrected[~df_corrected['Is_Anomaly']], x='Pstng Date', y='Amount_USD_Corrected', 
                color=color_normal, alpha=0.6, s=20, label='Normal Transaction')
sns.scatterplot(data=df_corrected[df_corrected['Is_Anomaly']], x='Pstng Date', y='Amount_USD_Corrected', 
                color=color_anomaly, alpha=0.9, s=40, edgecolor='k', label='Anomaly')
plt.title('4. ML Result: Timeline of Anomalies', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('ml_timeline.png')

# --- Anomaly Ratio (Entity Country) - 100% Stacked Bar ---
plt.figure(figsize=(9, 18))

# 1. Group by Country and Anomaly Status
country_counts_stack = df_corrected.groupby([country_col, 'Is_Anomaly']).size().unstack(fill_value=0)

# 2. Filter for Volume (Keep only countries with > 10 transactions to avoid noise)
country_counts_stack = country_counts_stack[country_counts_stack.sum(axis=1) > 10]

# 3. Calculate Percentages (Normalize to 100%)
country_pct = country_counts_stack.div(country_counts_stack.sum(axis=1), axis=0) * 100

# 4. Sort by Anomaly Percentage
country_pct = country_pct.sort_values(by=True, ascending=True)

# 5. Plot 100% Stacked Bar
ax = country_pct.plot(kind='barh', stacked=True, color=[color_normal, color_anomaly], figsize=(12, 6), width=0.8)

# 6. Customize
plt.title('Risk Concentration: Anomaly Ratio by Country (100% Stacked)', fontsize=14, fontweight='bold')
plt.xlabel('Percentage (%)')
plt.ylabel('Country')
plt.legend(title='Transaction Type', labels=['Normal', 'Anomaly'], bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Add Smart Labels
for c in ax.containers:
    if c.get_label() == 'True' or c.get_label() == True:
        labels = [f'{w:.1f}%' if w > 0 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=14, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('entity_country_anomaly_percentage_stacked.png')

# --- Anomaly Count by Day of Week ---
plt.figure(figsize=(10, 5))
# Filter specifically for Anomalies Only
anomaly_day_counts = df_corrected[df_corrected['Is_Anomaly']]['DayOfWeek'].value_counts().sort_index()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
colors_day = [color_anomaly if d < 5 else color_weekend for d in range(7)] # Pink for week, Yellow/Orange for weekend
# Reindex to ensure all days are present even if 0 count
anomaly_day_counts = anomaly_day_counts.reindex(range(7), fill_value=0)

ax = plt.bar(days, anomaly_day_counts.values, color=colors_day, edgecolor='black')
add_vertical_bar_labels(plt.gca())
plt.title('Temporal Risk: Anomaly Count by Day of Week', fontsize=14, fontweight='bold')
plt.ylabel('Number of Anomalies')
plt.tight_layout()
plt.savefig('anomaly_count_by_day.png')


