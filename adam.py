import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Datasets
main_df = pd.read_csv('Datathon Dataset.xlsx - Data - Main.csv').drop(['Assignment'], axis=1)
category_df = pd.read_csv('Datathon Dataset.xlsx - Others - Category Linkage.csv').rename(columns={'Category' : 'Category Flow'})
country_df = pd.read_csv('Datathon Dataset.xlsx - Others - Country Mapping.csv')

# 2. Drop unused columns and Merge
main_df = main_df.dropna(axis=1, how='all')
df = pd.merge(main_df, category_df, left_on='Category', right_on='Category Names', how='left')
df = pd.merge(df, country_df, left_on='Name', right_on='Code', how='left', suffixes=('', '_country'))

# 3. Date Preprocessing
df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
df['Year'] = df['Pstng Date'].dt.year
df['Month'] = df['Pstng Date'].dt.month
df['Day'] = df['Pstng Date'].dt.day
df['Weekday'] = df['Pstng Date'].dt.weekday

# 4. Encoding Categorical Variables
le = LabelEncoder()
cat_cols = ['Type', 'Category', 'Country']
for col in cat_cols:
    df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown').astype(str))

# Save result
df.to_csv('Preprocessed_Main_Data.csv', index=False)