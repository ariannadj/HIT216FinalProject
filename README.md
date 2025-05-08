# HIT216FinalProject
# HIT216-FinalProject
Analyzing U.S. overdose death rates from 1999–2020 using CDC data. Includes data cleaning, visualizations, and a linear regression model to explore trends and disparities.

This project explores the death rates of drug overdoses in the U.S. I used data from the CDC.

This dataset allows insight into public health trends and disparities. I am exploring death rates over time, across different types of drugs, and among various demographic catergories like age and race.

Data Collection

The data comes from the CDC and includes age-adjusted overdose death rates by drug type, sex, age, race, and Hispanic origin.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Load the data using relative path

file_path = "Drug_overdose_death_rates__by_drug_type__sex__age__race__and_Hispanic_origin__United_States.csv"
df = pd.read_csv(file_path)

Display initial data

df.head()

Data Cleaning and Preprocessing

To prepare the dataset for analysis, I removed missing values from the Estimate column, converted YEAR to an integer, and filtered the data to focus on the total population for clarity in trend analysis.

Drop rows with missing estimates

print("Missing values before:", df['ESTIMATE'].isna().sum())
df = df.dropna(subset=['ESTIMATE'])
print("Missing values after:", df['ESTIMATE'].isna().sum())

Convert YEAR to integer

df['YEAR'] = df['YEAR'].astype(int)

Focus on Total population for cleaner visuals

df_total = df[df['STUB_NAME'] == 'Total']

Exploratory Data Analysis (EDA)

I added a snapshot of statistics on the dataset.

print(df_total['ESTIMATE'].describe())

The different models are as follows:

Trend over time by drug type - Line Plot

plt.figure(figsize=(12,6))
sns.lineplot(data=df_total, x='YEAR', y='ESTIMATE', hue='PANEL')
plt.title("Overdose Death Rates Over Time by Drug Type")
plt.ylabel("Death Rate per 100,000")
plt.xlabel("Year")
plt.legend(title='Drug Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

Average death rate by race/ethnicity - Bar Chart

df_race = df[df['AGE'] == 'All ages']
race_grouped = df_race.groupby('STUB_LABEL')['ESTIMATE'].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=race_grouped.values, y=race_grouped.index)
plt.title("Average Overdose Death Rate by Race/Ethnicity")
plt.xlabel("Average Death Rate")
plt.ylabel("Race/Ethnicity")
plt.show()

Modeling

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

Predicting death rate based on year

X = df_total[['YEAR']]
y = df_total['ESTIMATE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

Showing the Line Regression as a model

plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Linear Regression: Overdose Death Rate vs. Year")
plt.xlabel("Year")
plt.ylabel("Overdose Death Rate")
plt.legend()
plt.show()

How do average overdose death rates compare across different racial and ethnic groups in the United States from 1999 to 2020?
This question guides the analysis by focusing on demographic differences in overdose mortality. The data reveals that overdose death rates have steadily increased over time, with a particularly sharp rise linked to synthetic opioids in recent years. Clear disparities exist between racial and ethnic groups, with some populations experiencing significantly higher average death rates. These patterns emphasize the importance of targeted public health interventions and ongoing data monitoring to address the specific needs of the most affected communities.
