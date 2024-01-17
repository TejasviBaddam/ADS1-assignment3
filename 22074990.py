
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Co2.csv'

co2_data = pd.read_csv(file_path, skiprows=4)

# Data Cleaning ... Dropping columns from 1960 to 1989
co2_data_cleaned = co2_data.drop(co2_data.columns[4:34], axis=1)  
co2_data_cleaned = co2_data_cleaned.drop(['Country Code', 'Indicator Name', 
                                          'Indicator Code', 'Unnamed: 67'], 
                                         axis=1)

# Imputing missing values with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
co2_data_imputed = imputer.fit_transform(co2_data_cleaned.iloc[:, 1:])

# Normalizing the data
scaler = StandardScaler()
co2_data_normalized = scaler.fit_transform(co2_data_imputed)

# Applying k-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(co2_data_normalized)

# Adding cluster labels to the dataframe
co2_data_cleaned['Cluster'] = kmeans.labels_

year_to_visualize = '2020'
visualization_df = co2_data_cleaned[['Country Name', year_to_visualize, 
                                     'Cluster']].dropna()

# Plotting the clustering results
plt.figure(figsize=(15, 8))
sns.scatterplot(x=visualization_df['Country Name'], 
                y=visualization_df[year_to_visualize], 
                hue=visualization_df['Cluster'], palette='viridis')
plt.xticks(rotation=90)
plt.title(f'Clustering of CO2 Emissions in {year_to_visualize}')
plt.ylabel('CO2 Emissions (kg per 2017 PPP $ of GDP)')
plt.xlabel('Country')
plt.legend(title='Cluster')
plt.show()

# Extracting India's CO2 emission data
india_co2_data = co2_data_cleaned[co2_data_cleaned['Country Name'] == \
                                  'India'].iloc[:, 1:-1].T
india_co2_data.columns = ['CO2 Emissions']
india_co2_data['Year'] = india_co2_data.index.astype(int)
india_co2_data.dropna(inplace=True)

# Function for a linear model
def linear_model(x, a, b):
    return a * x + b

# Preparing data for curve fitting
xdata = india_co2_data['Year']
ydata = india_co2_data['CO2 Emissions']

# Predictions for the next 10 and 20 years
last_year = xdata.iloc[-1]
future_years = np.arange(last_year + 1, last_year + 21)

# Curve fitting with the linear model
popt, pcov = curve_fit(linear_model, xdata, ydata)

# Making predictions
predictions = linear_model(future_years, *popt)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(xdata, ydata, label='Historical Data')
plt.plot(future_years, predictions, label='Predicted Data', color='red')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kg per 2017 PPP $ of GDP)')
plt.title('CO2 Emissions Forecast for India ')
plt.legend()
plt.show()

# Selecting countries
countries_cluster_0 = ['India', 'Brazil', 'Indonesia']
countries_cluster_1 = ['United States', 'Germany', 'Japan']

# Extracting data for these countries
co2_cluster_0 = co2_data_cleaned[co2_data_cleaned['Country Name'].
                                 isin(countries_cluster_0)]
co2_cluster_1 = co2_data_cleaned[co2_data_cleaned['Country Name'].
                                 isin(countries_cluster_1)]

# Melting the data for easier plotting
co2_cluster_0_melted = co2_cluster_0.melt(id_vars=['Country Name'], 
                                   var_name='Year', value_name='CO2 Emissions')
co2_cluster_1_melted = co2_cluster_1.melt(id_vars=['Country Name'], 
                                   var_name='Year', value_name='CO2 Emissions')

# Plotting CO2 emissions trends for these countries
plt.figure(figsize=(15, 8))
sns.lineplot(data=co2_cluster_0_melted, x='Year', y='CO2 Emissions', 
             hue='Country Name', marker='o', label='Cluster 0')
sns.lineplot(data=co2_cluster_1_melted, x='Year', y='CO2 Emissions', 
             hue='Country Name', marker='o', label='Cluster 1', linestyle='--')
plt.xticks(rotation=45)
plt.title('CO2 Emissions Trends: Cluster 0 vs. Cluster 1')
plt.ylabel('CO2 Emissions (kg per 2017 PPP $ of GDP)')
plt.xlabel('Year')
plt.legend()
plt.show()