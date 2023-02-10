# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import shap
import pickle

# Read CSV file for data
df = pd.read_csv('../data/raw/haystacks_ga_clean_new_format.csv').drop_duplicates()

# Mask for turning crime grades into integers.
mask = {'F': 0,
       'D-': 1,
       'D': 2,
       'D+': 3,
       'C-': 4,
       'C': 5,
       'C+': 6,
       'B-': 7,
       'B': 8,
       'B+': 9,
       'A-': 10,
       'A': 11}
df['overall_crime_grade'] = df['overall_crime_grade'].apply(lambda row: mask[row])
df['property_crime_grade'] = df['property_crime_grade'].apply(lambda row: mask[row])
df = df.drop(['details', 
              'special_features',   
              'state',  
              'rent',
              'caprate',
              'address',
              'longitude',
              'latitude',
              'address',
              'city',
              'listing_special_features',
              'listing_status',
              'transaction_type'], axis=1)

# Sort listings by zipcode in a series of objects indexing by zipcode
zip_dfs = pd.Series([], dtype='O')
for zipcode in sorted(set(df.zipcode.values)):
    zip_dfs[zipcode] = (df.loc[df.zipcode==zipcode,:], zipcode)

# Create linear models over all of GA (later: Atlanta)
models = []
shaps = []
X = df.drop(columns = ['price', 'county', 'zipcode'])
y = df.price

# Split models into test and train data
X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            train_size = 0.8,
            random_state = 0)
mod = LinearRegression().fit(X_train, y_train)

# Populate linear model and SHAP arrays
for zdf in zip_dfs:
    models.append((mod, zdf[1]))
    explainer = shap.Explainer(models[-1][0], X)
    shap_values = explainer(zdf[0].drop(columns = ['price', 'county', 'zipcode']))
    shaps.append([explainer, shap_values, zdf[1]])

# Generate series for each indexed by zipcode
mod_ser = pd.Series([], dtype='O')
for i in range(len(models)):
    mod_ser[models[i][1]] = models[i][0]
mod_ser = mod_ser.to_frame().reset_index()
mod_ser.columns = ['zipcode','model']
    
sv_ser = pd.Series([], dtype = 'O')
for i in range(len(shaps)):
    sv_ser[shaps[i][2]] = shaps[i][1]
sv_ser = sv_ser.to_frame().reset_index()
sv_ser.columns = ['zipcode', 'shap_values']

exp_ser = pd.Series([], dtype = 'O')
for i in range(len(shaps)):
    exp_ser[shaps[i][2]] = shaps[i][0]
exp_ser = exp_ser.to_frame().reset_index()
exp_ser.columns = ['zipcode','explainer']

# Merge series into one dataframe
MS_df = mod_ser.merge(sv_ser, on='zipcode')
MS_df = MS_df.merge(exp_ser, on='zipcode')
MS_df.index = MS_df.zipcode

# This can be removed if desired.
MS_df = MS_df.drop(['zipcode'], axis=1)

# Save pickle file
filename = '../data/pickle/MLR_modNshap.P'
pickle.dump(MS_df, open(filename, 'wb'))