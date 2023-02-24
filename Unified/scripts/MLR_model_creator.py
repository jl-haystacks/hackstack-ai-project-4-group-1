# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import pickle

# Read CSV file for data
df = pd.read_csv('../data/raw/haystacks_ga_clean_new_format.csv').drop_duplicates().reset_index().drop(columns='index')

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

# Sort listings by zipcode in a series of objects indexing by zipcode

zip_dfs = pd.Series([], dtype='O')
for zipcode in sorted(set(df.zipcode.values)):
    zip_dfs[zipcode] = (df.loc[df.zipcode==zipcode,:], zipcode)

zip_dfs['Georgia'] = (df.copy(), 'Georgia')

for county in sorted(set(df.county.values)):
    zip_dfs[county] = (df.loc[df.county == county, :], county)

house_features = ['square_footage', 'beds', 'lot_size', 'baths_full', 'baths_half'] 
regional_features = ['overall_crime_grade', 'property_crime_grade', 'ES_rating', 'MS_rating', 'HS_rating']
features = house_features+regional_features

###################################################################################
# MLR with all features
###################################################################################

models = []
shaps = []
limes = []
X = df[features]
y = df.price

# Split models into test and train data
X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            train_size = 0.8,
            random_state = 0)
mod = LinearRegression().fit(X_train, y_train)

# Populate linear model, SHAP arrays, and LIME
for zdf in zip_dfs:
    models.append((mod, zdf[1]))
    explainer = shap.Explainer(models[-1][0], zdf[0][features])
    shap_values = explainer(zdf[0][features])
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

lime_ser = pd.Series([], dtype = 'O')
for i in range(len(limes)):
    lime_ser[limes[i][1]] = limes[i][0]
lime_ser = lime_ser.to_frame().reset_index()
lime_ser.columns = ['zipcode', 'lime']

# Merge series into one dataframe
MLR_full_df = mod_ser.merge(sv_ser, on='zipcode')
MLR_full_df= MLR_full_df.merge(exp_ser, on='zipcode')
MLR_full_df.index = MLR_full_df.zipcode

###################################################################################
########### MLR with house features only
###################################################################################

models = []
shaps = []
limes = []
X = df[house_features]
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
    explainer = shap.Explainer(models[-1][0], zdf[0][house_features])
    shap_values = explainer(zdf[0][house_features])
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
MLR_house_df = mod_ser.merge(sv_ser, on='zipcode')
MLR_house_df = MLR_house_df.merge(exp_ser, on='zipcode')
MLR_house_df.index = MLR_house_df.zipcode

###################################################################################
############## MLR with regional features only
################################################################################### 
models = []
shaps = []
X = df[regional_features]
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
    explainer = shap.Explainer(models[-1][0], zdf[0][regional_features])
    shap_values = explainer(zdf[0][regional_features])
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
MLR_regional_df = mod_ser.merge(sv_ser, on='zipcode')
MLR_regional_df = MLR_regional_df.merge(exp_ser, on='zipcode')
MLR_regional_df.index = MLR_regional_df.zipcode


###################################################################################
## Random Forest with all features
###################################################################################

#models = []
#shaps = []
#X = df[features]
#y = df.price

# Split models into test and train data

#X_train, X_test, y_train, y_test = train_test_split(
#            X, 
#            y,
#            train_size = 0.8,
#            random_state = 0)
#RF = RandomForestRegressor(n_jobs = -1, random_state = 0)
#mod = RF.fit(X_train, y_train)

# Populate linear model and SHAP arrays 
#for zdf in zip_dfs:
#    models.append((mod, zdf[1]))
#    explainer = shap.TreeExplainer(models[-1][0])
#    shap_values = explainer(zdf[0][features], check_additivity = False)
#    shaps.append([explainer, shap_values, zdf[1]])

# Generate series for each indexed by zipcode
#mod_ser = pd.Series([], dtype='O')
#for i in range(len(models)):
#    mod_ser[models[i][1]] = models[i][0]
#mod_ser = mod_ser.to_frame().reset_index()
#mod_ser.columns = ['zipcode','model']
    
#sv_ser = pd.Series([], dtype = 'O')
#for i in range(len(shaps)):
#    sv_ser[shaps[i][2]] = shaps[i][1]
#sv_ser = sv_ser.to_frame().reset_index()
#sv_ser.columns = ['zipcode', 'shap_values']

#exp_ser = pd.Series([], dtype = 'O')
#for i in range(len(shaps)):
#    exp_ser[shaps[i][2]] = shaps[i][0]
#exp_ser = exp_ser.to_frame().reset_index()
#exp_ser.columns = ['zipcode','explainer']

# Merge series into one dataframe
##RF_full_df = RF_full_df.merge(exp_ser, on='zipcode')
#RF_full_df.index = RF_full_df.zipcode

# Merge all into a final series to save as pickle
Final_Series = pd.Series([], dtype='O')
Final_Series['MLR_full'] = MLR_full_df
Final_Series['MLR_house'] = MLR_house_df
Final_Series['MLR_regional'] = MLR_regional_df
#Final_Series['RF_full'] = RF_full_df

# Save pickle file

filename = '../data/pickle/modNshap.P'
pickle.dump(Final_Series, open(filename, 'wb'))


### corresponding dataframes

all_dfs = pd.Series([], dtype='O')
for zipcode in set(df.zipcode.values):
    all_dfs[zipcode] = df.loc[df.zipcode == zipcode]
all_dfs['Georgia'] = df.copy()
for county in set(df.county.values):
    all_dfs[county] = df.loc[df.county == county]

for zdf in all_dfs:
    for model in Final_Series.index:
        features = Final_Series[model].loc[30002, 'model'].feature_names_in_
        zdf[model+'_price'] = Final_Series[model].loc[30002, 'model'].predict(zdf[features])
        zdf[model+'_caprate'] = 100*12*(zdf['rent'])/(zdf[model+'_price'])-1
        zdf[model+'_score'] = Final_Series[model].loc[30002, 'model'].score(zdf[features], zdf['price'])
        zdf[model+'_differential'] = zdf[model+'_price']-zdf['price']

## saving all dataframes

filename = '../data/pickle/preloading.P'
pickle.dump(all_dfs, open(filename, 'wb'))