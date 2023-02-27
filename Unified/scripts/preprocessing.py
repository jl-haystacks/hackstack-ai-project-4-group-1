import pandas as pd
import numpy as np
import pickle
import re

## Major raw df  
GAdf = pd.read_csv('../data/raw/GA_LISTINGS_SALES_V2.csv')

## Only consider detatched homes (not multi-family or lots). Houses need bedrooms.
GAdf = GAdf[GAdf.details.str.contains("etached")]
GAdf['baths_half'].fillna(0, inplace = True)
GAdf['baths_full'].fillna(0, inplace=True)
GAdf['beds'].dropna(inplace=True)

# Rename zipcode, drop unnecessary column
GAdf.rename(columns = {'zip':'zipcode'}, inplace=True)
GAdf.drop('Unnamed: 0', inplace=True, axis=1)

## Acres was given as square footage instead   
GAdf.loc[31032, 'details'] = 'Detached, 3 Beds, 2½ Baths, 1 Acres'

## Take away CA zipcode
GAdf=GAdf[GAdf.zipcode!='92544'].copy()

## abz (average by zip) and the following functions will be used to impute values 
abz = GAdf.groupby('zipcode').agg({'square_footage':'mean', 'lot_size':'mean'}).reset_index()

## Sets sqft to average within zipcode
def fix_sqft(df, zipcode, abz):
    if zipcode in list(df.zipcode.values) and \
     abz.loc[abz.zipcode == zipcode, 'square_footage'].isnull().values[0] == False:
        return abz.loc[abz.zipcode==zipcode, 'square_footage'].values[0]
    else: pass

## Sets lotsize to average within zipcode    
def fix_lotsize(df, zipcode, abz):    
    if zipcode in list(abz.zipcode.values) and \
    abz.loc[abz.zipcode == zipcode, 'lot_size'].isnull().values[0] == False:
        return abz.loc[abz.zipcode==zipcode, 'lot_size'].values[0]
    else: pass

## Only consider null observations to not overwrite non-null observations\
sqftnull = GAdf.loc[GAdf['square_footage'].isnull()].copy()
sqftnull['square_footage']= sqftnull.apply(lambda row: fix_sqft(GAdf, row['zipcode'], abz), axis=1)
for row in sqftnull.index:
    GAdf.loc[row, 'square_footage'] = sqftnull.loc[row, 'square_footage']
    
lotnull = GAdf.loc[GAdf['lot_size'].isnull()].copy()
lotnull['lot_size'] = lotnull.apply(lambda row: fix_lotsize(GAdf, row['zipcode'], abz), axis =1)
for row in lotnull.index:
    GAdf.loc[row, 'lot_size'] = lotnull.loc[row, 'lot_size']

## Feature engineering (from other datasets)
GAdf['zipcode'] = pd.to_numeric(GAdf['zipcode'])

## For school ratings
hsdf = pd.read_csv('../data/raw/high_schools.csv')
msdf = pd.read_csv('../data/raw/middle_schools.csv')
esdf = pd.read_csv('../data/raw/elementary_schools.csv')

hsdf.rename(columns = {'rating': 'HS_rating'}, inplace=True)
msdf.rename(columns = {'rating': 'MS_rating'}, inplace=True)
esdf.rename(columns = {'rating': 'ES_rating'}, inplace=True)

hsdf = hsdf.groupby(['zipcode']).agg({'HS_rating':'mean'}).reset_index()
msdf = msdf.groupby(['zipcode']).agg({'MS_rating':'mean'}).reset_index()
esdf = esdf.groupby(['zipcode']).agg({'ES_rating':'mean'}).reset_index()

## For crime statistics
crime = pd.read_csv('../data/raw/crime_rating_zipcode.csv')
crime.rename(columns={"census_zcta5_geoid": 'zipcode'}, inplace=True)
crime = crime[['overall_crime_grade', 'property_crime_grade','zipcode']]

## Alphabetic scores to numerals
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
crime['overall_crime_grade'] = crime['overall_crime_grade'].apply(lambda row: mask[row])
crime['property_crime_grade'] =crime['property_crime_grade'].apply(lambda row: mask[row])

df = GAdf.merge(crime, on = 'zipcode', how= 'inner')
df = df.merge(hsdf, on = 'zipcode', how= 'inner')
df = df.merge(msdf, on = 'zipcode', how= 'inner')
df = df.merge(esdf, on = 'zipcode', how= 'inner')
df = df.drop('unit_count', axis=1)

df['HS_rating'].fillna(np.mean(df.HS_rating), inplace=True)
df['MS_rating'].fillna(np.mean(df.MS_rating), inplace=True)
df['ES_rating'].fillna(np.mean(df.ES_rating), inplace=True)

df.dropna(subset = ['beds', 'lot_size', 'square_footage'], inplace=True)
df['rent'] = 0

## Use to estimate rents at a county level. FMR values (fair market rents) provide 
## a 40th percentile rent of a standard quality unit within a given county 
fmr = pd.read_csv('../data/raw/fmrClean.csv')

## Construct rent values from fmr
def rents(beds, fmr, county):
    if beds <=4:
        return fmr.loc[fmr.county_name==county, 'fmr_'+str(int(beds))].values[0]
    else: return fmr.loc[fmr.county_name==county, 'fmr_4'].values[0]*(1.15**(beds-4)) ## by definition

df['rent']=df.apply(lambda row: rents(row['beds'], fmr, row['county_name']), axis=1)
df = df.drop(index = 2446) ## Certainly not single-family 
df = df.loc[~df.year_built.isnull()]
df = df.drop_duplicates(subset = ['full_street_address', 'city'])

## Shortening column names
df = df.drop('census_county_name', axis=1)
df = df.rename(columns ={'full_street_address': 'address',
                         'county_name': 'county', 
                          'census_state_name': 'state'})
df['county'] = df['county'].apply(lambda x: x.replace('-County', ''))

# A function found here for reordering columns and also dropping them if necessary:
# https://stackoverflow.com/questions/35321812/move-column-in-pandas-dataframe
def reorder_columns(columns, first_cols=[], last_cols=[], drop_cols=[]):
    columns = list(set(columns) - set(first_cols))
    columns = list(set(columns) - set(drop_cols))
    columns = list(set(columns) - set(last_cols))
    new_order = first_cols + columns + last_cols
    return new_order


# Now execute the function above
my_list = df.columns.tolist()
location_data = ['latitude', 'longitude', 'address', 'city', 'county', 'state', 'zipcode']
reordered_cols = reorder_columns(my_list, first_cols=location_data)
df = df[reordered_cols]
df.reset_index().drop(columns='index')

df.to_csv('../data/raw/final.csv')