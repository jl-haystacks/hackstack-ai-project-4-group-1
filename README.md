# Haystacks Project 4 Group 1
# Quantitative Explainability Solution

In order to run this, some csvs are necessary. Place them in Unified/data/raw. 
The data will not be provided. The required files are
crime_rating_zipcode.csv,

elementary_schools.csv,

fmrClean.csv, 

GA_LISTINGS_SALES_V2.csv,

high_schools.csv,

middle_schools.csv

############ Geolocation

The geoJSON files below should be placed in Unified/data/geojson 

### Counties Georgia 2010 (Princeton University Library):

https://maps.princeton.edu/catalog/harvard-tg00gazcta

Save as harvard-tg00gazcta-geojson.json

### UA Census Zip Code Tabulation Areas, 2010 - Georgia (ibid.):

https://maps.princeton.edu/catalog/tufts-gacounties10

Save as tufts-gacounties10-geojson.json

############ Preprocessing and preparing

We now must run some scripts. Go to Unified/scripts/ and run the following two functions in the respective order.

preprocessing.py ## Prepares dataframe

MLR_modNshap.P ## template models

############ Start app

Run index.py in Unified
