import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import callback_context
import pandas as pd
import numpy as np
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objs as go
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')
import io
import base64
import matplotlib.pyplot as plt
from logzero import logger
# from dash import dash_table
# from dash.dash_table.Format import Format, Group
# from dash.dash_table.FormatTemplate import FormatTemplate
from datetime import datetime as dt
from app import app
import scripts.utils_haystacks as f
import scripts.create_ga_fig as g
import json

####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### Corporate css formatting
corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)'
}

externalgraph_rowstyling = {
    'margin-left' : '15px',
    'margin-right' : '15px'
}

externalgraph_colstyling = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['superdark-green'],
    'background-color' : corporate_colors['superdark-green'],
    'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['light-green'],
    'background-color' : corporate_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }

navbarcurrentpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : corporate_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
    }

recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

recapdiv_text = {
    'text-align' : 'left',
    'font-weight' : '350',
    'color' : corporate_colors['white'],
    'font-size' : '1.5rem',
    'letter-spacing' : '0.04em'
    }

####################### Corporate chart formatting

corporate_title = {
    'font' : {
        'size' : 16,
        'color' : corporate_colors['white']}
}

corporate_xaxis = {
    'showgrid' : False,
    'linecolor' : corporate_colors['light-grey'],
    'color' : corporate_colors['light-grey'],
    'tickangle' : 315,
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_yaxis = {
    'showgrid' : True,
    'color' : corporate_colors['light-grey'],
    'gridwidth' : 0.5,
    'gridcolor' : corporate_colors['dark-green'],
    'linecolor' : corporate_colors['light-grey'],
    'titlefont' : {
        'size' : 12,
        'color' : corporate_colors['light-grey']},
    'tickfont' : {
        'size' : 11,
        'color' : corporate_colors['light-grey']},
    'zeroline': False
}

corporate_font_family = 'Dosis'

corporate_legend = {
    'orientation' : 'h',
    'yanchor' : 'bottom',
    'y' : 1.01,
    'xanchor' : 'right',
    'x' : 1.05,
	'font' : {'size' : 9, 'color' : corporate_colors['light-grey']}
} # Legend will be on the top right, above the graph, horizontally

corporate_margins = {'l' : 5, 'r' : 5, 't' : 45, 'b' : 15}  # Set top margin to in case there is a legend

corporate_layout = go.Layout(
    font = {'family' : corporate_font_family},
    title = corporate_title,
    title_x = 0.5, # Align chart title to center
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor = 'rgba(0,0,0,0)',
    xaxis = corporate_xaxis,
    yaxis = corporate_yaxis,
    height = 270,
    legend = corporate_legend,
    margin = corporate_margins
    )

####################################################################################################
# 000 - DATA MAPPING
####################################################################################################

########################### Page 1 - Maps
# Load pre computed data
# ga = f.load_pickle('ga_info.p')

# Deployment inforamtion
PORT = 8050

# Load necessary information for accessing source files
mapbox_access_token = f.config['mapbox']['token']
raw_dataset_path = f.RAW_PATH + f.config['path']['name']


####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

########################### Page 1 - Maps
# Import map and bar chart data
df_raw = pd.read_csv(raw_dataset_path)

# Create DataFrames for map and bar chart
#df_ga = g.process_ga_data(df_raw)

# Prepare figure
#fig_ga = g.create_ga_fig(df_ga, mapbox_access_token=mapbox_access_token)

#Create L1 dropdown options

repo_groups_l1_all = [
    {'label' : 'Number of Listings', 'value' : 'count'},
    {'label' : 'Average Listing Price', 'value' : 'avg_listing_price'},
    {'label': 'Maximum Listing Price', 'value': 'max_listing_price'},
    {'label' : 'Average Estimated Price', 'value' : 'avg_price'},
    {'label' : 'Average Estimated Price', 'value' : 'max_price'},
    {'label': 'Average Amount Undervalued According to Model', 'value': 'avg_differential'},
    {'label': 'Maximum Amount a Listing is Undervalued According to Model', 'value': 'max_differential'},
    {'label' : 'Model Score Over Region', 'value' : 'max_score'},
    ]
########################### Page 1 - Maps

########################### Page 2 - Analytics
# Read in external data into dash application
df = df_raw
df1 = df.copy()

# Read pickle file that can be obtained by running the first half or so of MLR.ipynb
MS_ser = pd.read_pickle('data/pickle/modNshap.P')

################################# Need to integrate below mask into pre-processing and take it out of here

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
#################################
all_dfs = pd.read_pickle('data/pickle/preloading.P')
# Series of zipcode data frames. May want to save elsewhere.

#all_dfs = pd.Series([], dtype='O')
#for zipcode in set(df.zipcode.values):
#    all_dfs[zipcode] = df.loc[df.zipcode == zipcode]
#all_dfs['Georgia'] = df.copy()
#for county in set(df.county.values):
#    all_dfs[county] = df.loc[df.county == county]

## Attach model predictions and scores to all_dfs dataframes.

#for zdf in all_dfs:
#    for model in MS_ser.index:
#        features = MS_ser[model].loc[30002, 'model'].feature_names_in_
#        zdf[model+'_price'] = MS_ser[model].loc[30002, 'model'].predict(zdf[features])
#        zdf[model+'_caprate'] = 100*12*(zdf['rent'])/(zdf[model+'_price'])-1
#        zdf[model+'_score'] = MS_ser[model].loc[30002, 'model'].score(zdf[features], zdf['price'])
#        zdf[model+'_differential'] = zdf[model+'_price']-zdf['price']

# https://maps.princeton.edu/catalog/tufts-gacounties10
with open('data/geojson/tufts-gacounties10-geojson.json') as json_data:
    map_ga_counties = json.load(json_data)

    # Load in zipcode-level geoJSON
with open('data/geojson/harvard-tg00gazcta-geojson.json') as json_data:
    map_ga_zipcodes = json.load(json_data)

# Define features to be utilized for generating scatter plots

house_features = ['square_footage', 'beds', 'lot_size', 'baths_full', 'baths_half'] 
regional_features = ['overall_crime_grade', 'property_crime_grade', 'ES_rating', 'MS_rating', 'HS_rating']
features = house_features+regional_features


# features = ['square_footage', 'overall_crime_grade', 'ES_rating', 'lot_size', 'baths_half', 
# 'MS_rating', 'HS_rating', 'beds', 'baths_full', 'year_built', 'property_crime_grade']

# Features to be omitted from 'features':
# omitted_features = ['listing_special_features', 'listing_status', 'transaction_type']

# Define sales prices based on models to be utilized for generating scatter plots
models = ['price']

# Calculate averages of selected features
df_average = df[features].mean()

# Calculate maximum value of entire dataframe to set limit of point plot
max_val = df.max()

# Load models indexed by zipcode
# Includes a model, a shap_value object, and the corresponding explainer

crossfilter_model_options = [
    {'label': 'Multiple Linear Regression, all features', 'value': 'MLR_full'},
    {'label': 'Multiple Linear Regression house features', 'value': 'MLR_house'},
    {'label': 'Multiple Linear Regression: regional features', 'value': 'MLR_regional'}
    ]

# Resolution group L2 options
crossfilter_resolution_options = [
    {'label': 'State', 'value': 'state'},
    # {'label': 'County', 'value': 'county'},
    {'label': 'Zip code', 'value': 'zipcode'},
    ]

### Holds model-specific statistics, should probably pre-load

model_stats = pd.Series([], dtype='O')
for model in MS_ser.index:
    model_stats[model] = pd.DataFrame() 
    for focus in all_dfs.index:
        X = pd.DataFrame(all_dfs[focus].agg(
            {model+'_price': ['mean', 'max'], 
             model+'_caprate': ['mean', 'max'],
             model+'_differential': ['mean', 'max'],
             model+'_score':['mean', 'max'], ## These are the same, but can't rename without
            'price': ['mean', 'max'],
            }, axis=0
        ).unstack(level=1)).T
        X.columns = ['avg_price', 'max_price', 'avg_caprate', 'max_caprate', 'avg_differential', 'max_differential', 'avg_score', 'max_score', 'avg_listing_price', 'max_listing_price']
        X.index = pd.Index([focus])
        model_stats[model] = model_stats[model].append(X)
    for idx in model_stats[model].index:
        temp = df.loc[((idx == df.zipcode) |(idx == df.county)) | (idx == 'Georgia')]
        model_stats[model].loc[idx,'count'] = len(temp)

reg_idx=pd.Series(dtype='O')  ## Masks above to only have zipcodes
for mod in model_stats.index:
    reg_idx[mod] = pd.Series(model_stats[mod].index).astype('str').str.isnumeric()
    reg_idx[mod].index = model_stats[mod].index

zip_idx = pd.Series(model_stats[model].index).astype('str').str.isnumeric()
zip_idx.index = model_stats[model].index

count_idx = ~pd.Series(model_stats[model].index).astype('str').str.isnumeric()
count_idx.index = model_stats[model].index


####################################################################################################
# 000 - DEFINE ADDITIONAL FUNCTIONS
####################################################################################################

def group_wavg(df, gr_by_cols, weight, value):
    """This function returns a df grouped by the gr_by_cols and calculate the weighted avg based
    on the entries in the weight and value lists"""
    # Calculate weight * value columns
    wcols = []
    cols = []
    for i in range(0,len(value),1):
        wcol = "w"+value[i]
        wcols.append(wcol)
        df[wcol] = df[weight[i]] * df[value[i]]
    # Group by summing the wcols and weight columns
    cols = weight
    for i in wcols:
        cols.append(i)
    df1 = df.groupby(gr_by_cols)[cols].agg('sum')
    df1.reset_index(inplace=True)
    # Divide wcols by weight and remove columns
    for i in range(0,len(value),1):
        df1[value[i]] = df1[wcols[i]] / df1[weight[i]]
        df1.drop(wcols[i], axis='columns', inplace=True)

    return df1

def colorscale_generator(n, starting_col = {'r' : 186, 'g' : 218, 'b' : 212}, finish_col = {'r' : 57, 'g' : 81, 'b' : 85}):
    """This function generate a colorscale between two given rgb extremes, for an amount of data points
    The rgb should be specified as dictionaries"""
    r = starting_col['r']
    g = starting_col['g']
    b = starting_col['b']
    rf = finish_col['r']
    gf = finish_col['g']
    bf = finish_col['b']
    ri = (rf - r) / n
    gi = (gf - g) / n
    bi = (bf - b) / n
    color_i = 'rgb(' + str(r) +','+ str(g) +',' + str(b) + ')'
    my_colorscale = []
    my_colorscale.append(color_i)
    for i in range(n):
        r = r + ri
        g = g + gi
        b = b + bi
        color = 'rgb(' + str(round(r)) +','+ str(round(g)) +',' + str(round(b)) + ')'
        my_colorscale.append(color)

    return my_colorscale

# Create a corporate colorcale
colors = colorscale_generator(n=11)

corporate_colorscale = [
    [0.0, colors[0]],
    [0.1, colors[1]],
    [0.2, colors[2]],
    [0.3, colors[3]],
    [0.4, colors[4]],
    [0.5, colors[5]],
    [0.6, colors[6]],
    [0.7, colors[7]],
    [0.8, colors[8]],
    [0.9, colors[9]],
    [1.0, colors[10]]]

####################################################################################################
####################################################################################################
####################################################################################################
# MAP PAGE
####################################################################################################
####################################################################################################
####################################################################################################

####################################################################################################
# 00A - BAR CHART AND CHOROPLETH MAP UPDATE
####################################################################################################
repo_groups_l1_all = [
    {'count': 'Number of Listings'},
    {'avg_listing_price': 'Average Listing Price'},
    {'max_listing_price': 'Maximum Listing Price' },
    {'avg_price': 'Average Estimated Price'},
    {'max_price': 'Average Estimated Price'},
    {'avg_differential' : 'Average Amount Undervalued'},
    {'max_differential': 'Maximum Amount a Listing is Undervalued '},
    {'max_score': 'Model score (R-squred)'},
    ]

################### Callback for producing a map

@app.callback(
   dash.dependencies.Output('ga-map', 'figure'),
   [
       dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
       dash.dependencies.Input('which_json', 'value'),
       dash.dependencies.Input('map-model', 'value')
   ]
)
################### Changes map on Map page

def update_map(coloring, reso, model):
    if reso == 'zipcode':
        fig = px.choropleth_mapbox(
            model_stats[model][zip_idx].reset_index(),
            geojson=map_ga_zipcodes,
            locations= 'index',
            color=coloring,
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            featureidkey = 'properties.ZCTA', 
            zoom=5,
            center = {"lat": 32.6461, "lon": -83.4317},
            opacity=0.5,
            labels={
                'index': 'Zip code',
                'count': 'Number of Listings',
                'avg_listing_price': 'Average Listing Price',
                'max_listing_price': 'Maximum Listing Price' ,
                'avg_price': 'Average Estimated Price',
                'max_price': 'Average Estimated Price',
                'avg_differential' : 'Average Amount Undervalued',
                'max_differential': 'Maximum Amount a Listing is Undervalued',
                'max_score': 'Model score (R-squred)'
                },
            )
    elif reso == 'county':
        fig = px.choropleth_mapbox(
            model_stats[model][count_idx].reset_index(),
            geojson=map_ga_counties,
            locations= 'index',
            color=coloring,
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            featureidkey = 'properties.name10', 
            zoom=5,
            center = {"lat": 32.6461, "lon": -83.4317},
            opacity=0.5,
            labels={
                'index': 'County',
                'count': 'Number of Listings',
                'avg_listing_price': 'Average Listing Price',
                'max_listing_price': 'Maximum Listing Price' ,
                'avg_price': 'Average Estimated Price',
                'max_price': 'Average Estimated Price',
                'avg_differential' : 'Average Amount Undervalued',
                'max_differential': 'Maximum Amount a Listing is Undervalued',
                'max_score': 'Model score (R-squred)'
                },
            )
    fig.update_layout(
        coloraxis_showscale=True,
        legend_title_text='Spend',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
    )
    return fig


####################################################################################################
# 00B - SCATTER PLOT UPDATE
####################################################################################################

################### Callback for updating features according to model

@app.callback(
    dash.dependencies.Output('feat_list', 'children'),
    dash.dependencies.Input('crossfilter-model', 'value')
)

################### Function updates features dropdown according to model

def update_feature_list(which_model):
    X = dcc.Dropdown(id = 'crossfilter-feature',
    options = [{'label': i, 'value': i} for i in MS_ser[which_model].loc[30002,'model'].feature_names_in_],
    # Default value when loading
    value = MS_ser[which_model].loc[30002,'model'].feature_names_in_[0],
    # Permit user to select only one option at a time
    multi = False,
    style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
    )
    return X

################### Callback for dynamic dropdown of zipcodes/counties

@app.callback(
    dash.dependencies.Output('reso_list', 'children'),
    dash.dependencies.Input('crossfilter-resolution', 'value')
)
################### Create dynamic dropdown. Effects change in scatter-plot as well.

def update_scale(resolution_dropdown):
    new_dropdown = dcc.Dropdown(
    list(set(df[resolution_dropdown])),
    id='filter-dropdown',
        value=df[resolution_dropdown].values[0]
    )
    return new_dropdown

################### First callback for SHAP values

@app.callback(
    dash.dependencies.Output('shap-bee', 'src'),
    [
        dash.dependencies.Input('filter-dropdown','value'),
        #dash.dependencies.State('crossfilter-resolution', 'value') ## Useful if we add county as well\

        dash.dependencies.Input('crossfilter-model', 'value'),    ## need to figure out a good way to do this. Will probably use a Series of those model dataframes
    ]
)
################### Function to update SHAP values

def update_shap(focus, which_model):
    plt.style.use("dark_background")
    shap.summary_plot( 
        MS_ser[which_model].loc[focus,'shap_values'], 
        show=False)
    fig = plt.gcf()
    plt.tick_params(colors = 'white')
    plt.ticklabel_format(axis='x', scilimits=[-3, 3])
    buf = io.BytesIO() # in-memory files
    plt.savefig(buf, format = "png", transparent = True)
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()
    X = "data:image/png;base64,{}".format(data)
    return X

################### First callback to coordinate scatterplot with feature, model, and color gradient schema

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature', 'value'),
        dash.dependencies.Input('crossfilter-model', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value'),
        dash.dependencies.Input('crossfilter-resolution', 'value'),
        dash.dependencies.Input('filter-dropdown','value'),
        dash.dependencies.Input('select-target', 'value')
    ]
)

################### Function to update graph in response to feature, model, and color gradient schema being updated

def update_graph(feature, model, gradient, resolution, focus, target):
    # Define points and respective colors of scatter plot
    cols = all_dfs[focus][feature].values #df[feature].values
    hover_names = []
    for ix, val in zip(all_dfs[focus].index.values, all_dfs[focus][feature].values):
        hover_names.append(f'Listing {ix:03d}<br>{feature} value of {val}')
    # Generate scatter plot
    fig = px.scatter(
        all_dfs[focus],
        x=all_dfs[focus][feature],
        y=all_dfs[focus][str(model)+'_'+str(target)],
        color=cols,
        opacity=0.8,
        hover_name=hover_names,
        hover_data= features,
        template='plotly_dark',
        color_continuous_scale=gradient,
        labels = {
            'HS_rating': 'High school rating',
            'MS_rating': 'Middle school rating',
            'ES_rating': 'Elementary school rating',
            'property_crime_grade': 'Property crime grade',
            'overall_crime_grade': 'Overall crime grade',
            'baths_half': 'Half bathrooms',
            'baths_full': 'Full bathrooms',
            'lot_size': 'Lot size',
            'square_footage': 'Square footage'
        }
    )
    if target == 'price':
        fig.update_layout(yaxis_title='Price')
    else:
        fig.update_layout(yaxis_title='Cap rate')
    # Layout of scatter plot
    fig.update_layout(
        coloraxis_colorbar={'title': f'{feature}'},
        coloraxis_showscale=True,
        legend_title_text='Spend',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
    )


    # Axis settings for scatter plot
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig

################### Function to create point plot of averages of selected features

def create_point_plot(df, title):

    fig = go.Figure(
        data=[
            go.Bar(name='Average', x=features, y=df_average.values, marker_color='#c178f6'),
            go.Bar(name=title, x=features, y=df.values, marker_color='#89efbd')
        ]
    )
    # Bar layout for point plot
    fig.update_layout(
        barmode='group',
        height=225,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
    )
    # Axis settings for point plot
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(type="log", range=[0,5])

    return fig

################### Callback to update point plot based on user hovering over points in scatter plot

@app.callback(
    dash.dependencies.Output('point-plot', 'figure'),
    [
        # Update graph on click
        dash.dependencies.Input('scatter-plot', 'clickData')
    ]
)

################### Function to trigger last function in response to user click point in scatter plot

def update_point_plot(clickData):
    index = clickData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)

################### Callback for accuracy value text

@app.callback(
    dash.dependencies.Output('model-accuracy-statement', 'children'),
    dash.dependencies.Input('acc-cutoff', 'value')
)

################### Changes accuracy value text

def display_accuracy(value):
    return html.H6(f'Minimum model R-squared: {value}', style = {'color': corporate_colors['superdark-green']})

@app.callback(
    dash.dependencies.Output('top-bars', 'figure'),
    [
        dash.dependencies.Input('acc-cutoff', 'value'),
        dash.dependencies.Input('choose-model','value'),
        dash.dependencies.Input('granularity', 'value'),
        dash.dependencies.Input('bar-options', 'value')
    ]
)

################### Updates  the top bar charts in Interpret page

def update_top_bars(cutoff, model, reso, selection):
    if reso == 'zipcode':
        allowed_regions = model_stats[model][(model_stats[model]['max_score'] >= cutoff) & (reg_idx[model])].sort_values(by='max_score', ascending = False)
    else:
        allowed_regions = model_stats[model][(model_stats[model]['max_score'] >= cutoff) & ~((reg_idx[model]) | (model_stats[model].index == 'Georgia'))].sort_values(by='max_score', ascending = False)
    top5 = allowed_regions.sort_values(by=selection, ascending = False).iloc[0:5].reset_index() 
    if reso == 'zipcode':
        where = 'Zip code'
    else: where = 'County'
    fig = px.bar(
        top5,
        x='index',
        y=selection,
        hover_data = ['count', 'avg_score'],
        opacity=0.8,
        labels={
            'index': where,
            'count': 'Number of listings',
            'avg_listing_price': 'Average listing price',
            'max_listing_price': 'Maximum listing price' ,
            'avg_price': 'Average estimated price',
            'max_price': 'Average estimated price',
            'avg_differential' : 'Average undervaluement',
            'max_differential': 'Maximum undervaluement',
            'max_score': 'Model score (R-squared)',
            'avg_score': 'Model score (R-squared)'
        },
    )
    fig.update_xaxes(type='category')

    fig.update_layout(
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
    )


    # Axis settings for scatter plot
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig

################### Callback for creating bar chart on bottom of Interpret page

@app.callback(
    dash.dependencies.Output('lower-bars', 'figure'),
    [
        dash.dependencies.Input('acc-cutoff', 'value'),
        dash.dependencies.Input('choose-model','value'),
        dash.dependencies.Input('granularity', 'value'),
        dash.dependencies.Input('bar-options', 'value'),
        dash.dependencies.Input('top-bars', 'clickData')
    ]
)
################### Creates lower bars from either click data or selected values

def lower_bars(cutoff, model, reso, selection, click): 
    if callback_context.triggered[0]['prop_id'] == 'top-bars.clickData':
        if reso == 'zipcode':
            top10  = all_dfs[int(click['points'][0]['x'])].sort_values(by = model +'_'+ selection[4:], ascending = False).iloc[0:10]
        else:
            top10 =  all_dfs[click['points'][0]['x']].sort_values(by = model +'_'+ selection[4:], ascending = False).iloc[0:10]
        top10 = top10.reset_index()
        fig = px.bar(
            top10,
            x= 'address',
            y = model+'_'+selection[4:],
            opacity = 0.8,
            hover_data = ['city', reso, 'index'],
            labels = {
                'MLR_full_price': 'Price',
                'MLR_regional_price': 'Price',
                'MLR_housing_price': 'Price',
                'MLR_full_differential': 'Undervalued by',
                'MLR_regional_differential': 'Undervalued by',
                'MLR_housing_differential': 'Undervalued by',
                'MLR_full_caprate': 'Cap rate',
                'MLR_regional_caprate': 'Cap Rate',
                'MLR_housing_caprate': 'Cap rate',
                'avg_score': 'Model score (R-squared)',
                'zipcode': 'Zip code',
                'city': 'City',
                'index': 'Listing number',
                'address': 'Address'
            },
        )
        fig.update_xaxes(type='category')

        fig.update_layout(

            height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},

            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
        )
        # Axis settings for scatter plot
        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)

        return fig
    elif reso == 'zipcode':
        allowed_regions = model_stats[model][(model_stats[model]['max_score'] >= cutoff) & (reg_idx[model])].sort_values(by='max_score', ascending = False)
        top10 = all_dfs['Georgia'][all_dfs['Georgia'][reso].isin(allowed_regions.index)].sort_values(by=model+'_'+selection[4:], ascending = False).iloc[0:10] 
    else:
        allowed_regions = model_stats[model][(model_stats[model]['max_score'] >= cutoff) & ~((reg_idx[model]) | (model_stats[model].index == 'Georgia'))].sort_values(by='max_score', ascending = False)
        top10 = all_dfs['Georgia'][all_dfs['Georgia'][reso].isin(allowed_regions.index)].sort_values(by=model+'_'+selection[4:], ascending = False).iloc[0:10] 
    top10 = top10.reset_index()
    fig = px.bar(
        top10,
        x='address', 
        y=model+'_'+selection[4:], #omit avg_ or max_
        opacity=0.8,
        hover_data = ['city', reso, 'index'],
        labels = {
            'MLR_full_price': 'Price',
            'MLR_regional_price': 'Price',
            'MLR_housing_price': 'Price',
            'MLR_full_differential': 'Undervalued by',
            'MLR_regional_differential': 'Undervalued by',
            'MLR_housing_differential': 'Undervalued by',
            'MLR_full_caprate': 'Cap rate',
            'MLR_regional_caprate': 'Cap Rate',
            'MLR_housing_caprate': 'Cap rate',
            'avg_score': 'Model score (R-squared)',
            'zipcode': 'Zip code',
            'city': 'City',
            'index': 'Listing number'
        },
    )
    fig.update_xaxes(type='category')

    fig.update_layout(
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},

        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
    )

    # Axis settings for scatter plot
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig


################### Generates SHAP graph

def generate_interpretation(model, lbcd):
    idx = int(lbcd['points'][0]['customdata'][2])
    row = MS_ser[model].loc['Georgia','shap_values'][idx].values
    fig = px.bar(
        x=row,
        y=MS_ser[model].loc['Georgia','model'].feature_names_in_, 
        opacity=0.8,
        labels = {
            'HS_rating': 'High school rating',
            'MS_rating': 'Middle school rating',
            'ES_rating': 'Elementary school rating',
            'property_crime_grade': 'Property crime grade',
            'overall_crime_grade': 'Overall crime grade',
            'baths_half': 'Half bathrooms',
            'baths_full': 'Full bathrooms',
            'lot_size': 'Lot size',
            'square_footage': 'Square footage',
            'MLR_full_price': 'Price',
            'MLR_regional_price': 'Price',
            'MLR_housing_price': 'Price',
            'MLR_full_caprate': 'Cap rate',
            'MLR_regional_caprate': 'Cap Rate',
            'MLR_housing_caprate': 'Cap rate',
            },
        color = row > 0,
        color_discrete_map={True: "red", False: "blue"},
    )
    fig.update_layout(
        margin={'l': 5, 'b': 5, 'r': 5, 't': 5},
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        xaxis_title= 'Marginal contribution of feature',
        yaxis_title= 'Feature',
        showlegend = False
    )
    fig.update_yaxes(type='category')
    return fig
    #elif inter == 'lime':
    #idx = int(lbcd['points'][0]['customdata'][2])
    #pass

################### Callback for waterfall plot

@app.callback(
    dash.dependencies.Output('limeshap', 'children'),
    [
        dash.dependencies.State('choose-model', 'value'),
        dash.dependencies.Input('lower-bars', 'clickData'),
        #dash.dependencies.State('interpretation-type', 'value'),
    ]
)

################### Updates SHAP waterfall based on clickdata 
    
def update_interpretation(which_model, lbcd):
    fig = generate_interpretation(which_model, lbcd)
    topmsg = 'Looking at address ' + lbcd['points'][0]['x']+', '+ lbcd['points'][0]['customdata'][0] +' '+ str(lbcd['points'][0]['customdata'][1])
    return [html.H6(topmsg), dcc.Graph(figure = fig)]