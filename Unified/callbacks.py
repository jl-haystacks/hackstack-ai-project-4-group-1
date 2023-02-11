import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from jupyter_dash import JupyterDash
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
########################### Page 1 - Maps

########################### Page 2 - Analytics

########################### Page 2 - Analytics

###########################
#Sales mapping
# sales_filepath = 'data/datasource.xlsx'

# sales_fields = {
#     'date' : 'Date',
#     'reporting_group_l1' : 'Country',
#     'reporting_group_l2' : 'City',
#     'sales' : 'Sales Units',
#     'revenues' : 'Revenues',
#     'sales target' : 'Sales Targets',
#     'rev target' : 'Rev Targets',
#     'num clients' : 'nClients'
#     }
# sales_formats = {
#     sales_fields['date'] : '%d/%m/%Y'
# }

####################################################################################################
# 000 - IMPORT DATA
####################################################################################################

########################### Page 1 - Maps
# Import map and bar chart data
df_raw = pd.read_csv(raw_dataset_path)

# Create DataFrames for map and bar chart
df_ga = g.process_ga_data(df_raw)

# Prepare figure
fig_ga = g.create_ga_fig(df_ga, mapbox_access_token=mapbox_access_token)

#Create L1 dropdown options
repo_groups_l1_all = [
    {'label' : 'Counties: Number of Listings', 'value' : 0},
    {'label' : 'Counties: Average Listing Price', 'value' : 1},
    {'label' : 'Zipcodes: Number of Listings', 'value' : 2},
    {'label' : 'Zipcodes: Average Listing Price', 'value' : 3},
    ]
########################### Page 1 - Maps

########################### Page 2 - Analytics
# Read in external data into dash application
df = df_raw
df1 = df.copy()

# Read pickle file that can be obtained by running the first half or so of MLR.ipynb
MLR_MS_df = pd.read_pickle('data/pickle/MLR_modNshap.P')

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

# Define features to be utilized for generating scatter plots
features = ['square_footage', 'overall_crime_grade', 'ES_rating', 'lot_size', 'baths_half', 
'MS_rating', 'HS_rating', 'beds', 'baths_full', 'year_built', 'property_crime_grade']

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

# Series of zipcode data frames. May want to save elsewhere.
zip_dfs = pd.Series([], dtype='O')
for zipcode in set(df.zipcode.values):
    zip_dfs[zipcode] = df.loc[df.zipcode == zipcode]

# Predictions... should probably pre-load for each model.
df['MLR_price'] = df.apply(lambda row: MLR_MS_df.loc[row.zipcode,'model'].predict(row[features].to_numpy().reshape(1,-1)).item(), axis=1)
df['MLR_caprate'] = 100*12*(df.rent/df.MLR_price)

# Model group L1 options
crossfilter_model_options = [
    #{'label': 'Final Sale Price', 'value': 'price'},
    {'label': 'Multiple Linear Regression: Price', 'value': 'MLR_price'},
    #{'label': 'Multiple Linear Regression: Caprate', 'value': 'MLR_caprate'},
    ]

# Resolution group L2 options
crossfilter_resolution_options = [
    {'label': 'State', 'value': 'state'},
    # {'label': 'County', 'value': 'county'},
    {'label': 'Zip code', 'value': 'zipcode'},
    ]
########################### Page 2 - Analytics

###########################
#Import sales data
# xls = pd.ExcelFile(sales_filepath)
# sales_import=xls.parse('Static')

#Format date field
# sales_import[sales_fields['date']] = pd.to_datetime(sales_import[sales_fields['date']], format=sales_formats[sales_fields['date']])
# sales_import['date_2'] = sales_import[sales_fields['date']].dt.date
# min_dt = sales_import['date_2'].min()
# min_dt_str = str(min_dt)
# max_dt = sales_import['date_2'].max()
# max_dt_str = str(max_dt)

#Create L1 dropdown options
# repo_groups_l1 = sales_import[sales_fields['reporting_group_l1']].unique()
# repo_groups_l1_all_2 = [
#     {'label' : k, 'value' : k} for k in sorted(repo_groups_l1)
#     ]
# repo_groups_l1_all_1 = [{'label' : '(Select All)', 'value' : 'All'}]
# repo_groups_l1_all = repo_groups_l1_all_1 + repo_groups_l1_all_2

#Initialise L2 dropdown options
# repo_groups_l2 = sales_import[sales_fields['reporting_group_l2']].unique()
# repo_groups_l2_all_2 = [
#     {'label' : k, 'value' : k} for k in sorted(repo_groups_l2)
#     ]
# repo_groups_l2_all_1 = [{'label' : '(Select All)', 'value' : 'All'}]
# repo_groups_l2_all = repo_groups_l2_all_1 + repo_groups_l2_all_2
# repo_groups_l1_l2 = {}
# for l1 in repo_groups_l1:
#     l2 = sales_import[sales_import[sales_fields['reporting_group_l1']] == l1][sales_fields['reporting_group_l2']].unique()
#     repo_groups_l1_l2[l1] = l2

################################################################################################################################################## SET UP END

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
@app.callback(
    dash.dependencies.Output('ga-map', 'figure'),
	[
     dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value')
     ])
def update_chart(value):
    ## L1 selection (dropdown value is a list!)
    return fig_ga[value]

####################################################################################################
# 00B - SCATTER PLOT UPDATE
####################################################################################################
# Callback for dynamic dropdown
@app.callback(
    dash.dependencies.Output('reso_list', 'children'),
    dash.dependencies.Input('crossfilter-resolution', 'value')
)
# Create dynamic dropdown. Effects change in scatter-plot as well.
def update_scale(resolution_dropdown):
    new_dropdown = dcc.Dropdown(
    list(set(df[resolution_dropdown])),
    id='filter-dropdown',
        value=df[resolution_dropdown].values[0]
    )
    return new_dropdown

# First callback for SHAP values
@app.callback(
    dash.dependencies.Output('shap-bee', 'src'),
    [
        dash.dependencies.Input('filter-dropdown','value'),
        #dash.dependencies.State('crossfilter-resolution', 'value') ## Useful if we add county as well
        #dash.dependencies.Input('crossfilter-model', 'State')    ## need to figure out a good way to do this. Will probably use a Series of those model dataframes
        dash.dependencies.State('shap-bee', 'src')
    ]
)
# Function to update SHAP values
def update_shap(focus, current):
    if focus == 'Georgia':
        return current
    shap.summary_plot( 
        MLR_MS_df.loc[focus,'shap_values'], 
        show=False)
    fig = plt.gcf()
    buf = io.BytesIO() # in-memory files
    plt.savefig(buf, format = "png", transparent = True)
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()
    X = "data:image/png;base64,{}".format(data)
    return X

# First callback to coordinate scatterplot with feature, model, and color gradient schema
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature', 'value'),
        dash.dependencies.Input('crossfilter-model', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value'),
        dash.dependencies.Input('crossfilter-resolution', 'value'),
        dash.dependencies.Input('filter-dropdown','value')
    ]
)
# Function to update graph in response to feature, model, and color gradient schema being updated
def update_graph(feature, model, gradient, resolution, focus):
    # Filter dataframe 
    df1 = df.loc[df[resolution] == focus]
    # Define points and respective colors of scatter plot
    cols = df1[feature].values #df[feature].values
    hover_names = []
    for ix, val in zip(df1.index.values, df1[feature].values):
        hover_names.append(f'Customer {ix:03d}<br>{feature} value of {val}')
    # Generate scatter plot
    fig = px.scatter(
        df1,
        x=df1[feature],
        y=df1[model],
        color=cols,
        opacity=0.8,
        hover_name=hover_names,
        hover_data=features,
        template='plotly_dark',
        color_continuous_scale=gradient,
    )
    # Update feature information when user hovers over data point
    # customdata_set = list(df[features].to_numpy())
    fig.update_traces(customdata=df1.index)
    
    # Layout of scatter plot
    fig.update_layout(
        coloraxis_colorbar={'title': f'{feature}'},
        coloraxis_showscale=True,
        legend_title_text='Spend',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark'
    )
    # Axis settings for scatter plot
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig

# Function to create point plot of averages of selected features
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
        template='plotly_dark'
    )
    # Axis settings for point plot
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(type="log", range=[0,5])

    return fig

# Callback to update point plot based on user hovering over points in scatter plot
@app.callback(
    dash.dependencies.Output('point-plot', 'figure'),
    [
        # Update graph on click
        dash.dependencies.Input('scatter-plot', 'clickData')
        # Update graph on hover
        # dash.dependencies.Input('scatter-plot', 'hoverData')
    ]
)

# Function to trigger last function in response to user click point in scatter plot
def update_point_plot(clickData):
    index = clickData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)

####################################################################################################
# 001 - L2 DYNAMIC DROPDOWN OPTIONS
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('reporting-groups-l2dropdown-sales', 'options'),
#     [dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value')])
# def l2dropdown_options(l1_dropdown_value):
#     isselect_all = 'Start' #Initialize isselect_all
#     #Rembember that the dropdown value is a list !
#     for i in l1_dropdown_value:
#         if i == 'All':
#             isselect_all = 'Y'
#             break
#         elif i != '':
#             isselect_all = 'N'
#         else:
#             pass
#     #Create options for individual selections
#     if isselect_all == 'N':
#         options_0 = []
#         for i in l1_dropdown_value:
#             options_0.append(repo_groups_l1_l2[i])
#         options_1 = [] # Extract string of string
#         for i1 in options_0:
#             for i2 in i1:
#                 options_1.append(i2)
#         options_list = [] # Get unique values from the string
#         for i in options_1:
#             if i not in options_list:
#                 options_list.append(i)
#             else:
#                 pass
#         options_final_1 = [
#             {'label' : k, 'value' : k} for k in sorted(options_list)]
#         options_final_0 = [{'label' : '(Select All)', 'value' : 'All'}]
#         options_final = options_final_0 + options_final_1
#     #Create options for select all or none
#     else:
#         options_final_1 = [
#             {'label' : k, 'value' : k} for k in sorted(repo_groups_l2)]
#         options_final_0 = [{'label' : '(Select All)', 'value' : 'All'}]
#         options_final = options_final_0 + options_final_1

#     return options_final

####################################################################################################
# 002 - RECAP TABLE
####################################################################################################
# @app.callback(
#     [dash.dependencies.Output('recap-table', 'data'), dash.dependencies.Output('recap-table', 'columns'), dash.dependencies.Output('recap-table', 'style_data_conditional')],
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df_1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     del sales_df

#     # Aggregate df
#     metrics = ['Sales (M u)','Revenues (M €)','Customers (M)']
#     result = [df_1[sales_fields['sales']].sum()/1000000, df_1[sales_fields['revenues']].sum()/1000000, df_1[sales_fields['num clients']].sum()/1000000]
#     target = [df_1[sales_fields['sales target']].sum()/1000000, df_1[sales_fields['rev target']].sum()/1000000, '']
#     performance = [df_1[sales_fields['sales']].sum()/df_1[sales_fields['sales target']].sum(), df_1[sales_fields['revenues']].sum()/df_1[sales_fields['rev target']].sum(), '']
#     df = pd.DataFrame({'KPI' : metrics, 'Result' : result, 'Target': target, 'Target_Percent' : performance})

#     # Configure table data
#     data = df.to_dict('records')
#     columns = [
#         {'id' : 'KPI', 'name' : 'KPI'},
#         {'id' : 'Result', 'name' : 'Result', 'type' : 'numeric', 'format' : Format(scheme=Scheme.fixed, precision=2, group=Group.yes, group_delimiter=',', decimal_delimiter='.')},
#         {'id' : 'Target', 'name' : 'Target',  'type' : 'numeric', 'format' : Format(scheme=Scheme.fixed, precision=2, group=Group.yes, group_delimiter=',', decimal_delimiter='.')},
#         {'id' : 'Target_Percent', 'name' : '% Target', 'type': 'numeric', 'format' : FormatTemplate.percentage(2)}
#     ]

#     # Configure conditional formatting
#     conditional_style=[
#         {'if' : {
#             'filter_query' : '{Result} >= {Target} && {Target} > 0',
#             'column_id' : 'Target_Percent'},
#         'backgroundColor' : corporate_colors['light-green'],
#         'color' : corporate_colors['dark-green'],
#         'fontWeight' : 'bold'
#         },
#         {'if' : {
#             'filter_query' : '{Result} < {Target} && {Target} > 0',
#             'column_id' : 'Target_Percent'},
#         'backgroundColor' : corporate_colors['pink-red'],
#         'color' : corporate_colors['dark-green'],
#         'fontWeight' : 'bold'
#         },
#     ]

#     return data, columns, conditional_style

####################################################################################################
# 003 - SALES COUNT DAY
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-count-day', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     #Aggregate df
#     val_cols = [sales_fields['sales'],sales_fields['sales target']]
#     sales_df = sales_df.groupby(sales_fields['date'])[val_cols].agg('sum')
#     sales_df.reset_index(inplace=True)

#     # Filter based on the date filters
#     df = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     del sales_df

#     # Build graph
#     hovertemplate_xy = (
#     "<i>Day</i>: %{x|%a, %d-%b-%Y}<br>"+
#     "<i>Sales</i>: %{y:,d}"+
#     "<extra></extra>") # Remove trace info
#     data = go.Scatter(
#         x = df[sales_fields['date']],
#         y = df[sales_fields['sales']],
#         line = {'color' : corporate_colors['light-green'], 'width' : 0.5},
#         hovertemplate = hovertemplate_xy)
#     fig = go.Figure(data=data, layout=corporate_layout)
#     fig.update_layout(
#         title={'text' : "Sales per Day"},
#         xaxis = {
#             'title' : "Day",
#             'tickformat' : "%d-%m-%y"},
#         yaxis = {
#             'title' : "Sales (units)",
#             'range' : [0, 100000]},
#         showlegend = False)

#     return fig

####################################################################################################
# 004 - SALES COUNT MONTH
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-count-month', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     df1['month'] = df1[sales_fields['date']].dt.month
#     del sales_df

#     #Aggregate df
#     val_cols = [sales_fields['sales'], sales_fields['sales target']]
#     df = df1.groupby('month')[val_cols].agg('sum')
#     df.reset_index(inplace=True)
#     del df1

#     # Build graph
#     hovertemplate_xy = (
#     "<i>Month</i>: %{x}<br>"+
#     "<i>Sales</i>: %{y:,d}"+
#     "<extra></extra>") # Remove trace info
#     data = go.Bar(
#         x = df['month'],
#         y = df[sales_fields['sales']],
#         marker = {'color' : corporate_colors['light-green'], 'opacity' : 0.75},
#         hovertemplate = hovertemplate_xy)
#     fig = go.Figure(data=data, layout=corporate_layout)

#     # Add target% as line on secondary axis
#     hovertemplate_xy2 = (
#     "<i>Month</i>: %{x}<br>"+
#     "<i>Target percentage</i>: %{y:%}"+
#     "<extra></extra>") # Remove trace info
#     fig.add_trace(
#         go.Scatter(
#             x = df['month'],
#             y = df[sales_fields['sales']]/df[sales_fields['sales target']],
#             line = {'color': corporate_colors['pink-red'], 'width' : 2},
#             yaxis = "y2",
#             opacity = 0.75,
#             hovertemplate = hovertemplate_xy2)
#     )
#     fig.update_layout(
#         title={'text' : "Sales per Month vs Target"},
#         xaxis = {
#             'title' : "Month",
#             'tickvals' : [1,2,3,4,5,6,7,8,9,10,11,12], #Display x values with different labels
#             'ticktext' : ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']},
#         yaxis = {'title' : "Sales (units)"},
#         showlegend = False)
#     fig.update_layout(yaxis2 = corporate_yaxis)
#     fig.update_layout(
#         yaxis2 = {
#             'title' : "% over Sales target",
#             'side' : "right",
#             'showgrid' : False,
#             'tickformat' : ".0%",
#             'range' : [0, 1.15],
#             'overlaying' : "y",
#             'linewidth' : 1},
#         hovermode = 'x')

#     return fig

####################################################################################################
# 005 - WEEKLY-WEEKDAY SALES HEATMAP
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-weekly-heatmap', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     df1['week'] = df1[sales_fields['date']].dt.strftime("%V")
#     df1['weekday'] = df1[sales_fields['date']].dt.weekday
#     del sales_df

#     #Aggregate df
#     val_cols = [sales_fields['sales']]
#     df = df1.groupby(['week','weekday'])[val_cols].agg('sum')
#     df.reset_index(inplace=True)
#     del df1

#     # Build graph
#     hovertemplate_here = (
#     "<i>Week</i>: %{x}<br>"+
#     "<i>Weekday</i>: %{y}<br>"+
#     "<i>Sales</i>: %{z}"+
#     "<extra></extra>") # Remove trace info
#     data = go.Heatmap(
#         x = df['weekday'],
#         y = df['week'],
#         z = df[sales_fields['sales']],
#         hovertemplate = hovertemplate_here,
#         hoverongaps = False,
#         colorscale = corporate_colorscale,
#         showscale = False,
#         xgap = 1,
#         ygap = 1)
#     fig = go.Figure(data=data, layout=corporate_layout)
#     fig.update_layout(
#         title={'text' : "Heatmap: Sales by week and weekeday"},
#         xaxis = {
#             'title' : "Weekday",
#             'tickvals' : [0,1,2,3,4,5,6], #Display x values with different labels
#             'ticktext' : ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']},
#         yaxis = {
#             'title' : "Calendar Week",
#             'showgrid' : False})

#     return fig

####################################################################################################
# 006 - SALES BY COUNTRY
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-count-country', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     del sales_df

#     #Aggregate df
#     val_cols = [sales_fields['sales']]
#     df = df1.groupby(sales_fields['reporting_group_l1'])[val_cols].agg('sum')
#     df.reset_index(inplace=True)
#     df.sort_values(sales_fields['reporting_group_l1'], axis=0, ascending=True, inplace=True, na_position='last')
#     del df1

#     #Prepare incr % data
#     hover_text=[]
#     sale_perc=[]
#     sale_base=[0]
#     sale_b=0
#     sales_tot = df[sales_fields['sales']].sum()
#     for index, row in df.iterrows():
#         sale_p = row[sales_fields['sales']]/sales_tot
#         hover_text.append(("<i>Country</i>: {}<br>"+
#                             "<i>Sales</i>: {:.2%}"+
#                             "<extra></extra>").format(row[sales_fields['reporting_group_l1']],
#                                                         sale_p))
#         sale_b = sale_b + sale_p
#         sale_perc.append(sale_p)
#         sale_base.append(sale_b)
#     sale_base = sale_base[:-1]
#     df['sale_p'] = sale_perc
#     df['hovertext'] = hover_text

#     # Build graph
#     data = go.Bar(
#         x = df[sales_fields['reporting_group_l1']],
#         y = df['sale_p'],
#         base = sale_base,
#         marker = {'color': corporate_colors['light-green'],
#                 'opacity' : 0.75},
#         hovertemplate = df['hovertext'])
#     fig = go.Figure(data=data, layout=corporate_layout)
#     fig.update_layout(
#         title={'text' : "Sales Percentage by Country"},
#         xaxis = {'title' : "Country", 'tickangle' : 0},
#         yaxis = {
#             'title' : "Sales Percentage",
#             'tickformat' : ".0%",
#             'range' : [0, 1]},
#         barmode = 'group',
#         showlegend = False)

#     return fig

####################################################################################################
# 007 - SALES BUBBLE CHART
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-bubble-county', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     del sales_df

#     #Aggregate df
#     val_cols = [sales_fields['sales'], sales_fields['num clients'], sales_fields['revenues']]
#     df = df1.groupby(sales_fields['reporting_group_l1'])[val_cols].agg('sum')
#     df.reset_index(inplace=True)
#     df['rev_per_cl'] = df[sales_fields['revenues']]/df[sales_fields['num clients']]
#     del df1

#     # Build graph
#     #Add hover text info on the df
#     hover_text = []
#     for index, row in df.iterrows():
#         hover_text.append(('<i>Country</i>: {}<br>'+
#                           '<i>Sales</i>: {:,d}<br>'+
#                           '<i>Clients</i>: {:,d}<br>'+
#                           '<i>Revenues</i>: {:,d}'+
#                           '<extra></extra>').format(row[sales_fields['reporting_group_l1']],
#                                                 row[sales_fields['sales']],
#                                                 row[sales_fields['num clients']],
#                                                 row[sales_fields['revenues']]))
#     df['hovertext'] = hover_text
#     sizeref = 2.*max(df[sales_fields['sales']])/(100**2)

#     #Create bubbles (1 color per country, one trace per city)
#     country_names = sorted(df[sales_fields['reporting_group_l1']].unique())
#     countries = len(country_names)
#     colorscale = colorscale_generator(n=countries, starting_col = {'r' : 57, 'g' : 81, 'b' : 85}, finish_col = {'r' : 251, 'g' : 251, 'b' : 252})

#     fig = go.Figure(layout=corporate_layout)
#     i = 0
#     for co in country_names:
#         color = colorscale[i]
#         i = i+1
#         df_i = df.loc[df[sales_fields['reporting_group_l1']]==co, :].copy()
#         fig.add_trace(
#             go.Scatter(
#                 x=df_i['rev_per_cl'],
#                 y=df_i[sales_fields['num clients']],
#                 name=co,
#                 hovertemplate=df_i['hovertext'],
#                 marker_size=df_i[sales_fields['sales']],
#                 marker = {
#                     'color' : color,
#                     'line_width' : 1,
#                     'line' : {'color' : corporate_colors['light-grey']}
#                 })
#             )

#     fig.update_traces(mode='markers', marker= {'sizemode' : 'area', 'sizeref' : sizeref})
#     corporate_margins_here = corporate_margins
#     corporate_margins_here['t'] = 65
#     fig.update_layout(
#         title={'text' : "Revenue per Client by Country"},
#         xaxis = {'title' : "Revenue per Client", 'tickangle' : 0},
#         yaxis = {'title' : "Sales (Units)"},
#         margin = corporate_margins_here)

#     return fig

####################################################################################################
# 008 - SALES BY COUNTRY & CITY
####################################################################################################
# @app.callback(
#     dash.dependencies.Output('sales-count-city', 'figure'),
# 	[dash.dependencies.Input('date-picker-sales', 'start_date'),
# 	 dash.dependencies.Input('date-picker-sales', 'end_date'),
#      dash.dependencies.Input('reporting-groups-l1dropdown-sales', 'value'),
#      dash.dependencies.Input('reporting-groups-l2dropdown-sales', 'value')])
# def update_chart(start_date, end_date, reporting_l1_dropdown, reporting_l2_dropdown):
#     start = dt.strptime(start_date, '%Y-%m-%d')
#     end = dt.strptime(end_date, '%Y-%m-%d')

#     # Filter based on the dropdowns
#     isselect_all_l1 = 'Start' #Initialize isselect_all
#     isselect_all_l2 = 'Start' #Initialize isselect_all
#     ## L1 selection (dropdown value is a list!)
#     for i in reporting_l1_dropdown:
#         if i == 'All':
#             isselect_all_l1 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l1 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l1 == 'N':
#         sales_df_1 = sales_import.loc[sales_import[sales_fields['reporting_group_l1']].isin(reporting_l1_dropdown), : ].copy()
#     else:
#         sales_df_1 = sales_import.copy()
#     ## L2 selection (dropdown value is a list!)
#     for i in reporting_l2_dropdown:
#         if i == 'All':
#             isselect_all_l2 = 'Y'
#             break
#         elif i != '':
#             isselect_all_l2 = 'N'
#         else:
#             pass
#     # Filter df according to selection
#     if isselect_all_l2 == 'N':
#         sales_df = sales_df_1.loc[sales_df_1[sales_fields['reporting_group_l2']].isin(reporting_l2_dropdown), :].copy()
#     else:
#         sales_df = sales_df_1.copy()
#     del sales_df_1

#     # Filter based on the date filters
#     df1 = sales_df.loc[(sales_df[sales_fields['date']]>=start) & (sales_df[sales_fields['date']]<=end), :].copy()
#     del sales_df

#     # Aggregate df
#     val_cols = [sales_fields['sales'],sales_fields['sales target']]
#     df = df1.groupby([sales_fields['reporting_group_l1'],sales_fields['reporting_group_l2']])[val_cols].agg('sum')
#     df.reset_index(inplace=True)
#     # Include hover data
#     hover_text=[]
#     for index, row in df.iterrows():
#         hover_text.append(("<i>Country</i>: {}<br>"+
#                             "<i>City</i>: {}<br>"+
#                             "<i>Sales</i>: {:,d}<br>"+
#                             "<i>Targets</i>: {:,d}"+
#                             "<extra></extra>").format(row[sales_fields['reporting_group_l1']],
#                                                         row[sales_fields['reporting_group_l2']],
#                                                         row[sales_fields['sales']],
#                                                         row[sales_fields['sales target']]))
#     df['hovertext'] = hover_text
#     df['l1l2'] = df[sales_fields['reporting_group_l1']] + "_" + df[sales_fields['reporting_group_l2']]
#     # Generate colors
#     ncolors = len(df[sales_fields['reporting_group_l2']].unique())
#     colorscale = colorscale_generator(n=ncolors)

#     # Build graph
#     data=[]
#     i = 0
#     for l in sorted(df['l1l2']):
#         df_l = df.loc[(df['l1l2']==l), :].copy()
#         trace= go.Bar(
#             name = l,
#             x = df_l[sales_fields['reporting_group_l1']],
#             y = df_l[sales_fields['sales']],
#             hovertemplate = df_l['hovertext'],
#             marker = {
#                 'color' : colorscale[i],
#                 'opacity' : 0.85,
#                 'line_width' : 1,
#                 'line' : {'color' : colorscale[i]}
#                 }
#             )
#         i=i+1
#         data.append(trace)
#     fig = go.Figure(data=data, layout=corporate_layout)
#     fig.update_layout(
#         barmode='stack',
#         title={'text' : "Sales by Country & City"},
#         xaxis = {'title' : "Country", 'tickangle' : 0},
#         yaxis = {'title' : "Sales (Units)"},
#         showlegend = False)

#     return fig
