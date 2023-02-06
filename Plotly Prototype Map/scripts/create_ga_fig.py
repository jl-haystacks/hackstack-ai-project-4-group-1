import os
import json
import numpy as np
import pandas as pd
from logzero import logger
from plotly import graph_objs as go
import plotly.express as px
import utils_haystacks as f
pd.set_option('chained_assignment', None)

################################################################################################
################################################################################################
################################################################################################

# def process_pandemic_data(df, startdate = '2020-03-01'):
def process_ga_data(df):
    '''
    This method below shall prepare aggregated data to feed into creating the figures in the next function
    '''
    # Aggregate counties by number of houses
    ga_county_count = pd.DataFrame(df.county.value_counts())
    ga_county_count = ga_county_count.reset_index().rename(columns = {'index':'county', 'county':'num_houses'})
    ga_county_count['county'] = ga_county_count['county'].astype(str)

    # Aggregate counties by average selling price of houses
    ga_county_price = pd.DataFrame(df.groupby(['county'])['price'].mean())
    ga_county_price = ga_county_price.reset_index().rename(columns = {'price':'avg_price'})
    ga_county_price['county'] = ga_county_price['county'].astype(str)

    # Aggregate zipcodes by number of houses
    ga_zipcode_count = pd.DataFrame(df.zipcode.value_counts())
    ga_zipcode_count = ga_zipcode_count.reset_index().rename(columns = {'index':'zipcode', 'zipcode':'num_houses'})
    ga_zipcode_count['zipcode'] = ga_zipcode_count['zipcode'].astype(str)

    # Aggregate zipcodes by average selling price of houses
    ga_zipcode_price = pd.DataFrame(df.groupby(['zipcode'])['price'].mean())
    ga_zipcode_price = ga_zipcode_price.reset_index().rename(columns = {'price':'avg_price'})
    ga_zipcode_price['zipcode'] = ga_zipcode_price['zipcode'].astype(str)
    
    # Output final result to input into next function
    return [ga_county_count, ga_county_price, ga_zipcode_count, ga_zipcode_price]


def create_ga_fig(df, mapbox_access_token):
    '''
    This method shall create a hybrid view of bar graphs to illustrate metrics as well as the choropleth map
    More about this method here:
    https://towardsdatascience.com/build-an-interactive-choropleth-map-with-plotly-and-dash-1de0de00dce0
    '''
    # Create custom color scale for choropleth maps
    pl_deep=[[0.0, 'rgb(253, 253, 204)'],
         [0.1, 'rgb(201, 235, 177)'],
         [0.2, 'rgb(145, 216, 163)'],
         [0.3, 'rgb(102, 194, 163)'],
         [0.4, 'rgb(81, 168, 162)'],
         [0.5, 'rgb(72, 141, 157)'],
         [0.6, 'rgb(64, 117, 152)'],
         [0.7, 'rgb(61, 90, 146)'],
         [0.8, 'rgb(65, 64, 123)'],
         [0.9, 'rgb(55, 44, 80)'],
         [1.0, 'rgb(39, 26, 44)']]
    
    # Unpack DataFrames generated from previous function
    Types = df
    
    # Generate list of counties and zipcodes
    counties = Types[0]['county'].str.title().tolist()
    zipcodes = Types[2]['zipcode'].str.title().tolist()

    # Load in county-level geoJSON
    # https://maps.princeton.edu/catalog/tufts-gacounties10
    with open('../data/geojson/tufts-gacounties10-geojson.json') as json_data:
        map_ga_counties = json.load(json_data)

    # Load in zipcode-level geoJSON
    # https://maps.princeton.edu/catalog/harvard-tg00gazcta
    with open('../data/geojson/harvard-tg00gazcta-geojson.json') as json_data:
        map_ga_zipcodes = json.load(json_data)

    # Create choropleth maps for trace1
    trace1 = []
    # For county-level granularity
    for q in Types[:2]:
        trace1.append(go.Choroplethmapbox(
            geojson = map_ga_counties,
            locations = counties,
            featureidkey='properties.name10',
            # Fix this to indicate what value is being aggregated using the correct syntax
            # x = q.columns[0],
            z = q[q.columns[1]].tolist(), 
            colorscale = pl_deep,
            text = counties, 
            colorbar = dict(thickness=20, ticklen=3),
            marker_line_width=0, marker_opacity=0.7,
            visible=False,
            subplot='mapbox1',
            hovertemplate = "<b>%{text}</b><br><br>" +
                            "Value: %{z}<br>" + # Fix this to indicate what value is being aggregated using the correct syntax
                            "<extra></extra>")) # "<extra></extra>" means we info in the secondary box is not displayed
    # # For zipcode-level granularity
    for q in Types[2:]:
        trace1.append(go.Choroplethmapbox(
            geojson = map_ga_zipcodes,
            locations = zipcodes,
            featureidkey='properties.ZCTA',
            # Fix this to indicate what value is being aggregated using the correct syntax
            # x = q.columns[0],
            z = q[q.columns[1]].tolist(), 
            colorscale = pl_deep,
            text = zipcodes, 
            colorbar = dict(thickness=20, ticklen=3),
            marker_line_width=0, marker_opacity=0.7,
            visible=False,
            subplot='mapbox1',
            hovertemplate = "<b>%{text}</b><br><br>" +
                            "Value: %{z}<br>" + # Fix this to indicate what value is being aggregated using the correct syntax
                            "<extra></extra>")) # "<extra></extra>" means we info in the secondary box is not displayed

    # Create bar plot for trace2
    trace2 = []
    for q in Types:
        trace2.append(go.Bar(
            # Sort items in descending order by value (independent variable)
            x = q.sort_values([q.columns[0]], ascending=False).head(10)[q.columns[0]].tolist(),
            # Sort items in corresponding order by name (dependent variable)
            y = q.sort_values([q.columns[0]], ascending=False).head(10).index.tolist(),
            xaxis = 'x2',
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(91, 207, 135, 0.3)',
                line = dict(
                    color = 'rgba(91, 207, 135, 2.0)',
                    width = 0.5),
            ),
            visible = False,
            # name='Top 10 suburbs with the highest {} median price'.format(q),
            orientation = 'h',
        ))
    trace2[0]['visible'] = True

    # Geographic center of Georgia:
    # https://georgiahistory.com/ghmi_marker_updated/geographic-center-of-georgia/
    center = {"lat": 32.6461, "lon": -83.4317}
    latitude = center['lat']
    longitude = center['lon']

    # Set up coordinate systems in the layout
    layout = go.Layout(
        title = {'text': 'Georgia House Quantity and Average Selling Price',
        'font': {'size': 28, 'family': 'Arial'}},
        height = 600,
        autosize = True,
        
        mapbox1 = dict(
            domain = {'x': [0.3, 1],'y': [0, 1]},
            center = dict(lat = latitude, lon = longitude),
            accesstoken = mapbox_access_token, 
            zoom = 6),
        xaxis2={
            'zeroline': False,
            "showline": False,
            "showticklabels": True,
            'showgrid': True,
            'domain': [0, 0.25],
            'side': 'left',
            'anchor': 'x2',
        },
        yaxis2={
            'domain': [0.4, 0.9],
            'anchor': 'y2',
            'autorange': 'reversed',
        },
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(204, 204, 204)',
        plot_bgcolor='rgb(204, 204, 204)',
    )
    # Add a dropdown menu in the layout
    layout.update(updatemenus=list([
        dict(x = 0,
            y = 1,
            xanchor = 'left',
            yanchor = 'middle',
            buttons = list([
                dict(
                    args = ['visible', [True, False, False, False]],
                    label = 'Counties: Number of Houses',
                    method = 'restyle'
                    ),
                dict(
                    args = ['visible', [False, True, False, False]],
                    label = 'Counties: Average House Price',
                    method = 'restyle'
                    ),
                dict(
                    args = ['visible', [False, False, True, False]],
                    label = 'Zipcodes: Number of Houses',
                    method = 'restyle'
                    ),
                dict(
                    args = ['visible', [False, False, False, True]],
                    label = 'Zipcodes: Average House Price',
                    method = 'restyle'
                    )
                ]),
            )]))

    # Output final result to create pickle file
    return go.Figure(data = trace2 + trace1, layout = layout)

################################################################################################
################################################################################################
################################################################################################


if __name__ =="__main__":    

    # See here: https://stackoverflow.com/questions/22282760/filenotfounderror-errno-2-no-such-file-or-directory
    # cwd = os.getcwd()  # Get the current working directory (cwd)
    # files = os.listdir(cwd)  # Get all the files in that directory
    # print("Files in %r: %s" % (cwd, files))

    # Change working directory. More about it here:
    # https://note.nkmk.me/en/python-os-getcwd-chdir/
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Loading necessary information
    mapbox_access_token = f.config['mapbox']['token']
    raw_dataset_path = "." + f.RAW_PATH + f.config['path']['name']
    
    # Creating dataFrames
    df_raw = pd.read_csv(raw_dataset_path)
    df_ga = process_ga_data(df_raw)
    # df_total_kpi = df_world.groupby('date').sum().sort_index().iloc[-1]
    
    # Preparing figure
    fig_ga = create_ga_fig(df_ga, mapbox_access_token=mapbox_access_token)

    # Storing all necessay information for app
    save = {
        'figure':fig_ga,
        # 'last_date':df_world.index[-1][0],
        # 'total_confirmed': f.spacify_number(int(df_total_kpi['confirmed'])),
        # 'total_deaths': f.spacify_number(int(df_total_kpi['deaths'])),
        # 'total_recovered': f.spacify_number(int(df_total_kpi['recovered']))
    }
    f.save_pickle(save, 'ga_info.p')

    # Display information
    logger.info('Georgia map updated.')
    logger.info('Data sorted for dash application.')
    # logger.info('Last date in new dataset is {}'.format(df_world.index[-1][0]))