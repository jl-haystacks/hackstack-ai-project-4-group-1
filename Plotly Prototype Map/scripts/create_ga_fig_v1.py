import os
import json
import numpy as np
import pandas as pd
from logzero import logger
from plotly import graph_objs as go
import plotly.express as px
import geopandas as gpd
import utils_haystacks as f
pd.set_option('chained_assignment', None)

################################################################################################
################################################################################################
################################################################################################

# def process_pandemic_data(df, startdate = '2020-03-01'):
def process_ga_data(df):
    # Read raw data file to create new DataFrame
    # haystacks_ga_data = pd.read_csv('data/haystacks_ga_clean_new_format.csv', dtype={"county": str, "zipcode": str})

    # Aggregate counties by number of houses
    ga_county_count = pd.DataFrame(df.county.value_counts())
    ga_county_count = ga_county_count.reset_index().rename(columns = {'index':'county', 'county':'num_houses'})

    return ga_county_count

    '''
    The code below is needed for a time-based animation.
    '''

    # Columns renaming
    # df.columns = [col.lower() for col in df.columns]

    # Keep only from a starting date
    # df = df[df['date'] > startdate]

    # Create a zone per zone/subzone
    # df['zone'] = df['zone'].apply(str) + ' ' + df['sub zone'].apply(lambda x: str(x).replace('nan', ''))
    
    # Extracting latitute and longitude
    # df['lat'] = df['latitude']
    # df['lon'] = df['longitude']
    # df['lat'] = df['location'].apply(lambda x: x.split(',')[0])
    # df['lon'] = df['location'].apply(lambda x: x.split(',')[1])

    # Saving countries positions (latitude and longitude per subzones)
    # country_position = df[['zone', 'lat', 'lon']].drop_duplicates(['zone']).set_index(['zone'])

    # Pivoting per category
    # df = pd.pivot_table(df, values='count', index=['date', 'zone'], columns=['category'])
    # df.columns = ['confirmed', 'deaths', 'recovered']

    # Merging locations after pivoting
    # df = df.join(country_position)

    # Filling nan values with 0
    # df = df.fillna(0)

    # Compute bubble sizes
    # df['size'] = df['confirmed'].apply(lambda x: (np.sqrt(x/100) + 1) if x > 500 else (np.log(x) / 2 + 1)).replace(np.NINF, 0) * 0.5
    # df['color'] = (df['recovered']/df['confirmed']).fillna(0).replace(np.inf , 0) * 100
    
    # Prepare display values for bubble hover
    # df['confirmed_display'] = df['confirmed'].apply(int).apply(f.spacify_number)
    # df['recovered_display'] = df['recovered'].apply(int).apply(f.spacify_number)
    # df['deaths_display'] = df['deaths'].apply(int).apply(f.spacify_number)

    
    # return df


def create_ga_fig(df, mapbox_access_token):
    # To create a static choropleth map

    # Set the filepath and load in a shapefile
    # Shape file found here:
    # https://maps.princeton.edu/catalog/tufts-gacounties10
    ga_counties = "../data/geojson/tufts-gacounties10-geojson.json"
    map_ga_counties = gpd.read_file(ga_counties)

    # plotly and geopandas necessary for producing these choropleths
    # For plotly express to print maps, jsons must be used
    # More here: https://plotly.com/python/mapbox-county-choropleth/
    # https://plotly.github.io/plotly.py-docs/generated/plotly.express.choropleth_mapbox.html
    # https://stackoverflow.com/questions/67362742/geojson-issues-with-plotly-choropleth-mapbox
    # https://community.plotly.com/t/choroplethmapbox-does-not-show/41229/6
    ga_counties_fig = px.choropleth_mapbox(df, geojson=map_ga_counties, locations='county', color='num_houses',
                                        color_continuous_scale="Viridis", #range_color=(0, 12), 
                                        mapbox_style="carto-positron", zoom=5.5, 
                                        # Geographic center of Georgia:
                                        # https://georgiahistory.com/ghmi_marker_updated/geographic-center-of-georgia/
                                        center = {"lat": 32.6461, "lon": -83.4317},
                                        opacity=0.5, labels={'num_houses':'Number of Houses'},
                                        featureidkey='properties.name10')
    
    return ga_counties_fig


    '''
    The code below is needed for a time-based animation.
    For a Plotly Graph Objects (go) Figure to be properly displayed,
    the following parameters need to be specified:
    1. data
    2. layout
    3. frames
    More here: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html
    '''
   
    # days = df.index.levels[0].tolist()
    # day = min(days)

    # Defining each Frame
    # frames = [{
    #     # 'traces':[0],
    #     'name':'frame_{}'.format(day),
    #     'data':[{
    #         'type':'scattermapbox',
    #         'lat':df.xs(day)['lat'],
    #         'lon':df.xs(day)['lon'],
    #         'marker':go.scattermapbox.Marker(
    #             size=df.xs(day)['size'],
    #             color=df.xs(day)['color'],
    #             showscale=True,
    #             colorbar={'title':'Recovered', 'titleside':'top', 'thickness':4, 'ticksuffix':' %'},
    #             # color_continuous_scale=px.colors.cyclical.IceFire,
    #         ),
    #         'customdata':np.stack((df.xs(day)['confirmed_display'], df.xs(day)['recovered_display'],  df.xs(day)['deaths_display'], pd.Series(df.xs(day).index)), axis=-1),
    #         'hovertemplate': "<extra></extra><em>%{customdata[3]}  </em><br>🚨  %{customdata[0]}<br>🏡  %{customdata[1]}<br>⚰️  %{customdata[2]}",
    #     }],           
    # } for day in days]  


    # Prepare the frame to display
    # data = frames[-1]['data']     

    # And specify the adequate button postion    
    # active_frame=len(days) - 1

    # Defining the slider to navigate between frames
    # sliders = [{
    #     'active':active_frame,
    #     'transition':{'duration': 0},
    #     'x':0.08,     #slider starting position  
    #     'len':0.88,
    #     'currentvalue':{
    #         'font':{'size':15}, 
    #         'prefix':'📅 ', # Day:
    #         'visible':True, 
    #         'xanchor':'center'
    #         },  
    #     'steps':[{
    #         'method':'animate',
    #         'args':[
    #             ['frame_{}'.format(day)],
    #             {
    #                 'mode':'immediate',
    #                 'frame':{'duration':250, 'redraw': True}, #100
    #                 'transition':{'duration':100} #50
    #             }
    #             ],
    #         'label':day
    #     } for day in days]
    # }]

    # play_button = [{
    #     'type':'buttons',
    #     'showactive':True,
    #     'y':-0.08,
    #     'x':0.045,
    #     'buttons':[{
    #         'label':'🎬', # Play
    #         'method':'animate',
    #         'args':[
    #             None,
    #             {
    #                 'frame':{'duration':250, 'redraw':True}, #100
    #                 'transition':{'duration':100}, #50
    #                 'fromcurrent':True,
    #                 'mode':'immediate',
    #             }
    #         ]
    #     }]
    # }]

    # Loading Map Outline Coordinates (County or Zipcode)
    # with open('../assets/map_outline.txt') as f:
    #     map_outline = f.readlines()
    # f.close()

    # Global Layout
    # Reference here: https://plotly.com/python/reference/layout/mapbox/
    # layout = go.Layout(
    #     height=600,
    #     autosize=True,
    #     hovermode='closest',
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     mapbox={
    #         'accesstoken':mapbox_access_token,
    #         'bearing':0,
            # Geographic center of Georgia:
            # https://georgiahistory.com/ghmi_marker_updated/geographic-center-of-georgia/
            # 'center':{"lat": 32.6461, "lon": -83.4317},
            # 'pitch':0,
            # 'zoom':5.5,
            # 'style':'light',
            # This is where to apply geoJSON layers for county or zipcode
            #  'layers':[{
            #      'source':{
            #          "type":"GeometryCollection",
            #          "geometries":[{
            #              "type":"MultiPolygon",
            #              "coordinates":[[[
            #                  map_outline
            #              ]]] 
            #          }]
            #     },
            #      'type':"line",
            #      'below':"traces",
            #      'color':'7392DA',
            #      'opacity': 0.5
            #  }]
        # },
        # updatemenus=play_button,
        # sliders=sliders,
    #     margin={"r":0,"t":0,"l":0,"b":0},
    # )

    # return go.Figure(data=data, layout=layout, frames=frames)

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