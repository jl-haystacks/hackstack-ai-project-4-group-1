import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
# from dash import dash_table
# from dash.dash_table.Format import Format, Group
# from dash.dash_table.FormatTemplate import FormatTemplate
import plotly.express as px
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
# 000 - Loading
####################################################################################################

# Read in external data into dash application
df = pd.read_csv('data/raw/final.csv')
df1 = df.copy()

# Read pickle file that can be obtained by running the first half or so of MLR.ipynb
MS_ser = pd.read_pickle('data/pickle/modNshap.P')

########################### Page 1 - Maps
#Create L1 dropdown options
repo_groups_l1_all = [
    {'label' : 'Number of Listings', 'value' : 'count'},
    {'label' : 'Average Listing Price', 'value' : 'avg_listing_price'},
    {'label': 'Maximum Listing Price', 'value': 'max_listing_price'},
    {'label' : 'Average Estimated Price', 'value' : 'avg_price'},
    {'label' : 'Average Listing Price', 'value' : 'max_price'},
    {'label': 'Average Amount Undervalued According to Model', 'value': 'avg_differential'},
    {'label': 'Maximum Amount a Listing is Undervalued According to Model', 'value': 'max_differential'},
    {'label' : 'Model Score Over Region', 'value' : 'max_score'},
    ]
########################### Page 2 - Analytics
# Model group L1 options
crossfilter_model_options = [
    {'label': 'Multiple Linear Regression: All features', 'value': 'MLR_full'},
    {'label': 'Multiple Linear Regression: House features', 'value': 'MLR_house'},
    {'label': 'Multiple Linear Regression: Regional features', 'value': 'MLR_regional'}
    ]

# Resolution group L2 options
crossfilter_resolution_options = [
    {'label': 'State', 'value': 'state'},
    {'label': 'County', 'value': 'county'},
    {'label': 'Zip code', 'value': 'zipcode'},
    ]
########################### Page 2 - Analytics

#####################
# Header with logo
def get_header():

    header = html.Div([

        html.Div([], className = 'col-2'), #Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children='Real Estate Analytics Dashboard',
                    style = {'textAlign' : 'center'}
            )],
            className='col-8',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = app.get_asset_url('logo_001c.png'),
                    height = '43 px',
                    width = 'auto')
            ],
            className = 'col-2',
            style = {
                    'align-items': 'center',
                    'padding-top' : '1%',
                    'height' : 'auto'})

        ],
        className = 'row',
        style = {'height' : '4%',
                'background-color' : corporate_colors['superdark-green']}
        )

    return header

#####################
# Nav bar
def get_navbar(p = 'page1'):

    navbar_page1 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Map',
                        style = navbarcurrentpage),
                href='/apps/page1'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Analytics'),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Interpret'),
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_page2 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Map'),
                href='/apps/page1'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Analytics',
                        style = navbarcurrentpage),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Interpret'),
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_page3 = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Map',),
                href='/apps/page1'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Analytics'),
                href='/apps/page2'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Interpret',
                        style = navbarcurrentpage),
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    if p == 'page1':
        return navbar_page1
    elif p == 'page2':
        return navbar_page2
    else:
        return navbar_page3

#####################
# Empty row

def get_emptyrow(h='45px'):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className = 'col-12')
    ],
    className = 'row',
    style = {'height' : h})

    return emptyrow

####################################################################################################
# 001 - SALES
####################################################################################################

page1 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page1'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row

        html.Div([ # External 12-column

            html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                #Filter pt 1
                html.Div([
                ],
                className = 'col-4'), # Filter part 1

                #Filter pt 2
                html.Div([

                    html.Div([ ## select granularity
                        html.H6(
                                'Look at ',
                                style = {'color': corporate_colors['superdark-green'],
                                'margin-top': '10px'}
                                ),
                        dcc.RadioItems( 
                            id = 'which_json',
                            options = [
                                {'label': 'Zip codes', 'value': 'zipcode'},
                                {'label': 'Counties', 'value': 'county'}
                            ],
                            value = 'county',
                            style = {
                                'font-size': '11px'
                            }
                        ),
                        ],
                        style = {
                            'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5
                        }
                    ),
                    html.Div([ ## select model
                        html.H6(
                                'Model selection',
                                style = {'color': corporate_colors['superdark-green'],
                                'margin-top': '10px'}
                                ),
                        dcc.Dropdown(
                            id = 'map-model',
                            options = crossfilter_model_options,
                            value = 'MLR_full',
                            multi = False,
                            style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                        ),
                        ],
                        style = {
                            'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5
                        }
                    ),    
                    html.Div([ ## Select feature
                        html.H6(
                                'Select feature',
                                style = {'color': corporate_colors['superdark-green'],
                                'margin-top': '10px'}
                                ),
                        html.Div([
                            dcc.Dropdown(id = 'reporting-groups-l1dropdown-sales',
                                options = repo_groups_l1_all,
                                # Default value when loading
                                value = 'count',
                                # Permit user to select only one option at a time
                                multi = False,
                                # Default message in dropdown before user select options
                                # placeholder = "Select " +sales_fields['reporting_group_l1']+ " (leave blank to include all)",
                                style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                        ],
                        style = {
                            'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5
                        }
                    ),    
                ],
                className = 'col-4'), # Filter part 2

                html.Div([
                ],
                className = 'col-2') # Blank 2 columns


            ],
            className = 'row') # Internal row

        ],
        className = 'col-12',
        style = filterdiv_borderstyling) # External 12-column

    ],
    className = 'row sticky-top'), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ # External 10-column

            # Maps and bar charts
            html.Div(
            id='ga_line',
            children = [
                dcc.Graph(
                    id='ga-map', 
                    #figure=fig_ga, 
                    config={'scrollZoom': True},                  
                    ),       
                ], 
            ), 

        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External 10-column

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row

])

####################################################################################################
# 002 - Page 2
####################################################################################################

page2 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page2'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row

        html.Div([ # External 12-column

            html.Div([ # Internal row

                #Internal columns
                html.Div([
                ],
                className = 'col-2'), # Blank 2 columns

                #Filter pt 1
                html.Div([

                    html.Div([
                        #Model group selection L1
                        html.H5(
                            children='Model:',
                            style = {'text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        html.Div([
                            dcc.Dropdown(id = 'crossfilter-model',
                                options = crossfilter_model_options,
                                value = 'MLR_full',
                                multi = False,
                                style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                ),
                            dcc.Dropdown(id = 'select-target',
                                options = [{'label': 'Price', 'value':'price'},
                                    {'label': 'Cap rate', 'value':'caprate'}
                                    ],
                                value = 'price',
                                multi = False,
                                style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                ),
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                        #Resolution group selection L2
                        html.H5(
                            children='Resolution:',
                            style = {'text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        html.Div([
                            dcc.Dropdown(id = 'crossfilter-resolution',
                                options = crossfilter_resolution_options,
                                value = 'state',
                                multi = False,
                                style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5})

                ],
                className = 'col-3'), # Filter part 1

                # ilter pt 2
                html.Div([

                    html.Div([
                        html.H5(
                            children='Feature:',
                            style = {'text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        # list of features
                        html.Div(id = 'feat_list', children = [
                            dcc.Dropdown(id = 'crossfilter-feature',
                                value = 'square_footage',
                                style = {'font-size': '1d3px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                        html.H5(
                            children='Region:',
                            style = {'text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        # This list is designed to be dynamically populated by callbacks from 'crossfilter-resolution'
                        html.Div(id = 'reso_list', children = [
                            dcc.Dropdown(id = 'filter-dropdown',
                            value = 'Georgia',
                                style = {'font-size': '13px', 'color' : corporate_colors['medium-blue-grey'], 'white-space': 'nowrap', 'text-overflow': 'ellipsis'}
                                )
                            ], 
                            style = {'width' : '70%', 'margin-top' : '5px'})
                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5})

                ],
                className = 'col-3'), # Filter part 2

                # Filter pt 3
                html.Div([

                    html.Div([
                        # Color gradient scheme selection L1
                        html.H5(
                            children='Color Gradient Scheme:',
                            style = {'text-align' : 'left', 'color' : corporate_colors['medium-blue-grey']}
                        ),
                        html.Div([
                            dcc.RadioItems(
                                id='gradient-scheme',
                                options=[
                                    {'label': 'Orange to Red', 'value': 'OrRd'},
                                    {'label': 'Viridis', 'value': 'Viridis'},
                                    {'label': 'Plasma', 'value': 'Plasma'}
                                ],
                                value='Plasma',
                                labelStyle={'float': 'left', 'display': 'inline-block', 'margin-right': 10}
                                )
                            ],
                            style = {'width' : '70%', 'margin-top' : '5px'}),
                    ],
                    style = {'margin-top' : '10px',
                            'margin-bottom' : '5px',
                            'text-align' : 'left',
                            'paddingLeft': 5})

                ],
                className = 'col-3'), # Filter part 3

                html.Div([
                ],
                className = 'col-1') # Blank 2 columns


            ],
            className = 'row') # Internal row

        ],
        className = 'col-12',
        style = filterdiv_borderstyling) # External 12-column

    ],
    className = 'row sticky-top'), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ 

            html.Div([ # Internal row

                ## Scatter plot 
                html.Div([
                    dcc.Graph(
                        id='scatter-plot',
                        # Update graph on click
                        clickData={'points': [{'customdata': 0}]},
                        style = {'width': '90%'}
                        )
                ],
                className = 'col-8'),

                ## SHAP values  
                html.Div([
                    html.Img(
                        id='shap-bee', 
                        style = {'width': '100%', 'height': '80%'}
                        )
                ],
                className = 'col-4'),

            ],
            className = 'row'), # Internal row

            html.Div([ # Internal row

                ##  Bar charts
                html.Div([
                    dcc.Graph(
                        id='point-plot'
                        )
                ],
                className = 'col-12'),

            ],
            className = 'row') # Internal row


        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External 10-column

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row

])

####################################################################################################
# 003 - Page 3
####################################################################################################

page3 = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('page3'),

    #####################
    #Row 3 : Filters
    html.Div([ # External row
        html.Div([ # External 12-col row
            html.Div([
                html.Div([], className='col-2'),
                html.Div([ ## Slider, radio, model
                    html.Div([
                        html.Div([ ## accuracy slider
                            html.Div(
                                id = 'model-accuracy-statement',
                                children = [],
                                style={'text-align':'center'}
                            ),
                            dcc.Slider( 
                                id='acc-cutoff',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0,
                                marks = {0: {'label': '0', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.1 : {'label': '0.1', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.2 : {'label':'0.2', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.3 : {'label':'0.3', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.4 : {'label':'0.4', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.5 : {'label':'0.5', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.6 : {'label':'0.6', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.7 : {'label':'0.7', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.8 : {'label':'0.8', 'style':{'color':corporate_colors['superdark-green']}},
                                        0.9: {'label':'0.9', 'style':{'color':corporate_colors['superdark-green']}},
                                        1: {'label':'1', 'style':{'color':corporate_colors['superdark-green']}}
                                },
                            ),
                        ], style ={
                                'font-size': '11px',
                                'color' : corporate_colors['superdark-green'],
                                'white-space': 'nowrap',
                                'text-overflow': 'ellipsis',
                                'margin-top': '5px',
                                'display':'inline-block',
                                'width' : '40%'
                                }),
                        html.Div([ ## Model selection
                            html.Div([html.H6(
                                'Select model',
                                style = {'color': corporate_colors['superdark-green']})]),
                            dcc.Dropdown(
                                id = 'choose-model',
                                options = [
                                    {'label': 'Multiple Linear Regression: All Features', 'value': 'MLR_full'},
                                    {'label': 'Multiple Linear Regression: House Features', 'value': 'MLR_house'},
                                    {'label': 'Multiple Linear Regression: Hegional Features', 'value': 'MLR_regional'}
                                ],
                                value = 'MLR_full',
                                style ={
                                'font-size': '11px',
                                'margin-top': '5px',
                                'margin-bottom': '10px',
                                'width': '180%',
                                'white-space': 'nowrap',
                                #'display': 'inline-block'
                                }
                            ),
                            dcc.Dropdown( ## Target seleection
                                id = 'bar-options',
                                options = [
                                    {'label': 'Average price', 'value': 'avg_price'},
                                    {'label': 'Maximum price', 'value' : 'max_price'},
                                    {'label': 'Average cap rate', 'value': 'avg_caprate'},
                                    {'label': 'Maximum cap rate', 'value': 'max_caprate'},
                                    {'label': 'Average undervaluement', 'value': 'avg_differential'},
                                    {'label': 'Maximum undervaluement', 'value': 'max_differential'},
                                ],
                                value = 'avg_price',
                                style = {
                                    'font-size': '11px',
                                    'margin-top': '5px',
                                    'margin-bottom': '10px',
                                    'width': '180%',
                                    'white-space': 'nowrap',
                                }
                            ),
                        ], style = {'display': 'inline-block',
                                    'paddingLeft': '5%',
                                    }
                        ),
                        html.Div([
                            html.Div([
                                #html.H6('Select interpreter',
                                 #   style= {
                                 #   'color': corporate_colors['superdark-green']
                                 #   }
                                #)
                            ],
                                style = {
                                    'text-align': 'center',
                                    'margin-top': '10px'
                                }
                            ),
                            #dcc.RadioItems(
                            #    id = 'interpretation-type',
                            #    options = [
                            #        {'label': 'SHAP', 'value': 'shap'},
                            #        {'label': 'LIME', 'value': 'lime'}
                            #    value = 'shap',
                            #    style = {
                            #        'font-size': '11px',
                            #        #'display': 'inline-block',
                            #    }
                            #),
                            html.H6(
                                'Look at the best',
                                style = {'color': corporate_colors['superdark-green'],
                                'margin-top': '10px'}
                                ),
                            dcc.RadioItems( ## granularity selection
                                id = 'granularity',
                                options = [
                                    {'label': 'Zip codes', 'value': 'zipcode'},
                                    {'label': 'Counties', 'value': 'county'}
                                ],
                                value = 'zipcode',
                                style = {
                                    'font-size': '11px'
                                }
                            )
                        ], style = {
                            'margin-top' : '5px',
                            'color': corporate_colors['superdark-green'],
                            'display': 'inline-block',
                            'paddingLeft': '25%'
                            }
                        ),
                    ],  style = filterdiv_borderstyling
                    )
                ], className = 'col-12'),
            ])
        ])
    ]), # External row

    #####################
    #Row 4
    get_emptyrow(),

    #####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ 

            html.Div([ # Internal row
                # Upper bar chart 
                html.Div([
                    html.H5('The top regions'),
                    dcc.Graph(id = 'top-bars',
                        style = {'width': '90%'}
                        )
                ],
                className = 'col-8'),
                # LIME/SHAP values here  
                html.Div(id = 'limeshap',
                    children = [
                    html.H6('Select an address to interpret')
                    ],
                className = 'col-4'
                ),

            ],
            className = 'row'), # Internal row

            html.Div([ # Internal row

                # Lower bar chart  
                html.Div([
                    html.H5('Select an address'),
                    dcc.Graph(id = 'lower-bars'
                        )
                ],style = {'paddingTop' : '10px'},
                className = 'col-12'),

            ],
            className = 'row') # Internal row


        ],
        className = 'col-10',
        style = externalgraph_colstyling), # External 10-column

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

    ],
    className = 'row',
    style = externalgraph_rowstyling
    ), # External row

])



