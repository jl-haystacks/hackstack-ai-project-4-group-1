############################################################################################
########################################## IMPORTS #########################################
############################################################################################

# Classic libraries
import os
import numpy as np
import pandas as pd

# Logging information
import logging
import logzero
from logzero import logger

# Dash imports
import dash
from dash import dcc
from dash import html

# Custom function
import scripts.utils_haystacks as f

############################################################################################
############################## PARAMETERS and PRE-COMPUTATION ##############################
############################################################################################

# Load pre computed data
ga = f.load_pickle('ga_info.p')

# Deployment inforamtion
PORT = 8050

############################################################################################
########################################## APP #############################################
############################################################################################

# Creating app
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

# Associating server
server = app.server
app.title = 'Georgia - Data on House Sales'
app.config.suppress_callback_exceptions = True

############################################################################################
######################################### LAYOUT ###########################################
############################################################################################

app.layout = html.Div(
    children=[

        # HEADER
        html.Div(
            className="header",
            children=[
                html.H1("Georgia - Data on House Sales", className="header__text"),
                # html.Span('(Last update: {})'.format(world['last_date'])),
                # html.Hr(),
            ],
        ),

        # CONTENT
        html.Section([

            # Line 1 : MAP - GA
            html.Div(
                id='ga_line',
                children = [
                    dcc.Graph(
                        id='ga_map', 
                        figure=ga['figure'], 
                        config={'scrollZoom': True}
                        ),         
                    ], 
                ), 
            html.Br(),
            ], 
        ),
    ],
)

############################################################################################
######################################### RUNNING ##########################################
############################################################################################

if __name__ == "__main__":
    
    # Display app start
    logger.error('*' * 80)
    logger.error('App initialisation')
    logger.error('*' * 80)

    # Starting flask server
    app.run_server(debug=True, port=PORT)