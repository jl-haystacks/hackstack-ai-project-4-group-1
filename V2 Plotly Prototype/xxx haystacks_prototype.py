# Initialize dash application by importing necessary packages
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Define external stylesheets and apply them
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Prototype Real Estate Dashboard (Haystacks)', external_stylesheets=[external_stylesheets])

# Read in external data into dash application
data = pd.read_csv("Data/clean_Ames_HousePrice.csv")

# Define features to be utilized for generating scatter plots
features = ['GrLivArea', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageArea']

# # Create scatterplot figures
# fig1 = px.scatter(data, x="GrLivArea", y="SalePrice")
# fig2 = px.scatter(data, x="LotArea", y="SalePrice", log_x=True)

#Define layout of dash application
app.layout = html.Div([
    # Title and caption
    html.Div(
        children=[
            html.H1(
                children="Real Estate Analytics"
                ),
            html.P(
                children="Analyze the final sale price of homes"
                " based on quantitative and categorical features.",
            ),
        # dcc.Graph(
        #     figure = fig1
        #     # figure={
        #     #     "data": [
        #     #         {
        #     #             "x": data["GrLivArea"],
        #     #             "y": data["SalePrice"],
        #     #             "type": "scatter",
        #     #         },
        #     #     ],
        #     #     "layout": {"title": "Ground Living Area vs Sale Price"},
        #     # },
        # ),
        # dcc.Graph(
        #     figure = fig2
        #     # figure={
        #     #     "data": [
        #     #         {
        #     #             "x": data["LotArea"],
        #     #             "y": data["SalePrice"],
        #     #             "type": "scatter",
        #     #         },
        #     #     ],
        #     #     "layout": {"title": "Lot Area vs Sale Price"},
        #     # },
        # ),
        ]
    ),
    html.Div([
        html.Div([
            # Feature selection caption
            html.Div([
                html.Label('Feature selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),
            # Radio button for color schema of scatter plots
            html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                        {'label': 'Orange to Red', 'value': 'OrRd'},
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'}
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            # Feature selection dropdown menu based on those defined at the beginning
            dcc.Dropdown(
                id='crossfilter-feature',
                options=[{'label': i, 'value': i} for i in features],
                value='None',
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
        
        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),
    # Scatter plot of correlation between selected feature and sale price
    html.Div([

        dcc.Graph(
            id='scatter-plot',
            hoverData={'points': [{'customdata': 0}]}
        )

    ], style={'width': '100%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
    
    # html.Div([
    #     dcc.Graph(id='point-plot'),
    # ], style={'display': 'inline-block', 'width': '100%'}),

    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)

# First callback to coordinate scatterplot with feature and color gradient schema
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value')
    ]
)
def update_graph(feature, gradient):

    cols = data[feature].values #df[feature].values
    sizes = [np.max([max_val/10, x]) for x in df[feature].values]
    hover_names = []
    for ix, val in zip(df.index.values, df[feature].values):
        hover_names.append(f'Customer {ix:03d}<br>{feature} value of {val}') 

    fig = px.scatter(
        data,
        x=data[f'{model.lower()}_x'],
        y=df[f'{model.lower()}_y'],
        color=cols,
        size=sizes,
        opacity=0.8,
        hover_name=hover_names,
        hover_data=features,
        template='plotly_dark',
        color_continuous_scale=gradient,
    )

    fig.update_traces(customdata=df.index)

    fig.update_layout(
        coloraxis_colorbar={'title': f'{feature}'},
        # coloraxis_showscale=False,
        legend_title_text='Spend',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark'
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def create_point_plot(df, title):

    fig = go.Figure(
        data=[
            go.Bar(name='Average', x=features, y=df_average.values, marker_color='#c178f6'),
            go.Bar(name=title, x=features, y=df.values, marker_color='#89efbd')
        ]
    )

    fig.update_layout(
        barmode='group',
        height=225,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
        template='plotly_dark'
    )

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type="log", range=[0,5])

    return fig


@app.callback(
    dash.dependencies.Output('point-plot', 'figure'),
    [
        dash.dependencies.Input('scatter-plot', 'hoverData')
    ]
)
def update_point_plot(hoverData):
    index = hoverData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)

# The final two critical lines of code to RUN the application
if __name__ == "__main__":
    app.run_server(debug=True)