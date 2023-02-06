import dash
from dash import dcc
from dash import html

from app import app
from app import server
from layouts import page1, page2, page3
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/page1':
         return page1
    elif pathname == '/apps/page2':
         return page2
    elif pathname == '/apps/page3':
         return page3
    else:
        return page1 # This is the "home page"

if __name__ == '__main__':
    app.run_server(debug=False)
