import dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Georgia - Data on House Sales'
server = app.server
