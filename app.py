import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import plotly
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)
CORS(app)

# Load dataset
data = pd.read_csv("adverse_events.csv")

# Helper to preprocess dates
def preprocess_dates(data):
    data['Date of Event'] = pd.to_datetime(data['Date of Event'], errors='coerce')
    data['Manufacturer Aware Date'] = pd.to_datetime(data['Manufacturer Aware Date'], errors='coerce')
    return data.dropna(subset=['Date of Event', 'Manufacturer Aware Date'])

# 1. Forecast Event Frequency (Graph 1)
@app.route('/forecast', methods=['GET'])
def forecast_events():
    malfunction_data = data[data['Event Type'] == 'M']
    malfunction_data.loc['Date of Event'] = pd.to_datetime(malfunction_data['Date of Event'], errors='coerce')
    malfunction_data = malfunction_data.dropna(subset=['Date of Event'])
    malfunction_data = malfunction_data.set_index('Date of Event')
    time_series_data = malfunction_data.resample('ME').size()

    model = ExponentialSmoothing(time_series_data, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(steps=12).tolist()

    time_periods = list(range(1, len(forecast) + 1))

    trace1 = {
        'x': time_periods,
        'y': forecast,
        'mode': 'lines+markers',
        'name': 'Forecast',
        'line': {'color': 'blue'}
    }

    trace2 = {
        'x': list(range(1, len(time_series_data) + 1)),
        'y': time_series_data.tolist(),
        'mode': 'lines+markers',
        'name': 'Historical',
        'line': {'color': 'red'}
    }

    layout = {
        'title': 'Event Forecast vs Historical Data',
        'xaxis': {'title': 'Time Period', 'showgrid': True},
        'yaxis': {'title': 'Event Frequency', 'showgrid': True},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace1, trace2], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

# 2. Recommendations (Graph 2)

@app.route('/recommendations', methods=['GET'])
def recommend_resolutions():
    event_type = request.args.get('event_type')

    if not event_type:
        return jsonify({'error': 'Event type is required'}), 400

    # Filter and prepare resolution data
    resolution_data = data[['Event Type', 'Device Problem Codes']].dropna()
    if event_type not in resolution_data['Event Type'].values:
        return jsonify({'error': f'Event type {event_type} not found in data'}), 404

    # Encoding event type and device problem codes
    resolution_data['Event Type Encoded'] = resolution_data['Event Type'].astype('category').cat.codes
    resolution_data['Device Problem Encoded'] = resolution_data['Device Problem Codes'].astype('category').cat.codes

    # Fit the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=3).fit(resolution_data[['Event Type Encoded', 'Device Problem Encoded']])

    # Get the encoded event type
    event_type_encoded = resolution_data[resolution_data['Event Type'] == event_type]['Event Type Encoded'].iloc[0]
    distances, indices = nbrs.kneighbors([[event_type_encoded, 0]])

    # Get recommendations (Device Problem Codes)
    recommendations = resolution_data.iloc[indices[0]]['Device Problem Codes']

    # Create the Plotly graph
    trace = {
        'x': recommendations.tolist(),
        'y': [1] * len(recommendations),  # Dummy y-axis for simple bar plot
        'type': 'bar',
        'name': 'Recommendations'
    }

    layout = {
        'title': f'Recommendations for Event Type: {event_type}',
        'xaxis': {'title': 'Device Problem Codes'},
        'yaxis': {'visible': False},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify({'plot': graph_json})


# 3. Average Turnaround Time (Graph 3)
@app.route('/turnaround', methods=['GET'])
def turnaround_time():
    processed_data = preprocess_dates(data)
    processed_data['Turnaround Time (days)'] = (processed_data['Manufacturer Aware Date'] - processed_data['Date of Event']).dt.days
    avg_turnaround = processed_data['Turnaround Time (days)'].mean()

    trace = {
        'x': ['Average Turnaround Time'],
        'y': [avg_turnaround],
        'type': 'bar',
        'name': 'Turnaround Time'
    }

    layout = {
        'title': 'Average Turnaround Time (Days)',
        'xaxis': {'title': 'Metric'},
        'yaxis': {'title': 'Days'},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

# 4. Manufacturer Complaints Scatter Plot (Graph 4)
@app.route('/complaints', methods=['GET'])
def manufacturer_complaints():
    complaints = data['Manufacturer Name'].value_counts()
    trace = {
        'x': complaints.index.tolist(),
        'y': complaints.values.tolist(),
        'mode': 'markers',
        'type': 'scatter',
        'name': 'Manufacturer Complaints',
        'marker': {'color': 'rgba(0, 0, 255, 0.5)', 'size': 12}
    }

    layout = {
        'title': 'Complaints by Manufacturer',
        'xaxis': {'title': 'Manufacturer'},
        'yaxis': {'title': 'Number of Complaints'},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

# 5. Monthly Submitted vs Resolved Events (Graph 5)
@app.route('/monthly-events', methods=['GET'])
def monthly_events():
    submitted = data.set_index('Date of Event').resample('M').size()
    resolved = data.set_index('Manufacturer Aware Date').resample('M').size()

    trace1 = {
        'x': submitted.index.tolist(),
        'y': submitted.tolist(),
        'mode': 'lines',
        'name': 'Submitted',
        'line': {'color': 'blue'}
    }

    trace2 = {
        'x': resolved.index.tolist(),
        'y': resolved.tolist(),
        'mode': 'lines',
        'name': 'Resolved',
        'line': {'color': 'red'}
    }

    layout = {
        'title': 'Monthly Submitted vs Resolved Events',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': 'Event Count'},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace1, trace2], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

# 6. Events by Manufacturer Over Time (Graph 6)
@app.route('/events-by-manufacturer', methods=['GET'])
def events_by_manufacturer():
    events_by_manufacturer = data.groupby([data['Date of Event'].dt.to_period('M'), 'Manufacturer Name']).size().unstack().fillna(0)

    traces = []
    for manufacturer in events_by_manufacturer.columns:
        trace = {
            'x': events_by_manufacturer.index.astype(str),
            'y': events_by_manufacturer[manufacturer].tolist(),
            'mode': 'lines',
            'name': manufacturer
        }
        traces.append(trace)

    layout = {
        'title': 'Events by Manufacturer Over Time',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': 'Event Count'},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': traces, 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

# 7. Forecast Deviation (Graph 7)
@app.route('/forecast-deviation', methods=['GET'])
def forecast_deviation():
    malfunction_data = data[data['Event Type'] == 'M']
    malfunction_data['Date of Event'] = pd.to_datetime(malfunction_data['Date of Event'], errors='coerce')
    malfunction_data = malfunction_data.dropna(subset=['Date of Event'])
    malfunction_data = malfunction_data.set_index('Date of Event')
    time_series_data = malfunction_data.resample('M').size()

    model = ExponentialSmoothing(time_series_data, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(steps=12).tolist()

    deviation = [abs(actual - forecasted) for actual, forecasted in zip(time_series_data[-12:], forecast)]

    trace = {
        'x': list(range(1, len(deviation) + 1)),
        'y': deviation,
        'mode': 'bar',
        'name': 'Deviation',
        'marker': {'color': 'red'}
    }

    layout = {
        'title': 'Forecast Deviation (Actual vs Forecast)',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': 'Deviation'},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(255, 255, 255, 0.8)',
        'font': {'size': 14}
    }

    fig = {'data': [trace], 'layout': layout}
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'plot': graph_json})

if __name__ == '__main__':
    app.run(debug=True)
