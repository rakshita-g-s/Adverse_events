from flask import Flask, render_template
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

@app.route("/")
def index():
    # Example plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[10, 20, 15, 25],
        mode='lines+markers',
        name='Sample Line'
    ))
    fig.update_layout(title="Sample Graph")
    
    # Convert to JSON for front-end rendering
    graph_json = pio.to_json(fig)
    return render_template("index.html", graph_json=graph_json)

if __name__ == "__main__":
    app.run(debug=True)
