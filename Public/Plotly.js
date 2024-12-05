import React, { useState } from "react";
import Plot from "react-plotly.js";

const App = () => {
  const [graphData, setGraphData] = useState(null);

  const uploadCSV = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:5000/forecast", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setGraphData(data);
  };

  return (
    <div>
      <h1>Dynamic Graphs</h1>
      <input type="file" onChange={uploadCSV} />
      {graphData && (
        <Plot
          data={[
            {
              x: Object.keys(graphData.historical),
              y: Object.values(graphData.historical),
              mode: "lines",
              name: "Historical",
            },
            {
              x: Object.keys(graphData.forecast),
              y: Object.values(graphData.forecast),
              mode: "lines",
              name: "Forecast",
              line: { dash: "dot" },
            },
          ]}
          layout={{
            title: "Event Forecast",
            xaxis: { title: "Date" },
            yaxis: { title: "Event Count" },
          }}
        />
      )}
    </div>
  );
};

export default App;
