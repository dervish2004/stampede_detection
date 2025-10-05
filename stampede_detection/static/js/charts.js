/**
 * Creates a configuration object for a Chart.js line chart.
 * This makes the main function cleaner and avoids repeating settings.
 * @param {string} label - The label for the dataset (e.g., 'PPSM (%)').
 * @param {string} yAxisLabel - The label for the Y-axis.
 * @param {string} borderColor - The color for the line.
 * @param {string} backgroundColor - The color for the area under the line.
 * @returns {object} A Chart.js options object.
 */
function createChartConfig(label, yAxisLabel, borderColor, backgroundColor) {
  return {
    type: 'line',
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: yAxisLabel }
        },
        x: {
          title: { display: true, text: 'Frame Number' }
        }
      },
      plugins: {
        legend: { display: false }
      }
    },
    data: {
      labels: [], // Will be populated later
      datasets: [{
        label: label,
        data: [], // Will be populated later
        borderColor: borderColor,
        backgroundColor: backgroundColor,
        tension: 0.1,
        fill: true,
      }]
    }
  };
}

/**
 * Main function to render all charts on the page.
 * @param {object} chartData - An object containing the data from the server.
 * Expected properties: labels, ppsmData, entropyData.
 */
function renderCharts(chartData) {
  const ppsmCtx = document.getElementById('ppsmChart');
  const entropyCtx = document.getElementById('entropyChart');

  if (!ppsmCtx || !entropyCtx) {
    console.error("Chart canvas elements not found!");
    return;
  }

  // --- Create and Render PPSM Chart ---
  const ppsmConfig = createChartConfig(
    'PPSM (%)',
    'Density (%)',
    'rgb(54, 162, 235)',
    'rgba(54, 162, 235, 0.5)'
  );
  ppsmConfig.data.labels = chartData.labels;
  ppsmConfig.data.datasets[0].data = chartData.ppsmData;
  new Chart(ppsmCtx, ppsmConfig);


  // --- Create and Render Entropy Chart ---
  const entropyConfig = createChartConfig(
    'Movement Entropy',
    'Entropy Value',
    'rgb(255, 99, 132)',
    'rgba(255, 99, 132, 0.5)'
  );
  entropyConfig.data.labels = chartData.labels;
  entropyConfig.data.datasets[0].data = chartData.entropyData;
  new Chart(entropyCtx, entropyConfig);
}