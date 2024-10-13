document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
        displayChart(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}


function displayChart(data) {
    // Input: data (object) - contains the following keys:
    //        - documents (list) - list of documents
    //        - indices (list) - list of indices
    //        - similarities (list) - list of similarities

    // Get the context of the canvas element where we will draw the chart
    var ctx = document.getElementById('similarity-chart').getContext('2d');

    // If there was a previous chart, destroy it first to avoid overlap
    if (window.similarityChart) {
        window.similarityChart.destroy();
    }

    // Prepare the labels and data for the chart
    var labels = data.indices.map(function(index, i) {
        return `Doc ${index + 1}`; // You can customize the label format
    });
    var similarityScores = data.similarities;

    // Create a new bar chart
    window.similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels, // Document indices as labels
            datasets: [{
                label: 'Cosine Similarity',
                data: similarityScores, // Similarity values
                backgroundColor: 'rgba(75, 192, 192, 0.2)', // Light teal background
                borderColor: 'rgba(75, 192, 192, 1)', // Teal border
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true, // Ensure y-axis starts at zero
                    title: {
                        display: true,
                        text: 'Cosine Similarity'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Document Index'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false // Hide the legend
                }
            }
        }
    });
}

