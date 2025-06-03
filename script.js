document.getElementById("predictForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form from refreshing the page

    let formData = new FormData(this);

    fetch("/predict_sales", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            document.getElementById("predictionResult").innerText = "Predicted Sales: $" + data.prediction;
        } else {
            document.getElementById("predictionResult").innerText = "Error: " + data.error;
        }
    })
    .catch(error => console.error("Error:", error));
});
