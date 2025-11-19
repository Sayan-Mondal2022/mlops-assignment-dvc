document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        cpu_request: parseFloat(document.getElementById("cpu_request").value),
        mem_request: parseFloat(document.getElementById("mem_request").value),
        cpu_limit: parseFloat(document.getElementById("cpu_limit").value),
        mem_limit: parseFloat(document.getElementById("mem_limit").value),
        runtime_minutes: parseInt(document.getElementById("runtime_minutes").value)
    };

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("resultText").textContent = 
        "Predicted output: " + JSON.stringify(result.prediction);

    document.getElementById("resultBox").classList.remove("hidden");
});
