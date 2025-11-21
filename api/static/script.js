// Configuration
const CONFIG = {
    backendUrl: window.location.origin, // Use same origin as the page
    endpoints: {
        predict: '/predict',
        health: '/health'
    }
};

// Utility functions
const utils = {
    showLoading(button) {
        button.dataset.originalText = button.textContent;
        button.innerHTML = '<span class="loading"></span> Predicting...';
        button.disabled = true;
    },

    hideLoading(button) {
        if (button.dataset.originalText) {
            button.textContent = button.dataset.originalText;
        }
        button.disabled = false;
    },

    showResult(message, isError = false) {
        const resultBox = document.getElementById("resultBox");
        const resultText = document.getElementById("resultText");
        
        resultText.innerHTML = message;
        resultBox.className = isError ? "result error" : "result";
        resultBox.classList.remove("hidden");
        
        // Scroll to result
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    },

    formatPredictions(result) {
        return `
            <div style="margin-bottom: 15px;">
                <strong style="color: #b5bd68;">Average Prediction:</strong> 
                <span style="font-family: 'Fira Code', monospace; font-size: 1.1em;">${result.avg_prediction} cores</span>
            </div>
            <div>
                <strong style="color: #81a2be;">Individual Models:</strong><br>
                <div style="margin-top: 8px; font-family: 'Fira Code', monospace;">
                    <span style="color: #de935f;">LightGBM</span>: ${result.predictions.lightgbm !== null ? result.predictions.lightgbm : 'N/A'}<br>
                    <span style="color: #b294bb;">Random Forest</span>: ${result.predictions.random_forest !== null ? result.predictions.random_forest : 'N/A'}<br>
                    <span style="color: #8abeb7;">XGBoost</span>: ${result.predictions.xgboost !== null ? result.predictions.xgboost : 'N/A'}
                </div>
            </div>
        `;
    }
};

// Health check
async function checkHealth() {
    try {
        const response = await fetch(CONFIG.endpoints.health);
        const health = await response.json();
        console.log('🔧 Server health:', health);
        
        if (health.status !== 'ready') {
            console.warn('Server models not fully loaded:', health);
        }
        return health;
    } catch (error) {
        console.error('Health check failed:', error);
        return null;
    }
}

// Main prediction function
async function handlePrediction(formData) {
    try {
        console.log("📤 Sending prediction request:", formData);

        const response = await fetch(CONFIG.endpoints.predict, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        console.log("📥 Received response:", result);

        if (response.ok && result.status === "success") {
            utils.showResult(utils.formatPredictions(result));
        } else {
            utils.showResult(`
                <strong>Error:</strong> ${result.error || 'Unknown error occurred'}<br>
                ${result.details ? `<small>${result.details}</small>` : ''}
            `, true);
        }

        return result;

    } catch (error) {
        console.error("💥 Prediction error:", error);
        utils.showResult(`
            <strong>Network Error:</strong> ${error.message}<br>
            <small>Please check your connection and try again.</small>
        `, true);
        return null;
    }
}

// Form submission handler
document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const button = this.querySelector("button");
    utils.showLoading(button);

    const formData = {
        cpu_request: parseFloat(document.getElementById("cpu_request").value),
        mem_request: parseFloat(document.getElementById("mem_request").value),
        cpu_limit: parseFloat(document.getElementById("cpu_limit").value),
        mem_limit: parseFloat(document.getElementById("mem_limit").value),
        runtime_minutes: parseInt(document.getElementById("runtime_minutes").value),
        controller_kind: document.getElementById("controller_kind").value
    };

    await handlePrediction(formData);
    utils.hideLoading(button);
});

// Initialize app when page loads
window.addEventListener('load', async () => {
    console.log('🚀 CPU Predictor App Initializing...');
    console.log('📍 Current URL:', window.location.href);
    console.log('🔗 Backend URL:', CONFIG.backendUrl);
    
    // Add code font to result area
    document.getElementById('resultText').classList.add('code-font');
    
    // Check server health
    await checkHealth();
});

// Add touch device detection for better mobile UX
if ('ontouchstart' in window || navigator.maxTouchPoints) {
    document.body.classList.add('touch-device');
}