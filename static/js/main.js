// Global variables
let currentChart = null;

// Utility functions
function showSpinner() {
    const spinner = document.querySelector('.spinner-overlay');
    if (spinner) spinner.style.display = 'flex';
}

function hideSpinner() {
    const spinner = document.querySelector('.spinner-overlay');
    if (spinner) spinner.style.display = 'none';
}

function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Form handling
document.getElementById('upload-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('sequence-file');
    
    if (!fileInput?.files[0]) {
        showNotification('Please select a file to upload', 'error');
        return;
    }
    
    formData.append('file', fileInput.files[0]);
    showSpinner();
    
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
            showError(result.error);
            return;
        }
        
        displayResult(result);
        showNotification('Classification completed successfully');
    } catch (error) {
        showError('An error occurred while processing your request');
    } finally {
        hideSpinner();
    }
});

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    if (!resultDiv) return;

    const resultText = document.getElementById('result-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const alert = resultDiv.querySelector('.alert');
    
    resultDiv.style.display = 'block';
    
    if (result.is_edna) {
        resultText.textContent = 'This sequence is classified as eDNA';
        alert.classList.remove('alert-danger');
        alert.classList.add('alert-success');
    } else {
        resultText.textContent = 'This sequence is not classified as eDNA';
        alert.classList.remove('alert-success');
        alert.classList.add('alert-danger');
    }
    
    const confidence = (result.confidence * 100).toFixed(2);
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.textContent = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;
    
    resultDiv.classList.add('fade-in');
    
    // Update sequence viewer if available
    const sequenceViewer = document.getElementById('sequence-viewer');
    if (sequenceViewer && result.sequence) {
        sequenceViewer.textContent = result.sequence;
    }
}

function showError(message) {
    showNotification(message, 'error');
    
    const resultDiv = document.getElementById('result');
    if (!resultDiv) return;

    const alert = resultDiv.querySelector('.alert');
    const resultText = document.getElementById('result-text');
    
    resultDiv.style.display = 'block';
    alert.classList.remove('alert-success', 'alert-danger');
    alert.classList.add('alert-danger');
    resultText.textContent = message;
    
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    if (confidenceBar) confidenceBar.style.width = '0%';
    if (confidenceText) confidenceText.textContent = '';
}

// Performance chart initialization
document.addEventListener('DOMContentLoaded', () => {
    const chartCanvas = document.getElementById('trainingChart');
    if (!chartCanvas) return;

    const ctx = chartCanvas.getContext('2d');
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 50}, (_, i) => i + 1),
            datasets: [{
                label: 'Training Loss',
                data: Array.from({length: 50}, () => Math.random() * 0.01 + 0.008),
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }, {
                label: 'Validation Loss',
                data: Array.from({length: 50}, () => Math.random() * 0.01 + 0.015),
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Training History'
                },
                tooltip: {
                    enabled: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
});

// Database search functionality
const searchForm = document.getElementById('database-search');
if (searchForm) {
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        showSpinner();
        
        const searchInput = document.getElementById('search-input');
        try {
            const response = await fetch(`/search?q=${encodeURIComponent(searchInput.value)}`);
            const results = await response.json();
            updateSearchResults(results);
        } catch (error) {
            showNotification('Error searching database', 'error');
        } finally {
            hideSpinner();
        }
    });
}

function updateSearchResults(results) {
    const resultsContainer = document.getElementById('search-results');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = results.map(result => `
        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">${result.id}</h5>
                <p class="card-text"><small>Type: ${result.type}</small></p>
                <div class="sequence-viewer">${result.sequence}</div>
                <button class="btn btn-sm btn-primary mt-2" onclick="downloadSequence('${result.id}')">
                    Download
                </button>
            </div>
        </div>
    `).join('');
}