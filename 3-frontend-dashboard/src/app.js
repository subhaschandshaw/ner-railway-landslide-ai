// Initialize the map centered on the Lumding-Badarpur Railway Section
const map = L.map('map').setView([25.15, 93.15], 11);

// Add Esri World Imagery (Satellite) tiles
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
}).addTo(map);

// Define the railway track path (approximate coordinates for demonstration)
const trackCoordinates = [
    [25.10, 93.10],
    [25.13, 93.13],
    [25.15, 93.15],
    [25.18, 93.18],
    [25.20, 93.20]
];

// Draw the track polyline on the map
const railwayTrack = L.polyline(trackCoordinates, {
    color: 'green',
    weight: 6,
    opacity: 0.8
}).addTo(map);

// Add a marker for the primary sensor node
const sensorMarker = L.marker([25.15, 93.15]).addTo(map)
    .bindPopup('<b>Primary Sensor Node</b><br>Lumding-Badarpur Hill Section')
    .openPopup();

// DOM Elements
const riskScoreEl = document.getElementById('risk-score');
const alertLevelEl = document.getElementById('alert-level');
const timestampEl = document.getElementById('timestamp');
const riskCard = document.getElementById('risk-card');
const alertCard = document.getElementById('alert-card');
const smsAlert = document.getElementById('sms-alert');

// Function to fetch real-time predictions from the ML Engine
async function fetchRiskScore() {
    try {
        // Fetch from ML Engine API (Ensure it is running on port 8001 and has CORS configured, or via relative path if proxied)
        // Since we are running microservices, we will point directly to localhost for development
        const response = await fetch('http://localhost:8001/api/predict-realtime');
        
        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateDashboard(data);
        }
    } catch (error) {
        console.error("Failed to fetch risk score:", error);
        riskScoreEl.innerText = 'ERR';
        alertLevelEl.innerText = 'OFFLINE';
    }
}

// Function to update the UI based on the fetched data
function updateDashboard(data) {
    const score = data.risk_score_percent;
    const alertLevel = data.alert_level; // SAFE, WARNING, CRITICAL
    
    // Update Text
    riskScoreEl.innerText = `${score}%`;
    alertLevelEl.innerText = alertLevel;
    timestampEl.innerText = `Last Updated: ${new Date(data.timestamp).toLocaleString()}`;
    
    // Reset styles
    riskCard.className = 'bg-white p-6 rounded-lg shadow-sm border-l-4 transition-colors duration-300';
    alertCard.className = 'bg-white p-6 rounded-lg shadow-sm border-l-4 transition-colors duration-300';
    smsAlert.classList.add('hidden');
    
    // Apply styling based on alert level
    if (alertLevel === 'CRITICAL' || score >= 80) {
        // Red / Critical State
        riskCard.classList.add('border-red-500');
        alertCard.classList.add('border-red-500');
        alertLevelEl.classList.add('text-red-600');
        
        // Change track color to red
        railwayTrack.setStyle({ color: 'red' });
        
        // Trigger automated SMS visual alert
        smsAlert.classList.remove('hidden');
        
    } else if (alertLevel === 'WARNING' || score >= 40) {
        // Yellow / Warning State
        riskCard.classList.add('border-yellow-500');
        alertCard.classList.add('border-yellow-500');
        alertLevelEl.classList.add('text-yellow-600');
        
        // Change track color to orange
        railwayTrack.setStyle({ color: 'orange' });
        
    } else {
        // Green / Safe State
        riskCard.classList.add('border-green-500');
        alertCard.classList.add('border-green-500');
        alertLevelEl.classList.add('text-green-600');
        
        // Change track color back to green
        railwayTrack.setStyle({ color: 'green' });
    }
}

// Start polling the API every 5 seconds
fetchRiskScore();
setInterval(fetchRiskScore, 5000);

