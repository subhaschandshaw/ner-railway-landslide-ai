// Initialize the map centered on the Lumding-Badarpur Railway Section
const map = L.map('map', {
    fullscreenControl: true,
    fullscreenControlOptions: {
        position: 'topleft'
    }
}).setView([25.3, 93.0], 9);

// Add Esri World Imagery (Satellite) tiles
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri'
}).addTo(map);

// Add Esri Reference Overlay (Roads & Labels) to make it look realistic
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19
}).addTo(map);

// Define the railway track path (approximate coordinates between Lumding and Badarpur)
const trackCoordinates = [
    [25.75, 93.17], // Lumding
    [25.45, 93.16], // Midpoint 1
    [25.15, 93.15], // Sensor Node (Hill Section)
    [25.00, 92.80], // Midpoint 2
    [24.89, 92.60]  // Badarpur
];

// Draw the track polyline on the map
const railwayTrack = L.polyline(trackCoordinates, {
    color: '#00ff00',
    weight: 5,
    opacity: 0.9,
    dashArray: '10, 10'
}).addTo(map);

// Add markers for the Stations
const lumdingMarker = L.marker([25.75, 93.17]).addTo(map).bindPopup('<b>Lumding Junction</b>');
const badarpurMarker = L.marker([24.89, 92.60]).addTo(map).bindPopup('<b>Badarpur Junction</b>');

// Add a pulse circle for the primary sensor node
const sensorRadius = L.circle([25.15, 93.15], {
    color: 'green',
    fillColor: '#00ff00',
    fillOpacity: 0.2,
    radius: 8000
}).addTo(map);

const sensorMarker = L.marker([25.15, 93.15]).addTo(map)
    .bindPopup('<b>Primary Sensor Node (Vulnerable Zone)</b><br>Active Monitoring')
    .openPopup();

// DOM Elements
const riskScoreEl = document.getElementById('risk-score');
const alertLevelEl = document.getElementById('alert-level');
const timestampEl = document.getElementById('timestamp');
const riskCard = document.getElementById('risk-card');
const alertCard = document.getElementById('alert-card');
const smsAlert = document.getElementById('sms-alert');
const demoBtn = document.getElementById('demo-btn');

const sensorRainEl = document.getElementById('sensor-rain');
const sensorSoilEl = document.getElementById('sensor-soil');
const sensorTempEl = document.getElementById('sensor-temp');
const eventLogEl = document.getElementById('event-log');

let isDemoMode = false;
let lastAlertLevel = null;

function logEvent(message, type = "INFO") {
    const time = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.innerHTML = `[${time}] <span class="${type === 'CRITICAL' ? 'text-red-500 font-bold' : (type === 'WARN' ? 'text-yellow-400' : 'text-green-400')}">${message}</span>`;
    eventLogEl.prepend(div);
    if(eventLogEl.children.length > 50) eventLogEl.removeChild(eventLogEl.lastChild);
}

// Demo Button Logic
demoBtn.addEventListener('click', () => {
    isDemoMode = true;
    demoBtn.innerText = "🚨 SIMULATING STORM...";
    demoBtn.classList.add('blink');
    logEvent("WARNING: STORM SIMULATION INITIATED BY USER", "WARN");
    
    // Dynamically zoom in on the vulnerable sensor zone
    map.flyTo([25.15, 93.15], 13, { animate: true, duration: 2 });
    logEvent("Map auto-zooming to primary vulnerable node...", "WARN");
    
    // Automatically turn off demo mode after 15 seconds
    setTimeout(() => {
        isDemoMode = false;
        demoBtn.innerText = "⚠️ Run Demo Scenario";
        demoBtn.classList.remove('blink');
        logEvent("Storm simulation ended. Returning to live telemetry.", "INFO");
        
        // Dynamically zoom out to full section view
        map.flyTo([25.3, 93.0], 9, { animate: true, duration: 2 });
    }, 15000);
});

// Function to fetch real-time predictions from the ML Engine
async function fetchRiskScore() {
    try {
        const url = isDemoMode 
            ? 'http://localhost:8001/api/predict-realtime?demo=true' 
            : 'http://localhost:8001/api/predict-realtime';
            
        const response = await fetch(url);
        
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
    
    // Update Sensors
    if(data.sensors) {
        sensorRainEl.innerText = `${data.sensors.rainfall_mm_hr} mm/hr`;
        sensorSoilEl.innerText = `${data.sensors.soil_moisture_pct} %`;
        sensorTempEl.innerText = `${data.sensors.temperature_C} °C`;
    }
    
    // Update Text
    riskScoreEl.innerText = `${score}%`;
    alertLevelEl.innerText = alertLevel;
    timestampEl.innerText = `Last Updated: ${new Date(data.timestamp).toLocaleString()}`;
    
    // Log state changes
    if (lastAlertLevel !== alertLevel) {
        logEvent(`Alert Level shifted from ${lastAlertLevel || 'BOOT'} to ${alertLevel}`, alertLevel);
        lastAlertLevel = alertLevel;
    }
    
    // Reset styles
    riskCard.className = 'bg-white p-6 rounded-lg shadow-sm border-l-4 transition-colors duration-300';
    alertCard.className = 'bg-white p-6 rounded-lg shadow-sm border-l-4 transition-colors duration-300';
    smsAlert.classList.add('hidden');
    
    // Apply styling based on alert level
    if (alertLevel === 'CRITICAL' || score >= 80) {
        riskCard.classList.add('border-red-500');
        alertCard.classList.add('border-red-500');
        alertLevelEl.className = 'text-3xl font-bold text-red-600';
        
        railwayTrack.setStyle({ color: 'red' });
        sensorRadius.setStyle({ color: 'red', fillColor: '#ff0000' });
        smsAlert.classList.remove('hidden');
        
    } else if (alertLevel === 'WARNING' || score >= 40) {
        riskCard.classList.add('border-yellow-500');
        alertCard.classList.add('border-yellow-500');
        alertLevelEl.className = 'text-3xl font-bold text-yellow-600';
        
        railwayTrack.setStyle({ color: 'orange' });
        sensorRadius.setStyle({ color: 'orange', fillColor: '#ffa500' });
        
    } else {
        riskCard.classList.add('border-green-500');
        alertCard.classList.add('border-green-500');
        alertLevelEl.className = 'text-3xl font-bold text-green-600';
        
        railwayTrack.setStyle({ color: '#00ff00' });
        sensorRadius.setStyle({ color: 'green', fillColor: '#00ff00' });
    }
}

// Start polling the API every 5 seconds
fetchRiskScore();
setInterval(fetchRiskScore, 5000);

