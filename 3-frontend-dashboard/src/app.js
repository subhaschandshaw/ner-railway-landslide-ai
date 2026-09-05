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
const activeAlertsEl = document.getElementById('active-alerts');
const activeAlertCountEl = document.getElementById('active-alert-count');
const alertHistoryEl = document.getElementById('alert-history');
const clearHistoryBtn = document.getElementById('clear-history-btn');

let isDemoMode = false;
let lastAlertLevel = null;
let activeAlert = null;
let alertHistory = JSON.parse(localStorage.getItem('landslide-alert-history') || '[]');

const alertLocation = 'Lumding-Badarpur Railway Section, Assam';
const alertCoordinates = '25.1500, 93.1500';

function logEvent(message, type = "INFO") {
    const time = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.innerHTML = `[${time}] <span class="${type === 'CRITICAL' ? 'text-red-500 font-bold' : (type === 'WARN' ? 'text-yellow-400' : 'text-green-400')}">${message}</span>`;
    eventLogEl.prepend(div);
    if(eventLogEl.children.length > 50) eventLogEl.removeChild(eventLogEl.lastChild);
}

function formatDateTime(value) {
    return new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

function buildAlertDetails(data) {
    const occurredAt = new Date().toISOString();
    return {
        id: `${occurredAt}-${Date.now()}`,
        level: data.alert_level,
        probability: data.risk_score_percent,
        occurredAt,
        location: alertLocation,
        coordinates: alertCoordinates,
        sensors: data.sensors || {}
    };
}

function renderActiveAlerts() {
    activeAlertCountEl.innerText = activeAlert ? '1 ACTIVE' : '0 ACTIVE';
    if (!activeAlert) {
        activeAlertsEl.innerHTML = '<p class="text-sm text-gray-500 py-4 text-center">No active alerts.</p>';
        return;
    }

    const sensorText = `Rain ${activeAlert.sensors.rainfall_mm_hr ?? '--'} mm/hr | Soil ${activeAlert.sensors.soil_moisture_pct ?? '--'}% | Temp ${activeAlert.sensors.temperature_C ?? '--'}°C`;
    const localFallbacks = activeAlert.level === 'CRITICAL'
        ? {
            assamese: 'লুমডিং-বদৰপুৰ ৰে\'লৱে ছেকচন, অসমত ভূমিস্খলনৰ গুৰুতৰ আশংকা আছে। ৰেলপথৰ ঢালৰ পৰা আঁতৰি থাকক আৰু চৰকাৰী নিৰ্দেশনা মানক।',
            bengali: 'আসামের লুমডিং-বদরপুর রেলওয়ে সেকশনে গুরুতর ভূমিধসের ঝুঁকি রয়েছে। রেলপথের ঢাল থেকে দূরে থাকুন এবং সরকারি নির্দেশনা মেনে চলুন।',
            hindi: 'असम के लुमडिंग-बदरपुर रेलवे सेक्शन में भूस्खलन का गंभीर खतरा है। रेलवे ढलान से दूर रहें और आधिकारिक निर्देशों का पालन करें।'
        }
        : {
            assamese: 'লুমডিং-বদৰপুৰ ৰে\'লৱে ছেকচন, অসমত ভূমিস্খলনৰ আশংকা আছে। ৰেলপথৰ ঢালৰ পৰা আঁতৰি থাকক আৰু চৰকাৰী নিৰ্দেশনা মানক।',
            bengali: 'আসামের লুমডিং-বদরপুর রেলওয়ে সেকশনে ভূমিধসের ঝুঁকি রয়েছে। রেলপথের ঢাল থেকে দূরে থাকুন এবং সরকারি নির্দেশনা মেনে চলুন।',
            hindi: 'असम के लुमडिंग-बदरपुर रेलवे सेक्शन में भूस्खलन का खतरा है। रेलवे ढलान से दूर रहें और आधिकारिक निर्देशों का पालन करें।'
        };
    activeAlertsEl.innerHTML = `
        <article class="border-l-4 border-red-500 bg-red-50 p-4 rounded-r">
            <div class="flex justify-between gap-3">
                <h3 class="font-bold text-red-800">Authority Alert · ${activeAlert.level}</h3>
                <time class="text-xs text-red-700">${formatDateTime(activeAlert.occurredAt)}</time>
            </div>
            <p class="text-sm text-gray-800 mt-2">Landslide probability: <strong>${activeAlert.probability}%</strong></p>
            <p class="text-xs text-gray-600 mt-1">${activeAlert.location} (${activeAlert.coordinates})</p>
            <p class="text-xs text-gray-600 mt-1">Realtime data: ${sensorText}</p>
        </article>
        <article class="border-l-4 border-amber-500 bg-amber-50 p-4 rounded-r">
            <div class="flex justify-between gap-3">
                <h3 class="font-bold text-amber-800">Local Community Alert</h3>
                <span class="text-xs text-amber-700">English · অসমীয়া · বাংলা · हिन्दी</span>
            </div>
            <p class="text-sm text-gray-800 mt-2"><strong>English:</strong> Landslide risk is ${activeAlert.level.toLowerCase()} near ${activeAlert.location}. Please stay away from the railway slope and follow official instructions.</p>
            <p class="text-sm text-gray-800 mt-1"><strong>অসমীয়া:</strong> ${activeAlert.localMessages?.assamese || localFallbacks.assamese}</p>
            <p class="text-sm text-gray-800 mt-1"><strong>বাংলা:</strong> ${activeAlert.localMessages?.bengali || localFallbacks.bengali}</p>
            <p class="text-sm text-gray-800 mt-1"><strong>हिन्दी:</strong> ${activeAlert.localMessages?.hindi || localFallbacks.hindi}</p>
        </article>`;
}

function renderHistory() {
    clearHistoryBtn.disabled = alertHistory.length === 0;
    clearHistoryBtn.classList.toggle('opacity-50', alertHistory.length === 0);
    alertHistoryEl.innerHTML = alertHistory.length === 0
        ? '<p class="text-sm text-gray-500 py-4 text-center">No alert history yet.</p>'
        : alertHistory.map(item => `
            <article class="border border-gray-200 p-3 rounded">
                <div class="flex justify-between gap-3"><strong class="text-gray-800">${item.level} · ${item.probability}%</strong><time class="text-xs text-gray-500">${formatDateTime(item.occurredAt)}</time></div>
                <p class="text-xs text-gray-600 mt-1">${item.location}</p>
            </article>`).join('');
}

async function requestLocalTranslations(alert) {
    try {
        const response = await fetch('http://localhost:8000/api/translate-alert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ level: alert.level, location: alert.location })
        });
        if (response.ok) alert.localMessages = (await response.json()).translations;
    } catch (error) {
        console.warn('Translation service unavailable; using local fallback.', error);
    }
    renderActiveAlerts();
}

function beginAlert(data) {
    activeAlert = buildAlertDetails(data);
    renderActiveAlerts();
    requestLocalTranslations(activeAlert);
}

function resolveAlert() {
    if (!activeAlert) return;
    alertHistory.unshift({ ...activeAlert, resolvedAt: new Date().toISOString() });
    alertHistory = alertHistory.slice(0, 50);
    localStorage.setItem('landslide-alert-history', JSON.stringify(alertHistory));
    activeAlert = null;
    renderActiveAlerts();
    renderHistory();
}

clearHistoryBtn.addEventListener('click', () => {
    alertHistory = [];
    localStorage.removeItem('landslide-alert-history');
    renderHistory();
});

renderHistory();
renderActiveAlerts();

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
        if (alertLevel === 'SAFE') resolveAlert();
        else if (lastAlertLevel === null || lastAlertLevel === 'SAFE') beginAlert(data);
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

