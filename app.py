from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# 1. GENERATE 1000 EV DATA
# ============================================================

np.random.seed(42)

N = 1000

speed = np.random.randint(20, 121, N)
temperature = np.random.uniform(15, 50, N)
battery_capacity = np.random.uniform(30, 100, N)
cargo_load = np.random.randint(0, 301, N)
charge_cycles = np.random.randint(10, 1001, N)
voltage_retention = np.random.uniform(50, 100, N)


# ============================================================
# 2. REMAINING RANGE CALCULATION
# ============================================================

base_range = battery_capacity * 4.5

speed_penalty = (speed - 20) * 0.08

temperature_penalty = np.maximum(
    temperature - 25, 0
) * 0.25

load_penalty = cargo_load * 0.015

cycle_penalty = charge_cycles * 0.012

voltage_factor = voltage_retention / 100


remaining_range = (
    base_range
    - speed_penalty
    - temperature_penalty
    - load_penalty
    - cycle_penalty
) * voltage_factor

remaining_range += np.random.normal(0, 2, N)

remaining_range = np.maximum(
    remaining_range,
    5
)


# ============================================================
# 3. DATAFRAME
# ============================================================

df = pd.DataFrame({

    "Speed_KMH": speed,

    "Temperature_C": temperature,

    "Battery_Capacity_KWh":
        battery_capacity,

    "Cargo_Load_KG":
        cargo_load,

    "Charge_Cycles":
        charge_cycles,

    "Voltage_Retention_Pct":
        voltage_retention,

    "Remaining_Range_KM":
        remaining_range
})


# ============================================================
# 4. RANDOM FOREST
# ============================================================

features = [

    "Speed_KMH",

    "Temperature_C",

    "Battery_Capacity_KWh",

    "Cargo_Load_KG",

    "Charge_Cycles",

    "Voltage_Retention_Pct"

]

X = df[features]

y = df["Remaining_Range_KM"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


rf_model = RandomForestRegressor(

    n_estimators=50,

    random_state=42

)

rf_model.fit(
    X_train,
    y_train
)


y_pred = rf_model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 5. K-MEANS
# ============================================================

cluster_data = df[
    [
        "Charge_Cycles",
        "Voltage_Retention_Pct"
    ]
]


scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    cluster_data
)


kmeans = KMeans(

    n_clusters=3,

    random_state=42,

    n_init=10

)

df["Cluster"] = kmeans.fit_predict(
    scaled_data
)


# ============================================================
# 6. BATTERY HEALTH LABELS
# ============================================================

cluster_average = df.groupby(
    "Cluster"
)[
    [
        "Charge_Cycles",
        "Voltage_Retention_Pct"
    ]
].mean()


health_score = (

    cluster_average[
        "Voltage_Retention_Pct"
    ]

    -

    cluster_average[
        "Charge_Cycles"
    ] * 0.03

)


sorted_clusters = health_score.sort_values(
    ascending=False
).index.tolist()


cluster_health = {

    sorted_clusters[0]:
        "Excellent Health",

    sorted_clusters[1]:
        "Moderate Degradation",

    sorted_clusters[2]:
        "Immediate Replacement"

}


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_vehicle(

    speed_value,

    temperature_value,

    battery_capacity_value,

    load_value,

    cycles_value,

    voltage_value

):

    input_data = pd.DataFrame([{

        "Speed_KMH":
            speed_value,

        "Temperature_C":
            temperature_value,

        "Battery_Capacity_KWh":
            battery_capacity_value,

        "Cargo_Load_KG":
            load_value,

        "Charge_Cycles":
            cycles_value,

        "Voltage_Retention_Pct":
            voltage_value

    }])


    # Random Forest prediction

    predicted_range = rf_model.predict(
        input_data
    )[0]


    # K-Means prediction

    cluster_input = scaler.transform([

        [
            cycles_value,
            voltage_value
        ]

    ])


    cluster = int(
        kmeans.predict(
            cluster_input
        )[0]
    )


    battery_health = cluster_health[
        cluster
    ]


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    if (
        temperature_value >= 46
        and
        speed_value >= 95
    ):

        safety_status = (
            "High Temperature Risk"
        )

        safety_message = (
            "High temperature and speed detected. "
            "Reduce vehicle speed immediately."
        )

        safety_alert = True

    else:

        safety_status = "Safe"

        safety_message = (
            "Vehicle is operating within "
            "normal safety conditions."
        )

        safety_alert = False


    # --------------------------------------------------------
    # CHARGING
    # --------------------------------------------------------

    if predicted_range < 40:

        charging_status = (
            "Charging Required"
        )

        charging_message = (
            "Fast-Charging Node: "
            "Station Echo - 2.6 KM away"
        )

    else:

        charging_status = (
            "Charging Not Required"
        )

        charging_message = (
            "Current predicted range is sufficient."
        )


    # --------------------------------------------------------
    # FINANCIAL SAVINGS
    # --------------------------------------------------------

    fuel_price = 105

    ice_efficiency = 18

    fuel_cost_per_hour = (

        speed_value /
        ice_efficiency

    ) * fuel_price


    ev_cost_per_hour = 12


    savings = max(

        fuel_cost_per_hour
        -
        ev_cost_per_hour,

        0

    )


    # --------------------------------------------------------
    # CO2
    # --------------------------------------------------------

    co2_factor = 2.31

    fuel_consumption = (
        speed_value /
        ice_efficiency
    )

    co2_prevented = (
        fuel_consumption *
        co2_factor
    )


    return {

        "remaining_range":
            round(
                predicted_range,
                2
            ),

        "battery_health":
            battery_health,

        "cluster":
            cluster + 1,

        "safety_status":
            safety_status,

        "safety_message":
            safety_message,

        "safety_alert":
            safety_alert,

        "charging_status":
            charging_status,

        "charging_message":
            charging_message,

        "savings":
            round(
                savings,
                2
            ),

        "co2_prevented":
            round(
                co2_prevented,
                2
            )

    }


# ============================================================
# 8. DASHBOARD HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
Intelligent EV Performance Dashboard
</title>


<!-- CHART.JS -->

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<style>


/* =========================================================
   GLOBAL
   ========================================================= */

* {

    margin: 0;

    padding: 0;

    box-sizing: border-box;

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;

}


body {

    background:
        linear-gradient(
            135deg,
            #eef4ff,
            #f8faff
        );

    color: #172033;

}


/* =========================================================
   HEADER
   ========================================================= */

.header {

    background:
        linear-gradient(
            135deg,
            #111827,
            #1d4ed8
        );

    color: white;

    padding: 30px 45px;

    box-shadow:
        0 5px 20px
        rgba(37,99,235,0.20);

}


.header-content {

    max-width: 1450px;

    margin: auto;

}


.header h1 {

    font-size: 29px;

    letter-spacing: 0.3px;

}


.header p {

    margin-top: 8px;

    color: #dbeafe;

    font-size: 14px;

}


/* =========================================================
   CONTAINER
   ========================================================= */

.container {

    max-width: 1450px;

    margin: auto;

    padding: 30px 40px;

}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 20px;

    margin-bottom: 28px;

}


.kpi {

    background: white;

    border-radius: 17px;

    padding: 23px;

    position: relative;

    overflow: hidden;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.07);

}


.kpi::after {

    content: "";

    position: absolute;

    right: -25px;

    top: -25px;

    width: 90px;

    height: 90px;

    border-radius: 50%;

    background:
        rgba(37,99,235,0.08);

}


.kpi-icon {

    font-size: 25px;

    margin-bottom: 12px;

}


.kpi-title {

    color: #64748b;

    font-size: 13px;

}


.kpi-value {

    font-size: 25px;

    font-weight: 700;

    margin-top: 8px;

}


/* =========================================================
   MAIN
   ========================================================= */

.main-grid {

    display: grid;

    grid-template-columns:
        350px 1fr;

    gap: 25px;

}


.panel {

    background: white;

    border-radius: 18px;

    padding: 25px;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.07);

}


.panel-title {

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 22px;

}


/* =========================================================
   INPUT SECTION
   ========================================================= */

.input-box {

    margin-bottom: 20px;

}


.input-label {

    display: flex;

    justify-content: space-between;

    margin-bottom: 8px;

    font-size: 13px;

    font-weight: 600;

}


.input-value {

    color: #2563eb;

}


input[type="range"] {

    width: 100%;

    accent-color: #2563eb;

    cursor: pointer;

}


.predict-btn {

    width: 100%;

    border: none;

    padding: 15px;

    margin-top: 5px;

    border-radius: 10px;

    color: white;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    box-shadow:
        0 7px 18px
        rgba(37,99,235,0.25);

    transition: 0.2s;

}


.predict-btn:hover {

    transform: translateY(-2px);

    box-shadow:
        0 10px 25px
        rgba(37,99,235,0.32);

}


/* =========================================================
   RANGE
   ========================================================= */

.range-card {

    text-align: center;

    padding: 25px;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            #eff6ff,
            #eef2ff
        );

    border: 1px solid #dbeafe;

    margin-bottom: 20px;

}


.range-title {

    color: #64748b;

    font-size: 14px;

}


.range-value {

    color: #2563eb;

    font-size: 45px;

    font-weight: 800;

    margin-top: 7px;

}


/* =========================================================
   RESULTS
   ========================================================= */

.result-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 15px;

}


.result-card {

    padding: 19px;

    border-radius: 13px;

    background: #f8fafc;

    border: 1px solid #e2e8f0;

}


.result-title {

    color: #64748b;

    font-size: 12px;

}


.result-value {

    font-size: 18px;

    font-weight: 700;

    margin-top: 8px;

}


/* =========================================================
   ALERTS
   ========================================================= */

.alert {

    display: none;

    padding: 16px;

    margin-top: 18px;

    border-radius: 11px;

    font-size: 14px;

    font-weight: 600;

}


.danger {

    background: #fee2e2;

    color: #b91c1c;

    border-left:
        5px solid #ef4444;

}


.success {

    background: #dcfce7;

    color: #166534;

    border-left:
        5px solid #22c55e;

}


/* =========================================================
   RECOMMENDATION
   ========================================================= */

.recommendation {

    margin-top: 18px;

    padding: 18px;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #fffbeb
        );

    border-left:
        5px solid #f97316;

}


.recommendation h3 {

    font-size: 14px;

    margin-bottom: 7px;

}


.recommendation p {

    font-size: 13px;

    color: #475569;

}


/* =========================================================
   CHART SECTION
   ========================================================= */

.chart-section {

    margin-top: 28px;

}


.chart-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 22px;

}


.chart-card {

    background: white;

    padding: 22px;

    border-radius: 18px;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.07);

}


.chart-card h2 {

    font-size: 17px;

    margin-bottom: 18px;

}


.chart-container {

    height: 300px;

}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    margin-top: 30px;

    padding: 20px;

    color: #64748b;

    font-size: 13px;

}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media(max-width: 1000px) {

    .kpi-grid {

        grid-template-columns:
            repeat(2, 1fr);

    }

    .main-grid {

        grid-template-columns: 1fr;

    }

}


@media(max-width: 650px) {

    .container {

        padding: 20px;

    }

    .kpi-grid {

        grid-template-columns: 1fr;

    }

    .chart-grid {

        grid-template-columns: 1fr;

    }

    .result-grid {

        grid-template-columns: 1fr;

    }

    .header {

        padding: 25px 20px;

    }

    .header h1 {

        font-size: 22px;

    }

}


</style>

</head>


<body>


<!-- =========================================================
     HEADER
     ========================================================= -->

<div class="header">

<div class="header-content">

<h1>
⚡ Intelligent EV Performance & Battery Analytics Dashboard
</h1>

<p>
Machine Learning Based Electric Vehicle Fleet Monitoring,
Performance Prediction and Battery Health Analytics
</p>

</div>

</div>


<div class="container">


<!-- =========================================================
     KPI CARDS
     ========================================================= -->

<div class="kpi-grid">


<div class="kpi">

<div class="kpi-icon">
🚗
</div>

<div class="kpi-title">
Total EV Records
</div>

<div class="kpi-value">
1000
</div>

</div>


<div class="kpi">

<div class="kpi-icon">
🤖
</div>

<div class="kpi-title">
ML Algorithm
</div>

<div class="kpi-value">
Random Forest
</div>

</div>


<div class="kpi">

<div class="kpi-icon">
📈
</div>

<div class="kpi-title">
R² Model Score
</div>

<div class="kpi-value">
{{ r2 }}
</div>

</div>


<div class="kpi">

<div class="kpi-icon">
🔋
</div>

<div class="kpi-title">
Battery Clusters
</div>

<div class="kpi-value">
3
</div>

</div>


</div>


<!-- =========================================================
     INPUT + OUTPUT
     ========================================================= -->

<div class="main-grid">


<!-- ========================================================
     INPUT PANEL
     ======================================================== -->

<div class="panel">

<div class="panel-title">
🚗 Vehicle Telemetry
</div>


<!-- SPEED -->

<div class="input-box">

<div class="input-label">

<span>
Speed
</span>

<span
class="input-value"
id="speedValue">

70 km/h

</span>

</div>

<input
type="range"
id="speed"
min="20"
max="120"
value="70"
oninput="updateValues()">

</div>


<!-- TEMPERATURE -->

<div class="input-box">

<div class="input-label">

<span>
Temperature
</span>

<span
class="input-value"
id="temperatureValue">

30 °C

</span>

</div>

<input
type="range"
id="temperature"
min="15"
max="50"
value="30"
oninput="updateValues()">

</div>


<!-- BATTERY CAPACITY -->

<div class="input-box">

<div class="input-label">

<span>
Battery Capacity
</span>

<span
class="input-value"
id="batteryValue">

60 kWh

</span>

</div>

<input
type="range"
id="battery"
min="30"
max="100"
value="60"
oninput="updateValues()">

</div>


<!-- LOAD -->

<div class="input-box">

<div class="input-label">

<span>
Cargo Load
</span>

<span
class="input-value"
id="loadValue">

100 kg

</span>

</div>

<input
type="range"
id="load"
min="0"
max="300"
value="100"
oninput="updateValues()">

</div>


<!-- CHARGE CYCLES -->

<div class="input-box">

<div class="input-label">

<span>
Charge Cycles
</span>

<span
class="input-value"
id="cyclesValue">

300

</span>

</div>

<input
type="range"
id="cycles"
min="10"
max="1000"
value="300"
oninput="updateValues()">

</div>


<!-- VOLTAGE -->

<div class="input-box">

<div class="input-label">

<span>
Voltage Retention
</span>

<span
class="input-value"
id="voltageValue">

90 %

</span>

</div>

<input
type="range"
id="voltage"
min="50"
max="100"
value="90"
oninput="updateValues()">

</div>


<button
class="predict-btn"
onclick="predictVehicle()">

⚡ Predict EV Performance

</button>


</div>


<!-- ========================================================
     OUTPUT PANEL
     ======================================================== -->

<div class="panel">

<div class="panel-title">
📊 Prediction & Analytics
</div>


<div class="range-card">

<div class="range-title">
Predicted Remaining Range
</div>

<div
class="range-value"
id="remainingRange">

-- KM

</div>

</div>


<div class="result-grid">


<div class="result-card">

<div class="result-title">
Battery Health
</div>

<div
class="result-value"
id="batteryHealth">

--

</div>

</div>


<div class="result-card">

<div class="result-title">
Battery Cluster
</div>

<div
class="result-value"
id="cluster">

--

</div>

</div>


<div class="result-card">

<div class="result-title">
Safety Status
</div>

<div
class="result-value"
id="safetyStatus">

--

</div>

</div>


<div class="result-card">

<div class="result-title">
Financial Savings
</div>

<div
class="result-value"
id="savings">

₹ --

</div>

</div>


<div class="result-card">

<div class="result-title">
CO₂ Prevented
</div>

<div
class="result-value"
id="co2">

-- KG

</div>

</div>


<div class="result-card">

<div class="result-title">
Charging Status
</div>

<div
class="result-value"
id="chargingStatus">

--

</div>

</div>


</div>


<div
class="alert danger"
id="dangerAlert">
</div>


<div
class="alert success"
id="successAlert">
</div>


<div class="recommendation">

<h3>
⚡ Smart Charging Recommendation
</h3>

<p id="chargingMessage">

Enter vehicle values and click
Predict EV Performance.

</p>

</div>


</div>

</div>


<!-- =========================================================
     CHARTS
     ========================================================= -->

<div class="chart-section">


<div class="chart-grid">


<!-- SPEED RANGE -->

<div class="chart-card">

<h2>
📈 Speed vs Remaining Range
</h2>

<div class="chart-container">

<canvas id="speedChart"></canvas>

</div>

</div>


<!-- TEMPERATURE -->

<div class="chart-card">

<h2>
🌡️ Temperature vs Remaining Range
</h2>

<div class="chart-container">

<canvas id="temperatureChart"></canvas>

</div>

</div>


<!-- BATTERY CAPACITY -->

<div class="chart-card">

<h2>
🔋 Battery Capacity vs Remaining Range
</h2>

<div class="chart-container">

<canvas id="batteryChart"></canvas>

</div>

</div>


<!-- BATTERY HEALTH -->

<div class="chart-card">

<h2>
🔋 Battery Health Distribution
</h2>

<div class="chart-container">

<canvas id="healthChart"></canvas>

</div>

</div>


</div>

</div>


<div class="footer">

Intelligent EV Performance & Battery Analytics Dashboard
<br>
Random Forest Regression • K-Means Clustering • EV Analytics

</div>


</div>


<script>


// ==========================================================
// DATA FROM PYTHON
// ==========================================================

const chartData = {{ chart_data | safe }};


// ==========================================================
// UPDATE SLIDER VALUES
// ==========================================================

function updateValues() {


document.getElementById(
    "speedValue"
).innerText =

document.getElementById(
    "speed"
).value + " km/h";


document.getElementById(
    "temperatureValue"
).innerText =

document.getElementById(
    "temperature"
).value + " °C";


document.getElementById(
    "batteryValue"
).innerText =

document.getElementById(
    "battery"
).value + " kWh";


document.getElementById(
    "loadValue"
).innerText =

document.getElementById(
    "load"
).value + " kg";


document.getElementById(
    "cyclesValue"
).innerText =

document.getElementById(
    "cycles"
).value;


document.getElementById(
    "voltageValue"
).innerText =

document.getElementById(
    "voltage"
).value + " %";

}


// ==========================================================
// PREDICTION
// ==========================================================

async function predictVehicle() {


const data = {

speed:
    document.getElementById(
        "speed"
    ).value,

temperature:
    document.getElementById(
        "temperature"
    ).value,

battery_capacity:
    document.getElementById(
        "battery"
    ).value,

load:
    document.getElementById(
        "load"
    ).value,

cycles:
    document.getElementById(
        "cycles"
    ).value,

voltage:
    document.getElementById(
        "voltage"
    ).value

};


try {


const response =
await fetch(
    "/predict",
    {

        method: "POST",

        headers: {
            "Content-Type":
                "application/json"
        },

        body:
            JSON.stringify(data)

    }
);


const result =
await response.json();


document.getElementById(
    "remainingRange"
).innerText =

result.remaining_range +
" KM";


document.getElementById(
    "batteryHealth"
).innerText =

result.battery_health;


document.getElementById(
    "cluster"
).innerText =

"Cluster " +
result.cluster;


document.getElementById(
    "safetyStatus"
).innerText =

result.safety_status;


document.getElementById(
    "savings"
).innerText =

"₹ " +
result.savings +
" / hr";


document.getElementById(
    "co2"
).innerText =

result.co2_prevented +
" KG";


document.getElementById(
    "chargingStatus"
).innerText =

result.charging_status;


document.getElementById(
    "chargingMessage"
).innerText =

result.charging_message;


const danger =
document.getElementById(
    "dangerAlert"
);


const success =
document.getElementById(
    "successAlert"
);


if (result.safety_alert) {


danger.style.display =
"block";


success.style.display =
"none";


danger.innerText =
"⚠️ " +
result.safety_message;


}

else {


danger.style.display =
"none";


success.style.display =
"block";


success.innerText =
"✓ " +
result.safety_message;

}


}

catch(error) {


alert(
    "Unable to connect to server."
);


console.error(error);

}

}


// ==========================================================
// CREATE SPEED CHART
// ==========================================================

new Chart(

document.getElementById(
    "speedChart"
),

{

type: "scatter",

data: {

datasets: [{

label:
"EV Records",

data:
chartData.speed,

backgroundColor:
"rgba(37,99,235,0.45)",

borderColor:
"#2563eb",

pointRadius: 3

}]

},

options: {

responsive: true,

maintainAspectRatio: false,

scales: {

x: {

title: {

display: true,

text:
"Speed (KM/H)"

}

},

y: {

title: {

display: true,

text:
"Remaining Range (KM)"

}

}

}

}

}

);


// ==========================================================
// TEMPERATURE CHART
// ==========================================================

new Chart(

document.getElementById(
    "temperatureChart"
),

{

type: "scatter",

data: {

datasets: [{

label:
"EV Records",

data:
chartData.temperature,

backgroundColor:
"rgba(239,68,68,0.45)",

borderColor:
"#ef4444",

pointRadius: 3

}]

},

options: {

responsive: true,

maintainAspectRatio: false,

scales: {

x: {

title: {

display: true,

text:
"Temperature (°C)"

}

},

y: {

title: {

display: true,

text:
"Remaining Range (KM)"

}

}

}

}

}

);


// ==========================================================
// BATTERY CAPACITY CHART
// ==========================================================

new Chart(

document.getElementById(
    "batteryChart"
),

{

type: "scatter",

data: {

datasets: [{

label:
"EV Records",

data:
chartData.battery,

backgroundColor:
"rgba(16,185,129,0.45)",

borderColor:
"#10b981",

pointRadius: 3

}]

},

options: {

responsive: true,

maintainAspectRatio: false,

scales: {

x: {

title: {

display: true,

text:
"Battery Capacity (KWh)"

}

},

y: {

title: {

display: true,

text:
"Remaining Range (KM)"

}

}

}

}

}

);


// ==========================================================
// HEALTH DISTRIBUTION
// ==========================================================

new Chart(

document.getElementById(
    "healthChart"
),

{

type: "doughnut",

data: {

labels:
chartData.health.labels,

datasets: [{

data:
chartData.health.values,

backgroundColor: [

"#22c55e",

"#f59e0b",

"#ef4444"

],

borderWidth: 2

}]

},

options: {

responsive: true,

maintainAspectRatio: false,

plugins: {

legend: {

position:
"bottom"

}

}

}

}

);


</script>


</body>

</html>

"""


# ============================================================
# 9. HOME ROUTE
# ============================================================

@app.route("/")
def home():


    # -----------------------------------------------
    # SPEED GRAPH DATA
    # -----------------------------------------------

    speed_chart = [

        {
            "x":
                round(
                    float(row["Speed_KMH"]),
                    2
                ),

            "y":
                round(
                    float(
                        row[
                            "Remaining_Range_KM"
                        ]
                    ),
                    2
                )
        }

        for _, row in df.iterrows()

    ]


    # -----------------------------------------------
    # TEMPERATURE GRAPH
    # -----------------------------------------------

    temperature_chart = [

        {
            "x":
                round(
                    float(
                        row[
                            "Temperature_C"
                        ]
                    ),
                    2
                ),

            "y":
                round(
                    float(
                        row[
                            "Remaining_Range_KM"
                        ]
                    ),
                    2
                )
        }

        for _, row in df.iterrows()

    ]


    # -----------------------------------------------
    # BATTERY GRAPH
    # -----------------------------------------------

    battery_chart = [

        {
            "x":
                round(
                    float(
                        row[
                            "Battery_Capacity_KWh"
                        ]
                    ),
                    2
                ),

            "y":
                round(
                    float(
                        row[
                            "Remaining_Range_KM"
                        ]
                    ),
                    2
                )
        }

        for _, row in df.iterrows()

    ]


    # -----------------------------------------------
    # BATTERY HEALTH COUNTS
    # -----------------------------------------------

    health_counts = {

        "Excellent Health":
            0,

        "Moderate Degradation":
            0,

        "Immediate Replacement":
            0

    }


    for cluster in df["Cluster"]:

        health = cluster_health[cluster]

        health_counts[health] += 1


    chart_data = {

        "speed":
            speed_chart,

        "temperature":
            temperature_chart,

        "battery":
            battery_chart,

        "health": {

            "labels": [

                "Excellent Health",

                "Moderate Degradation",

                "Immediate Replacement"

            ],

            "values": [

                health_counts[
                    "Excellent Health"
                ],

                health_counts[
                    "Moderate Degradation"
                ],

                health_counts[
                    "Immediate Replacement"
                ]

            ]

        }

    }


    import json


    return render_template_string(

        HTML,

        r2=round(
            r2,
            4
        ),

        chart_data=json.dumps(
            chart_data
        )

    )


# ============================================================
# 10. PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():


    data = request.get_json()


    result = predict_vehicle(

        float(
            data["speed"]
        ),

        float(
            data["temperature"]
        ),

        float(
            data["battery_capacity"]
        ),

        float(
            data["load"]
        ),

        float(
            data["cycles"]
        ),

        float(
            data["voltage"]
        )

    )


    return jsonify(
        result
    )


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":


    print()
    print("=" * 65)

    print(
        " Intelligent EV Performance & Battery Analytics Dashboard"
    )

    print("=" * 65)

    print(
        f" Total EV Records : {len(df)}"
    )

    print(
        " Random Forest    : 50 Estimators"
    )

    print(
        " K-Means          : 3 Clusters"
    )

    print(
        f" R² Score         : {r2:.4f}"
    )

    print("=" * 65)

    print()

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(
        debug=True
    )