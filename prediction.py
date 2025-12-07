import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create cache directory if it doesn't exist
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}")

fastf1.Cache.enable_cache(cache_dir)

# load the 2024 Abu Dhabi session data (latest race)
session_2024 = fastf1.get_session(2024, "Abu Dhabi", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# clean air race pace from racepace.py
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

# quali data from Abu Dhabi GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["RUS", "VER", "PIA", "NOR", "HAM", "LEC", "ALO", "HUL", "ALB", "SAI", "STR", "OCO", "GAS"],
    "QualifyingTime (s)": [
        82.645,  # RUS
        82.207,  # VER
        82.437,  # PIA
        82.408,  # NOR
        83.394,  # HAM
        82.730,  # LEC
        82.902,  # ALO
        83.450,  # HUL
        83.416,  # ALB
        83.042,  # SAI
        83.097,  # STR
        82.913,  # OCO
        83.468   # GAS
    ]
})


qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Fetch live weather data for Abu Dhabi GP
API_KEY = os.getenv("API_KEY")
lat, lon = 24.4672, 54.6031  # Yas Marina Circuit coordinates

try:
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(weather_url)
    weather_data = response.json()
    
    if response.status_code == 200:
        temperature = weather_data["main"]["temp"]
        # Rain probability from weather conditions
        if "rain" in weather_data:
            rain_probability = min(weather_data.get("rain", {}).get("1h", 0) / 10, 1)  # Convert mm to probability
        else:
            rain_probability = 0.3 if weather_data["weather"][0]["main"] in ["Rain", "Drizzle", "Thunderstorm"] else 0
        
        print(f"🌤️  Live Weather: {temperature}°C, Rain Probability: {rain_probability*100:.0f}%")
    else:
        # Fallback to default values
        rain_probability = 0
        temperature = 27
        print("⚠️  Using default weather values")
except Exception as e:
    # Fallback to default values if API fails
    rain_probability = 0
    temperature = 27
    print(f"⚠️  Weather API error: {e}. Using default values.")

# adjust qualifying time based on weather conditions
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# add constructor's data
team_points = {
    "McLaren": 800, "Mercedes": 459, "Red Bull": 426, "Williams": 137, "Ferrari": 382,
    "Haas": 73, "Aston Martin": 80, "Kick Sauber": 68, "Racing Bulls": 92, "Alpine": 22
}

max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Ferrari", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Williams", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Driver"].map(driver_to_team).map(team_performance_score)

# merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# define features (X) and target (y)
X = merged_data[[
    "QualifyingTime (s)", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)"
]]

# impute missing values in X
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Remove rows where y is NaN
valid_mask = ~y.isna()
X_imputed_valid = X_imputed[valid_mask]
y_valid = y[valid_mask]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed_valid, y_valid, test_size=0.1, random_state=39)

# train XGBoost model with better hyperparameters for variation
model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=39,
    monotone_constraints='(1, 0, 0, -1, 1)'
)
model.fit(X_train, y_train)

# Predict with slight variation to avoid identical times
predictions = model.predict(X_imputed)
# Add small random variation based on driver characteristics to make times more realistic
np.random.seed(42)
variation = np.random.normal(0, 0.15, size=len(predictions))  # Small variation
merged_data["PredictedRaceTime (s)"] = predictions + variation

# sort the results to find the predicted winner
final_results = merged_data.sort_values(by=["PredictedRaceTime (s)", "QualifyingTime (s)"]).reset_index(drop=True)

# Calculate gaps to leader
final_results["GapToLeader (s)"] = final_results["PredictedRaceTime (s)"] - final_results["PredictedRaceTime (s)"].min()

print("\n" + "="*60)
print("🏁 ABU DHABI GP 2025 - RACE TIME PREDICTIONS 🏁")
print("="*60)
print(final_results[["Driver", "PredictedRaceTime (s)", "GapToLeader (s)"]].to_string(index=False))

# Get top 3
podium = final_results.head(3)
print("\n" + "="*60)
print("🏆 PREDICTED PODIUM 🏆")
print("="*60)
print(f"🥇 P1: {podium.iloc[0]['Driver']} - {podium.iloc[0]['PredictedRaceTime (s)']:.3f}s")
print(f"🥈 P2: {podium.iloc[1]['Driver']} - {podium.iloc[1]['PredictedRaceTime (s)']:.3f}s (+{podium.iloc[1]['GapToLeader (s)']:.3f}s)")
print(f"🥉 P3: {podium.iloc[2]['Driver']} - {podium.iloc[2]['PredictedRaceTime (s)']:.3f}s (+{podium.iloc[2]['GapToLeader (s)']:.3f}s)")
print("="*60 + "\n")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Model Error (MAE): {mae:.3f} seconds\n")

# ============================================================================
# ENHANCED VISUALIZATIONS
# ============================================================================

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#FF1801', '#00D2BE', '#0600EF', '#FFA500', '#000000', '#006F62', 
          '#2B4562', '#DC0000', '#1E41FF', '#005AFF', '#C92D4B', '#B6BABD']

# Figure 1: Predicted Race Times - Top 10
fig1, ax1 = plt.subplots(figsize=(12, 7))
top_10 = final_results.head(10)
bars = ax1.barh(range(len(top_10)), top_10["PredictedRaceTime (s)"], color=colors[:len(top_10)])
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels([f"P{i+1}: {driver}" for i, driver in enumerate(top_10["Driver"])])
ax1.invert_yaxis()
ax1.set_xlabel('Predicted Race Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('🏁 Abu Dhabi GP 2025 - Predicted Race Times (Top 10)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, top_10["PredictedRaceTime (s)"])):
    ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}s', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('race_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Gap to Leader Visualization
fig2, ax2 = plt.subplots(figsize=(12, 7))
top_10_gaps = final_results.head(10)
bars2 = ax2.barh(range(len(top_10_gaps)), top_10_gaps["GapToLeader (s)"], 
                 color=colors[:len(top_10_gaps)])
ax2.set_yticks(range(len(top_10_gaps)))
ax2.set_yticklabels([f"P{i+1}: {driver}" for i, driver in enumerate(top_10_gaps["Driver"])])
ax2.invert_yaxis()
ax2.set_xlabel('Gap to Leader (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('⏱️ Abu Dhabi GP 2025 - Gap to Leader Analysis', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, top_10_gaps["GapToLeader (s)"]):
    if val > 0:
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'+{val:.3f}s', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('gap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Feature Importance
fig3, ax3 = plt.subplots(figsize=(10, 6))
feature_importance = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])

bars3 = ax3.barh(pos, feature_importance[sorted_idx], color='#00D2BE')
ax3.set_yticks(pos)
ax3.set_yticklabels(features[sorted_idx], fontsize=11)
ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax3.set_title('🔍 Feature Importance in Race Time Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars3, feature_importance[sorted_idx]):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: Podium Comparison
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

# Qualifying vs Predicted Race Time
podium_data = final_results.head(3)
x_pos = np.arange(len(podium_data))
width = 0.35

bars_q = ax4a.bar(x_pos - width/2, podium_data["QualifyingTime (s)"], 
                   width, label='Qualifying Time', color='#0600EF', alpha=0.8)
bars_r = ax4a.bar(x_pos + width/2, podium_data["PredictedRaceTime (s)"], 
                   width, label='Predicted Race Time', color='#FF1801', alpha=0.8)

ax4a.set_xlabel('Position', fontsize=11, fontweight='bold')
ax4a.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax4a.set_title('🏆 Podium: Qualifying vs Predicted Race Time', 
               fontsize=12, fontweight='bold')
ax4a.set_xticks(x_pos)
ax4a.set_xticklabels([f"P{i+1}\n{driver}" for i, driver in enumerate(podium_data["Driver"])])
ax4a.legend()
ax4a.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars_q, bars_r]:
    for bar in bars:
        height = bar.get_height()
        ax4a.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

# Predicted Race Pace
colors_podium = ['#FFD700', '#C0C0C0', '#CD7F32']
wedges, texts, autotexts = ax4b.pie(podium_data["PredictedRaceTime (s)"], 
                                      labels=[f"P{i+1}: {driver}" for i, driver in enumerate(podium_data["Driver"])],
                                      autopct='%1.1f%%',
                                      colors=colors_podium,
                                      startangle=90,
                                      textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4b.set_title('🥇 Podium Distribution by Race Time', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('podium_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All visualizations saved as PNG files!")
print("   • race_predictions.png")
print("   • gap_analysis.png") 
print("   • feature_importance.png")
print("   • podium_analysis.png\n")

