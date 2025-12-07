# 🏎️ F1 2025 Final Lap Prediction

Real-time Formula 1 race prediction using **machine learning** and **live weather data**. Predicts race outcomes based on 2024 Abu Dhabi GP data, qualifying times, team performance, and current weather conditions.

## 🚀 Features

- **XGBoost Machine Learning Model** for race time predictions
- **Live Weather Integration** via OpenWeatherMap API
- **FastF1 API** for historical race data and telemetry
- **Real-time predictions** based on qualifying results
- Team performance scoring and clean air race pace analysis

## 📊 Data Sources

- **FastF1 API**: 2024 Abu Dhabi GP race data, lap times, and sector analysis
- **OpenWeatherMap API**: Real-time temperature and rain probability
- **2025 Qualifying Data**: Latest qualifying session results
- **Team Performance**: 2024 constructor standings

## 🏁 How It Works

1. **Load Historical Data**: Fetches 2024 Abu Dhabi GP race data via FastF1
2. **Live Weather**: Gets current weather conditions for Yas Marina Circuit
3. **Feature Engineering**: Combines qualifying times, team scores, race pace, and weather
4. **Model Training**: XGBoost regressor trained on 2024 data
5. **Prediction**: Predicts race times and ranks drivers for podium positions
6. **Evaluation**: Displays Mean Absolute Error (MAE) and feature importance

## 📦 Dependencies

```bash
pip install fastf1 pandas numpy scikit-learn xgboost matplotlib requests python-dotenv
```

## 🔑 API Key Setup

1. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create a `.env` file in the project root:

```
API_KEY=your_api_key_here
```

3. The `.env` file is automatically ignored by git and won't be pushed

## 🔧 Usage

Run the prediction script:

```bash
python prediction.py
```

Expected output:

```
🌤️  Live Weather: 29.1°C, Rain Probability: 0%
   Driver  PredictedRaceTime (s)
0     VER              89.512375
...
🏆 Predicted in the Top 3 🏆
🥇 P1: VER
🥈 P2: NOR
🥉 P3: PIA
Model Error (MAE): 0.99 seconds
```

## 📈 Model Performance

- **MAE (Mean Absolute Error)**: ~0.99 seconds
- **Features Used**: Qualifying time, rain probability, temperature, team performance, clean air race pace
- **Algorithm**: XGBoost Regressor with monotone constraints
- Feature importance visualization included

## 📌 Future Improvements

- Add pit stop strategy predictions
- Incorporate tire degradation models
- Multi-race trend analysis
- Deep learning model exploration

## 📜 License

This project is licensed under the MIT License.

## 🔗 Repository

[https://github.com/mugenkyou/f12025-final-lap-prediction](https://github.com/mugenkyou/f12025-final-lap-prediction)

🏎️ **Predict F1 races with real-time data!** 🚀
