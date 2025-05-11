# 🌦️ Weather Forecasting using LSTM

A deep learning-based weather forecasting system built using LSTM (Long Short-Term Memory) networks for both **daily** and **hourly** temperature prediction. This project uses historical weather data from [Meteostat](https://meteostat.net/) and provides interactive visualization and forecasting through a [Streamlit app](https://weatherforecastinglstm.streamlit.app/).

---

## 🔗 Demo

**Live App**: [weatherforecastinglstm.streamlit.app](https://weatherforecastinglstm.streamlit.app/)

---

## 🧠 Project Highlights

- **Models**: Separate LSTM models for daily `tmin`, `tmax`, `tavg` and hourly `temp`.
- **Training Periods**:
  - Daily: 2000 - 2024 (9,132 rows)
  - Hourly: 2008 - 2024 (149,017 rows)
- **Forecasting Support**:
  - Daily: Forecast 7, 15, or 30 days
  - Hourly: Forecast 7, 15, or 24 hours
- **Features Used**:
  - **Daily**: `sin_day_of_year`, `cos_day_of_year`, `month`
  - **Hourly**: `month`, `hour`, `dayofyear`
- **Performance (MAE)**:
  - `tmin`: 0.69
  - `tmax`: 0.88
  - `tavg`: 0.66
  - `hourly temp`: 0.50
- **Frameworks & Tools**:
  - TensorFlow, Keras, Streamlit, Pandas, Scikit-learn
  - GPU support via Kaggle (T4x2)
- **Utilities**:
  - Rolling predictions
  - ModelCheckpoint callbacks
  - StandardScaler (fit on `x_train`, transform `x_train` and `x_test`)

---

## 🗂️ Project Structure

```
jainam576-weather_forecasting_lstm/
├── requirements.txt
├── trial.ipynb
├── Weather_Forecase_App.py         # Streamlit app entry point
├── Final_Models/                   # Final models used in app
│   ├── Daily/
│   │   ├── temp_avg/
│   │   ├── temp_max/
│   │   └── temp_min/
│   └── Hourly/
│       └── Standard_Scaler/
├── Test_Models/                    # Experimentation and testing
│   ├── Daily/
│   └── Hourly/
```

---

## 📊 Streamlit App Features

### 🔍 Insights Pages

- **Daily Temperature Explorer**: Visualize `tmin`, `tmax`, and `tavg` for a given date range.
- **Hourly Temperature Explorer**: See temperature variations across specific hours.

### 🔮 Forecast Pages

- **Hourly Forecast**:
  - Input: City and duration (7, 15, 24 hours)
  - Output: Predicted temperature plot
- **Daily Forecast**:
  - Input: City and duration (7, 15, 30 days)
  - Output: Predicted min, max, and avg temperature plots

---

## ⚙️ Setup Instructions

### 🔧 Prerequisites

- Python 3.8+
- pip

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 Run the App

```bash
streamlit run Weather_Forecase_App.py
```

---

## 📁 Models Description

### 📌 Final_Models/

These are the trained LSTM models used in the production app:
- `Daily/`: Three separate folders for avg, max, and min models, each with `.keras` weights and scaler `.pkl`.
- `Hourly/`: One model with its associated scaler.

### 🧪 Test_Models/

All experimentation history:
- Tried different scalers (MinMaxScaler, StandardScaler)
- Multiple architecture variants
- Saved reports and MAE scores for comparison

---

## 📚 Dataset Source

- **Meteostat** Python Library:
  - Hourly and Daily historical temperature data
  - Used for cities globally (based on user input)

---

## 🏗️ Future Work

- Add precipitation and humidity predictions
- Implement GRU/BiLSTM comparisons
- Improve model interpretability with SHAP
- Model quantization for mobile deployment

---

## 🙋‍♂️ Author

**Jainam Sanghavi**  

🔗 [LinkedIn](https://www.linkedin.com/in/jainamsanghavi/)  
📧 sanghavijainam86@gmail.com

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.
