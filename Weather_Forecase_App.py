import streamlit as st
from geopy.geocoders import Nominatim
from meteostat import Point, Daily, Hourly
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import pickle as pkl
from keras.models import load_model
from plotly import graph_objs as go


def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        st.error("City not found. Please enter a valid city name.")
        return None, None


def plot_temperature(data, freq='hourly'):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['temperature'],  label='Temperature' if freq=="hourly" else "Average Temperature", color='blue')
    
    if freq == 'daily' and 'tmax' in data.columns:
        plt.scatter(data.index, data['tmax'], color='red', label='Maximum Temperature', alpha=0.6)
        plt.scatter(data.index, data['tmin'], color='green', label='Minimum Temperature', alpha=0.6)


 
    plt.title(f"Temperature Time Series ({freq.capitalize()})")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    st.pyplot(plt)



def temperature_insights(data):
    if 'tmax' in data.columns and 'tmin' in data.columns:
        max_temp = data['tmax'].max()
        min_temp = data['tmin'].min()
        avg_temp = data['temperature'].mean()
        high_temp_count = (data['tmax'] > avg_temp).sum()

        st.markdown(f"### Temperature Insights (Daily):")
        st.write(f"- Maximum Temperature: {max_temp:.2f} °C")
        st.write(f"- Minimum Temperature: {min_temp:.2f} °C")
        st.write(f"- Average Temperature (TAVG): {avg_temp:.2f} °C")
        st.write(f"- Number of Days with Above Average TMAX: {high_temp_count}")
    else:
        max_temp = data['temperature'].max()
        min_temp = data['temperature'].min()
        avg_temp = data['temperature'].mean()
        high_temp_count = (data['temperature'] > avg_temp).sum()

        st.markdown(f"### Temperature Insights (Hourly):")
        st.write(f"- Maximum Temperature: {max_temp:.2f} °C")
        st.write(f"- Minimum Temperature: {min_temp:.2f} °C")
        st.write(f"- Average Temperature: {avg_temp:.2f} °C")
        st.write(f"- Number of Hours with Above Average Temperature: {high_temp_count}")




def modelPrediction(data_for_model,forecast_duration,window_size=100,freq="Hourly"):
        
   if freq=="Hourly":
        scaler=pkl.load(open(r"Models/Hourly/StandardScaler/StandardScaler_hourly1.pkl","rb"))
        hourly_model=load_model(r"Models/Hourly/StandardScaler/temperature_best_Standard_Hourly1.keras")

        data_for_model=scaler.transform(np.array(data_for_model["temp"].values[:100]).reshape(-1,1))
        data_for_model=data_for_model.reshape(1,100,1)
        data_for_model=data_for_model.flatten()


        output=[]
        for i in range(forecast_duration):
            print(data_for_model)

            prediction = hourly_model.predict(np.array(data_for_model).reshape(-1, window_size, 1))
            
            output.append(prediction)
            data_for_model = np.append(data_for_model[1:], prediction)


        return  list(np.array(output).flatten())
   elif freq=="Daily":
        scaler_min = pkl.load(open("Models/Daily/temperature_best_min/StandardScaler/StandardScaler_min.pkl", "rb"))
        daily_min_model = load_model("Models/Daily/temperature_best_min/StandardScaler/temperature_best_min_StandardScaler.keras")
        
        scaler_max = pkl.load(open("Models/Daily/temperature_best_max/StandardScaler/StandardScaler.pkl", "rb"))
        daily_max_model = load_model("Models/Daily/temperature_best_max/StandardScaler/temperature_best_max_StandardScaler.keras")
        
        scaler_avg = pkl.load(open("Models/Daily/temperature_best_avg/StandardScaler/MinMax_Avg.pkl", "rb"))
        daily_avg_model = load_model("Models/Daily/temperature_best_avg/StandardScaler/temperature_best_avg.keras")
       
        data_for_model_min=scaler_min.transform(np.array(data_for_model["tmin"].values).reshape(-1,1))
        data_for_model_max=scaler_max.transform(np.array(data_for_model["tmax"].values).reshape(-1,1))
        data_for_model_avg=scaler_avg.transform(np.array(data_for_model["tavg"].values).reshape(-1,1))

        data_for_model_min=data_for_model_min.reshape(1,100,1)
        data_for_model_min=data_for_model_min.flatten()

        data_for_model_max=data_for_model_max.reshape(1,100,1)
        data_for_model_max=data_for_model_max.flatten()

        data_for_model_avg=data_for_model_avg.reshape(1,100,1)
        data_for_model_avg=data_for_model_avg.flatten()


        output_min=[]
        for i in range(forecast_duration):
        

            prediction = daily_min_model.predict(np.array(data_for_model_min).reshape(-1, window_size, 1))
            
            output_min.append(prediction)
            data_for_model_min = np.append(data_for_model_min[1:], prediction)

        output_max=[]
        for i in range(forecast_duration):
       

            prediction = daily_max_model.predict(np.array(data_for_model_max).reshape(-1, window_size, 1))
            
            output_max.append(prediction)
            data_for_model_max = np.append(data_for_model_max[1:], prediction)

        output_avg=[]
        for i in range(forecast_duration):

            prediction = daily_avg_model.predict(np.array(data_for_model_avg).reshape(-1, window_size, 1))
            
            output_avg.append(prediction)
            data_for_model_avg = np.append(data_for_model_avg[1:], prediction)


        return  list(np.array(output_min).flatten()),list(np.array(output_max).flatten()),list(np.array(output_avg).flatten())
    
       
def display_forecast_results(forecast_data, freq):
    if freq == "Daily":
        min_data, max_data, avg_data = forecast_data

        df = pd.DataFrame({
            "Day": range(1, len(min_data) + 1),
            "Min Temperature": min_data,
            "Max Temperature": max_data,
            "Average Temperature": avg_data,
        })

        st.subheader("Forecast Summary")
        st.write(f"Forecast Duration: {len(min_data)} days")
        st.write(f"Min Temperature Range: {min(min_data):.2f} to {max(min_data):.2f} °C")
        st.write(f"Max Temperature Range: {min(max_data):.2f} to {max(max_data):.2f} °C")
        st.write(f"Average Temperature Range: {min(avg_data):.2f} to {max(avg_data):.2f} °C")


        fig = go.Figure()
        fig.add_trace(go.Scatter(y=min_data, name='Min Temp (°C)', mode='lines+markers'))
        fig.add_trace(go.Scatter(y=max_data, name='Max Temp (°C)', mode='lines+markers'))
        fig.add_trace(go.Scatter(y=avg_data, name='Avg Temp (°C)', mode='lines+markers'))
        fig.update_layout(title='Daily Temperature Forecast', xaxis_title='Days', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig)

      
        st.write("Detailed Forecast Data")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast Data as CSV", data=csv, file_name='daily_forecast.csv')
    
    elif freq == "Hourly":
        
        df = pd.DataFrame({
            "Hour": range(1, len(forecast_data) + 1),
            "Temperature": forecast_data,
        })

        st.subheader("Forecast Summary")
        st.write(f"Forecast Duration: {len(forecast_data)} hours")
        st.write(f"Temperature Range: {min(forecast_data):.2f} to {max(forecast_data):.2f} °C")

      
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=forecast_data, name='Temperature (°C)', mode='lines+markers'))
        fig.update_layout(title='Hourly Temperature Forecast', xaxis_title='Hours', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig)

        st.write("Detailed Forecast Data")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast Data as CSV", data=csv, file_name='hourly_forecast.csv')




st.title("Temperature Analysis App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Hourly Temperature", "Daily Temperature", "Hourly Forecast","Daily Forecast"])



def calculate_dates(freq="hourly"):
    end_date = datetime.now()  
    if freq == "hourly":
        start_date = end_date - timedelta(hours=105) 
    elif freq == "daily":
        start_date = end_date - timedelta(days=100)  
    else:
        raise ValueError("Invalid frequency. Use 'hourly' or 'daily'.")
    return start_date, end_date

st.header(f"{page} Viewer")

window_size=100
city_name = st.text_input("Enter the City Name:", "")
if page=="Daily Forecast":

    forecast_period = st.selectbox("Select Forecast Period(days)", ["7", "15", "30"])
elif page=="Hourly Forecast": 
    forecast_period = st.selectbox("Select Forecast Period(hours)", ["7", "15", "24"])
if  page=="Hourly Temperature" or page=="Daily Temperature":
    with st.sidebar:
        st.markdown("""
        **Instructions:**
        1. Enter a valid city name.
        2. Select a date range.
        3. View the temperature trends and insights.
        """)

    start_date = st.date_input("Start Date:", value=datetime(2023, 1, 1).date())
    end_date = st.date_input("End Date:", value=datetime(2023, 1, 10).date())
    if   st.button("Show Data"):
        if city_name and start_date and end_date:
    
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.min.time())

            lat, lon = get_coordinates(city_name)
            if lat and lon:
                location = Point(lat, lon)
                
        
                if page == "Hourly Temperature":
                        data = Hourly(location, start=start_date, end=end_date).fetch()
                        data.rename(columns={'temp': 'temperature'}, inplace=True)
                else:
                        data = Daily(location, start=start_date, end=end_date).fetch()
                        data.rename(columns={'tavg': 'temperature'}, inplace=True)  # Use average temperature for plotting


                if not data.empty:
                
                    data.rename(columns={'temp': 'temperature'}, inplace=True)

                    
                    plot_temperature(data, freq="hourly" if page == "Hourly Temperature" else "daily")

                
                    temperature_insights(data)
                else:
                    st.warning("No data available for the given city and date range.")
        else:
            st.error("Please fill out all fields.")

elif page == "Hourly Forecast":
    with st.sidebar:
        st.markdown("""
        **Instructions:**
        1. Enter a valid city name.
        2. Select The Duration.
        3. View the temperature trends and insights.(Hourly)
        """)
    start_hourly, end_hourly = calculate_dates(freq="hourly")
    
    
    if st.button("Generate Hourly Forecast"):
        if city_name:
            lat, lon = get_coordinates(city_name)
            data = Hourly(Point(lat, lon), start_hourly, end_hourly).fetch()
            st.dataframe(data)

            if lat and lon:
                forecast_data = modelPrediction(data, int(forecast_period), window_size, "Hourly")
                st.subheader("Hourly Forecast Results")
                display_forecast_results(forecast_data, "Hourly")
            else:
                st.error("Invalid city. Please enter a valid city name.")
        else:
            st.error("Please enter a city name.")


elif page == "Daily Forecast":
    with st.sidebar:
        st.markdown("""
        **Instructions:**
        1. Enter a valid city name.
        2. Select The Duration.
        3. View the temperature trends and insights.(Daily)
        """)
    start_hourly, end_hourly = calculate_dates(freq="daily")
    
    
    if st.button("Generate Daily Forecast"):
        if city_name:
            lat, lon = get_coordinates(city_name)
            data = Daily(Point(lat, lon), start_hourly, end_hourly).fetch()
            st.dataframe(data)

            if lat and lon:
                forecast_data = modelPrediction(data, int(forecast_period), window_size, "Daily")
                st.subheader("Daily Forecast Results")
                display_forecast_results(forecast_data, "Daily")
            else:
                st.error("Invalid city. Please enter a valid city name.")
        else:
            st.error("Please enter a city name.")

