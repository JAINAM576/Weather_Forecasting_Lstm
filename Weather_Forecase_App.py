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
        scaler=pkl.load(open(r"Final_Models/Hourly/Standard_Scaler/StandardScaler_hourly.pkl","rb"))
        hourly_model=load_model(r"Final_Models/Hourly/Standard_Scaler/temperature_Standard.keras")

        data_for_model.reset_index(inplace=True)
        target_column="temp"
        
        last_row = data_for_model.iloc[-1]
        base_date = pd.to_datetime(last_row['time'])

        data_for_model=data_for_model[["time",target_column]]

        data_for_model['month'] = data_for_model['time'].dt.month
        data_for_model['hour'] = data_for_model['time'].dt.hour  
        data_for_model["dayofyear"]=data_for_model["time"].dt.dayofyear

        seasonal_mean = data_for_model.groupby(['month', 'hour'])["temp"].transform('mean') 
        data_for_model[target_column] = data_for_model[target_column].fillna(seasonal_mean) 



        data_for_model=data_for_model[[target_column,"month","hour","dayofyear"]]

        data_for_model=scaler.transform(np.array(data_for_model.values[:100]).reshape(-1,1))
        data_for_model=data_for_model.reshape(100,4)




        output = []
       
        for i in range(forecast_duration):
            future_datetime = base_date + timedelta(hours=i + 1)
            
            hour = future_datetime.hour
            day_of_year = future_datetime.timetuple().tm_yday
            month = future_datetime.month

            prediction = hourly_model.predict(data_for_model.reshape(-1, window_size, 4), verbose=1)
            predicted_temp = prediction[0][0]
            output.append(prediction)

            next_input = np.array([predicted_temp, month, hour, day_of_year])
            next_input_scaled = scaler.transform(next_input.reshape(-1, 1))
            data_for_model = np.vstack([data_for_model[1:], next_input_scaled.reshape(1, 4)])



        return  list(np.array(output).flatten())
   elif freq=="Daily":
        print("Updated Code")
        scaler_min = pkl.load(open("Final_Models/Daily/temp_min/Standard_Scaler_min.pkl", "rb"))
        daily_min_model = load_model("Final_Models/Daily/temp_min/temperature_min_Standard.keras")
        
        scaler_max = pkl.load(open("Final_Models/Daily/temp_max/Standard_Scaler.pkl", "rb"))
        daily_max_model = load_model("Final_Models/Daily/temp_max/temperature_max_Done5_Standard.keras")
        
        scaler_avg = pkl.load(open("Final_Models/Daily/temp_avg/Standard_Scaler_avg.pkl", "rb"))
        daily_avg_model = load_model("Final_Models/Daily/temp_avg/temperature_avg_Standard.keras")

        data_for_model.reset_index(inplace=True)

        data_for_model['month'] = data_for_model['time'].dt.month
        data_for_model['dayofyear'] = data_for_model['time'].dt.dayofyear  


        # for temp_min column 
        data_for_model_min=data_for_model.copy()

        seasonal_mean = data_for_model_min.groupby('month')["tmin"].transform('mean') 
        data_for_model_min["tmin"] = data_for_model_min["tmin"].fillna(seasonal_mean) 
        data_for_model_min['sin_day_of_year'] = np.round(np.sin(2 * np.pi * data_for_model_min['dayofyear'] / 365),4)
        data_for_model_min['cos_day_of_year'] = np.round(np.cos(2 * np.pi * data_for_model_min['dayofyear'] / 365),4)

        data_for_model_min=data_for_model_min[["tmin","sin_day_of_year","cos_day_of_year","month"]]
       
        # for temp_max column
        data_for_model_max=data_for_model.copy()

        seasonal_mean = data_for_model_max.groupby('month')["tmax"].transform('mean') 
        data_for_model_max["tmax"] = data_for_model_max["tmax"].fillna(seasonal_mean) 
        data_for_model_max['sin_day_of_year'] = np.round(np.sin(2 * np.pi * data_for_model_max['dayofyear'] / 365),4)
        data_for_model_max['cos_day_of_year'] = np.round(np.cos(2 * np.pi * data_for_model_max['dayofyear'] / 365),4)

        data_for_model_max=data_for_model_max[["tmax","sin_day_of_year","cos_day_of_year","month"]]


        # for temp_avg column 
        data_for_model_avg=data_for_model.copy()

        seasonal_mean = data_for_model_avg.groupby('month')["tavg"].transform('mean') 
        data_for_model_avg["tavg"] = data_for_model_avg["tavg"].fillna(seasonal_mean) 
        data_for_model_avg['sin_day_of_year'] = np.round(np.sin(2 * np.pi * data_for_model_avg['dayofyear'] / 365),4)
        data_for_model_avg['cos_day_of_year'] = np.round(np.cos(2 * np.pi * data_for_model_avg['dayofyear'] / 365),4)

        data_for_model_avg=data_for_model_avg[["tavg","sin_day_of_year","cos_day_of_year","month"]]

        # done

        # Apply the transformation
        
        data_for_model_min=scaler_min.transform(np.array(data_for_model_min.values).reshape(-1,1))
        data_for_model_max=scaler_max.transform(np.array(data_for_model_max.values).reshape(-1,1))
        data_for_model_avg=scaler_avg.transform(np.array(data_for_model_avg.values).reshape(-1,1))


        # reshaping the data
        data_for_model_min=data_for_model_min.reshape(100,4)
        data_for_model_max=data_for_model_max.reshape(100,4)
        data_for_model_avg=data_for_model_avg.reshape(100,4)


        last_row = data_for_model.iloc[-1]
        base_date = pd.to_datetime(last_row['time'])
        
        # for predicting of output of temp min
        output_min=[]
        for i in range(forecast_duration):
            future_date = base_date + timedelta(days=i + 1)
            day_of_year = future_date.timetuple().tm_yday
            sin_day = np.round(np.sin(2 * np.pi * day_of_year / 365),4)
            cos_day = np.round(np.cos(2 * np.pi * day_of_year / 365),4)
            month = future_date.month
        
            
            prediction = daily_min_model.predict(data_for_model_min.reshape(-1, window_size, 4), verbose=1)
            predicted_temp = prediction[0][0]
            output_min.append(prediction)

            
            next_input = np.array([predicted_temp, sin_day, cos_day, month])
            next_input_scaled = scaler_min.transform(next_input.reshape(-1,1))
            data_for_model_min = np.vstack([data_for_model_min[1:], next_input_scaled.reshape(1,4)])    


        # for predicting of output of temp max

        output_max=[]
        for i in range(forecast_duration):
            future_date = base_date + timedelta(days=i + 1)
            day_of_year = future_date.timetuple().tm_yday
            sin_day = np.round(np.sin(2 * np.pi * day_of_year / 365),4)
            cos_day = np.round(np.cos(2 * np.pi * day_of_year / 365),4)
            month = future_date.month
        
            
            prediction = daily_max_model.predict(data_for_model_max.reshape(-1, window_size, 4), verbose=1)
            predicted_temp = prediction[0][0]
            output_max.append(prediction)

            
            next_input = np.array([predicted_temp, sin_day, cos_day, month])
            next_input_scaled = scaler_max.transform(next_input.reshape(-1,1))
            data_for_model_max = np.vstack([data_for_model_max[1:], next_input_scaled.reshape(1,4)])   
        
        # for predicting of output of temp avg
        output_avg=[]
        for i in range(forecast_duration):
            future_date = base_date + timedelta(days=i + 1)
            day_of_year = future_date.timetuple().tm_yday
            sin_day = np.round(np.sin(2 * np.pi * day_of_year / 365),4)
            cos_day = np.round(np.cos(2 * np.pi * day_of_year / 365),4)
            month = future_date.month
        
            
            prediction = daily_avg_model.predict(data_for_model_avg.reshape(-1, window_size, 4), verbose=1)
            predicted_temp = prediction[0][0]
            output_avg.append(prediction)

            
            next_input = np.array([predicted_temp, sin_day, cos_day, month])
            next_input_scaled = scaler_avg.transform(next_input.reshape(-1,1))
            data_for_model_avg = np.vstack([data_for_model_avg[1:], next_input_scaled.reshape(1,4)])  


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
        start_date = end_date - timedelta(hours=100) 
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

