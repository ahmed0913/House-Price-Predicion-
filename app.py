import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    menu_items={
        'Get help': None,
        'Report a Bug': None,
        'About': 'House Price Prediction — University ML Project'
    }
)

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        metrics = joblib.load('metrics.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        predictions_df = joblib.load('predictions.pkl')
        return model, scaler, metrics, feature_columns, predictions_df
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Make sure you have run the notebook to generate the .pkl files.")
        st.stop()

model, scaler, metrics, feature_columns, predictions_df = load_artifacts()

# App Title
st.title("🏠 California House Price Prediction")
st.markdown("This app predicts the median house value for a district in California based on 1990 census data.")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🔍 Predict Price", "📊 Visualizations", "ℹ️ Model Info"])

if page == "🔍 Predict Price":
    st.header("Enter House Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Location")
        longitude = st.slider("Longitude", -124.35, -114.31, -119.57, 0.01, key="longitude")
        latitude = st.slider("Latitude", 32.54, 41.95, 35.63, 0.01, key="latitude")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'],
            key="ocean_proximity"
        )
        
    with col2:
        st.subheader("District Stats")
        housing_median_age = st.slider("Housing Median Age", 1.0, 52.0, 28.0, 1.0, key="housing_age")
        total_rooms = st.number_input("Total Rooms", min_value=1, value=2600, step=100, key="total_rooms")
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=500, step=50, key="total_bedrooms")
        population = st.number_input("Population", min_value=1, value=1400, step=100, key="population")
        households = st.number_input("Households", min_value=1, value=500, step=50, key="households")
        
    with col3:
        st.subheader("Demographics")
        median_income = st.slider("Median Income (Tens of thousands $)", 0.5, 15.0, 3.8, 0.1, key="median_income")

    # Helper function to run prediction
    def run_prediction():
        """Build input, scale, predict, and store result in session_state."""
        input_data = pd.DataFrame({
            'median_income': [median_income],
            'housing_median_age': [housing_median_age],
            'avg_rooms': [total_rooms / households if households > 0 else 0],
            'avg_bedrooms': [total_bedrooms / households if households > 0 else 0],
            'population': [population],
            'avg_occupancy': [population / households if households > 0 else 0],
            'latitude': [latitude],
            'longitude': [longitude]
        })
        
        # Add ocean_proximity one-hot columns (only those used during training)
        ocean_categories = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
        for cat in ocean_categories:
            col_name = f"ocean_proximity_{cat}"
            # Only add columns that exist in the trained model's features
            if col_name in feature_columns:
                input_data[col_name] = 1 if cat == ocean_proximity else 0
            
        input_data['rooms_per_household'] = input_data['avg_rooms']
        input_data['bedrooms_per_room'] = input_data['avg_bedrooms'] / input_data['avg_rooms'].replace(0, 1)
        input_data['population_per_household'] = input_data['avg_occupancy']
        
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]
        
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_columns)
        prediction = max(0, model.predict(input_scaled)[0])
        
        # Save prediction and input summary in session_state so it persists
        st.session_state['last_prediction'] = prediction
        st.session_state['last_inputs'] = {
            'Longitude': longitude, 'Latitude': latitude,
            'Ocean Proximity': ocean_proximity,
            'Housing Median Age': housing_median_age,
            'Total Rooms': total_rooms, 'Total Bedrooms': total_bedrooms,
            'Population': population, 'Households': households,
            'Median Income': median_income
        }

    # Predict button
    if st.button("🏠 Predict Price", use_container_width=True, type="primary"):
        run_prediction()

    # Always show last prediction if available (persists across reruns)
    if 'last_prediction' in st.session_state:
        st.success(f"### Predicted Median House Value: ${st.session_state['last_prediction']:,.0f}")
        st.caption("Change the values above and click **Predict Price** again to get a new prediction.")
        
elif page == "📊 Visualizations":
    st.header("Data & Model Visualizations")
    
    st.subheader("1. Actual vs Predicted Values (Test Set)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.3, color='steelblue')
    ax1.plot([predictions_df['actual'].min(), predictions_df['actual'].max()], 
             [predictions_df['actual'].min(), predictions_df['actual'].max()], 
             'r--', lw=2)
    ax1.set_xlabel('Actual Value ($)')
    ax1.set_ylabel('Predicted Value ($)')
    ax1.set_title('Linear Regression Performance')
    st.pyplot(fig1)
    plt.close(fig1)
    
    st.subheader("2. Sample Distribution: Test Set Actual Prices")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(predictions_df['actual'], bins=50, kde=True, color='coral', ax=ax2)
    ax2.set_xlabel("Median House Value ($)")
    ax2.set_title("Distribution of Actual House Values")
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.subheader("3. Correlation Heatmap")
    # Load dataset for correlation heatmap
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing(as_frame=True)
    corr_df = california.frame.copy()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    corr_matrix = corr_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, center=0, square=True, linewidths=0.5, ax=ax3)
    ax3.set_title('Feature Correlation Heatmap')
    st.pyplot(fig3)
    plt.close(fig3)

elif page == "ℹ️ Model Info":
    st.header("Model Information")
    st.markdown("The underlying model is a **Linear Regression** trained on the California Housing Dataset.")
    
    st.subheader("Performance Metrics (Test Set)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${metrics['MAE']:,.0f}")
    col2.metric("MSE", f"${metrics['MSE']:,.0f}")
    col3.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    col4.metric("R² Score", f"{metrics['R2']:.4f}")
    
    st.markdown("---")
    st.subheader("Features Used")
    st.write(feature_columns)
