import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components

# Streamlit app
st.set_page_config(page_title="Vacation Preference Predictor", layout="wide", page_icon="🌴")
st.markdown(
    """
    <style>
    body {
        background-color: #e8f5e9;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    .css-18e3th9 {
        padding: 2rem;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stApp {
        border: 5px solid #1f77b4;
        border-radius: 15px;
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
        padding: 20px;
    }
    h1 {
        color: #1f77b4;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #145a86;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌄 Vacation Preference Predictor 🏖️")

# Layout
input_col, visual_col = st.columns([1, 2])

# File uploader for CSV file
uploaded_file = st.file_uploader(r'C:\Users\HP\Downloads\mountains_vs_beaches_preferences.csv', type=["csv"])

if uploaded_file is not None:
    # Load the dataset from the uploaded file
    df = pd.read_csv(uploaded_file)

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Prepare data for training
    X = df.drop(columns=['Preference'])
    y = df['Preference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Input fields for user data
    with input_col:
        st.header("📋 Input Your Information")
        st.markdown(
            "<div style='font-size:14px; color:#6c757d'>Fill in the details to predict your vacation preference.</div>",
            unsafe_allow_html=True,
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        income = st.number_input("Income", min_value=10000, max_value=200000, value=50000, step=1000)
        education_level = st.selectbox("Education Level", label_encoders['Education_Level'].classes_)
        travel_frequency = st.slider("Travel Frequency (trips per year)", min_value=0, max_value=12, value=3)
        preferred_activities = st.selectbox("Preferred Activities", label_encoders['Preferred_Activities'].classes_)
        vacation_budget = st.number_input("Vacation Budget ($)", min_value=500, max_value=20000, value=3000, step=100)
        location = st.selectbox("Location", label_encoders['Location'].classes_)
        proximity_to_mountains = st.slider("Proximity to Mountains (miles)", min_value=0, max_value=500, value=50)
        proximity_to_beaches = st.slider("Proximity to Beaches (miles)", min_value=0, max_value=500, value=50)
        favorite_season = st.selectbox("Favorite Season", label_encoders['Favorite_Season'].classes_)
        pets = st.selectbox("Do you own pets?", ["No", "Yes"])
        environmental_concerns = st.selectbox("Environmental Concerns", ["No", "Yes"])

    # Convert user input to model input
    user_data = pd.DataFrame({
        "Age": [age],
        "Gender": [label_encoders['Gender'].transform([gender])[0]],
        "Income": [income],
        "Education_Level": [label_encoders['Education_Level'].transform([education_level])[0]],
        "Travel_Frequency": [travel_frequency],
        "Preferred_Activities": [label_encoders['Preferred_Activities'].transform([preferred_activities])[0]],
        "Vacation_Budget": [vacation_budget],
        "Location": [label_encoders['Location'].transform([location])[0]],
        "Proximity_to_Mountains": [proximity_to_mountains],
        "Proximity_to_Beaches": [proximity_to_beaches],
        "Favorite_Season": [label_encoders['Favorite_Season'].transform([favorite_season])[0]],
        "Pets": [1 if pets == "Yes" else 0],
        "Environmental_Concerns": [1 if environmental_concerns == "Yes" else 0]
    })

    # Make prediction
    with input_col:
        if st.button("Predict Preference", key="predict_btn"):
            prediction = model.predict(user_data)[0]
            preference = "Mountains" if prediction == 1 else "Beaches"
            st.success(f"✨ Your predicted vacation preference is: **{preference}**")

    # Data Visualization
    with visual_col:
        st.header("📊 Vacation Preference Trends")
        st.markdown(
            "<div style='font-size:14px; color:#6c757d'>Explore trends based on various demographic and lifestyle factors.</div>",
            unsafe_allow_html=True,
        )

        # Enhanced visualizations
        # Plot preference by age
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df, x='Age', hue='Preference', multiple='stack', ax=ax, palette="viridis", kde=True)
        ax.set_title("Preference by Age", fontsize=16, color="#1f77b4", weight='bold')
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        # Plot preference by income
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df, x='Income', hue='Preference', multiple='stack', ax=ax, palette="plasma", kde=True)
        ax.set_title("Preference by Income", fontsize=16, color="#1f77b4", weight='bold')
        ax.set_xlabel("Income", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        # Plot preference by travel frequency
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df, x='Travel_Frequency', hue='Preference', multiple='stack', ax=ax, palette="cool", kde=True)
        ax.set_title("Preference by Travel Frequency", fontsize=16, color="#1f77b4", weight='bold')
        ax.set_xlabel("Travel Frequency", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        # Plot preference by education level
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='Education_Level', hue='Preference', ax=ax, palette="magma")
        ax.set_title("Preference by Education Level", fontsize=16, color="#1f77b4", weight='bold')
        ax.set_xlabel("Education Level", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

        # Plot preference by location
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='Location', hue='Preference', ax=ax, palette="cubehelix")
        ax.set_title("Preference by Location", fontsize=16, color="#1f77b4", weight='bold')
        ax.set_xlabel("Location", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)

else:
    st.warning("Please upload your CSV file to get started.")
