import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load the trained model
model = xgb.XGBClassifier()
current_dir = os.path.dirname(os.path.abspath(__file__))
# Join the path to the model file
model_path = os.path.join(current_dir, "titanic_xgboost_model.json")

try:
    if not os.path.exists(model_path):
        st.error(f"File not found at: {model_path}")
        # Debugging: Show what files ARE there
        st.write("Files in current directory:", os.listdir(current_dir))
    else:
        model.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

def preprocess_data(df):
    """
    Applies the same preprocessing steps as the training script.
    """
    # Create a copy to avoid SettingWithCopy warnings
    data = df.copy()
    
    # Fill missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # Handle Embarked missing values
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Drop unnecessary columns if they exist
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])
    
    # Encode Sex (Manual mapping to ensure consistency with training)
    sex_mapping = {'male': 1, 'female': 0}
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map(sex_mapping)
        
    # Encode Embarked (Manual mapping approx. based on LabelEncoder)
    # Note: In production, you should save and load the exact encoder objects.
    # For this assignment, we assume standard mapping: C=0, Q=1, S=2
    emb_mapping = {'C': 0, 'Q': 1, 'S': 2}
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].map(emb_mapping)
        
    # Ensure column order matches training
    expected_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Filter to only expected columns and reorder
    data = data[expected_cols]
    
    return data

# 2. Streamlit App Layout
st.title("Titanic Survival Prediction App")
st.write("Upload a CSV file containing passenger data to predict survival.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(input_df.head())

    # Button to trigger prediction
    if st.button("Predict Survival"):
        try:
            # Preprocess
            processed_data = preprocess_data(input_df)
            
            # Predict
            predictions = model.predict(processed_data)
            
            # Add predictions to original dataframe
            output_df = input_df.copy()
            output_df['Survival_Prediction'] = predictions
            output_df['Survival_Label'] = output_df['Survival_Prediction'].map({1: 'Survived', 0: 'Did Not Survive'})
            
            st.subheader("Prediction Results")
            st.write(output_df[['PassengerId', 'Survival_Label']].head())
            
            # Allow download
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='titanic_predictions.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Ensure your CSV has columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")
