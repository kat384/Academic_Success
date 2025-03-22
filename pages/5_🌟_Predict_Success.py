import streamlit as st
import pandas as pd
import joblib

from source import result, df_test

st.set_page_config(page_title="Predict_Success", page_icon="🌟", layout="wide")

st.title('Test and Predict')
st.subheader('We will proceed with Random Forest Model as it demonstrated stability and higher Accuracy. We\'ll take new dataset of students\' performance to check the model. We\'ll get results - predict class Graduated, Dropout or Enrolled for all students in dataset automatically.')
with st.expander("Test dataset, description"):
    st.dataframe(df_test, use_container_width=True)

code1 = '''    
df_test = pd.read_csv('test.csv')
test = df_test.drop(columns='id')
preds = rfc.predict(test)
preds_labels = lab_enc.inverse_transform(preds)

result = pd.DataFrame({
    'id': df_test.id,
    'Target': preds
})
result['Target'] = preds_labels
result'''

st.code(code1, language="python")

st.dataframe(result, use_container_width=True)
rfc_model = joblib.load("random_forest_model.pkl")
class_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}


st.title("Student-Specific Success Prediction")
st.write(
    'Below are the top features influencing the prediction by Random Forest Feature Importance Analysis. All other features are predefined with median value. Hidden features have less influence on outcome, in real-life situation they all can be manually edited for precise result. \n\n'
    '**Business Impact:**\n'
    '- Institutions can focus on early intervention for at-risk students.\n'
    '- Profit forecasting.\n'
)

st.write("Adjust the inputs below to predict if a student will Dropout, Graduate, or stay Enrolled.")

important_feature_stats = {
    'Curricular units 2nd sem (approved)': {'min': 0, 'max': 20, 'default': 10},
    'Curricular units 2nd sem (grade)': {'min': 0.0, 'max': 18.0, 'default': 9.0},
    'Curricular units 1st sem (approved)': {'min': 0, 'max': 26, 'default': 13},
    'Curricular units 1st sem (grade)': {'min': 0.0, 'max': 18.0, 'default': 9.0},
    'Curricular units 2nd sem (evaluations)': {'min': 0, 'max': 33, 'default': 16},
    'Curricular units 1st sem (evaluations)': {'min': 0, 'max': 45, 'default': 22},
    'Admission grade': {'min': 95.0, 'max': 190.0, 'default': 130.0},
    'Previous qualification (grade)': {'min': 95.0, 'max': 190.0, 'default': 140.0},
    'Age at enrollment': {'min': 17, 'max': 70, 'default': 20},
    'Tuition fees up to date': {'min': 0, 'max': 1, 'default': 1},
}

st.subheader("Enter Student Data")
important_features = {}
for feature, stats in important_feature_stats.items():
    step = 1 if isinstance(stats['min'], int) else 0.1
    important_features[feature] = st.slider(
        feature,
        min_value=stats['min'],
        max_value=stats['max'],
        value=stats['default'],
        step=step
    )


training_features = [
    'Marital status', 'Application mode', 'Application order',
    'Course', 'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Admission grade',
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

hidden_features_defaults = {feature: 0 for feature in training_features if feature not in important_features}
final_input = {**important_features, **hidden_features_defaults}
user_df = pd.DataFrame([final_input])
user_df = user_df[training_features]  # Reorder columns to match training order

# Prediction
if st.button("🔍 Predict Student Outcome"):
    predicted_class = rfc_model.predict(user_df)[0]
    predicted_label = class_mapping[predicted_class]  # Convert numeric prediction to human-readable label

    # Displaying Prediction with Enhanced UI
    st.markdown(f"""
    <div style='text-align: center; font-size: 24px; font-weight: bold; color: #0078D4;'>
        🎯 Predicted Outcome: <span style='color: #FF5733;'>{predicted_label}</span>
    </div>
    """, unsafe_allow_html=True)
