import streamlit as st
import joblib
import pandas as pd
from source import  Figure_f, fig_target, fig_2


st.set_page_config(page_title="Academic Success Prediction", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    body, p, div {
        font-size: 13px !important;
    }

    .metric-label {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-value {
        font-size: 32px !important;
        t-weight: bold;
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title ('Academic Success Analysis and Prediction: A Data-Driven Approach')
st.write("######")

col1, col2, col3 = st.columns([5, 1, 10])
with col1:
    st.subheader('Problem')
    st.write('Educational institutions struggle with high dropout rates, impacting both students futures and institutional success.')
    st.subheader('\nKey Business Questions')
    st.markdown('- What are the key factors affecting student success? (Grades, socio-economic status, parental education?)\n'
                 '\n- Can we identify patterns in student dropouts vs. graduates?\n'
                 '\n- Can we predict which students are at risk?\n'
                 '\n- Are there missing pieces? (We lack behavioral & psychological data—how does this impact our insights?)\n')
with col3:
    st.plotly_chart(fig_target)

col1, col2, col3 = st.columns([10,1,5])
with col1:
     st.plotly_chart(Figure_f)
     st.subheader('\nAcademic performance is a key predictor of success. Many students disengage early, showing warning signs of dropping out before the end of the 1st semester.')

with col3:
    st.header('Exploratory Data Analysis (EDA)')
    st.write('Objective: Identify key factors influencing academic success (Graduated, Dropout, Enrolled).\n'
             '\nMethodology: Analyzed 38 features (academic, demographic, socio-economic) using pandas, Plotly, Seaborn, statistics. Explored distributions, correlations, and category impacts.')
    st.write('Key Insights:'
             '\nAcademic performance is the strongest predictor of success (Curricular Units).'
             '\nEconomic factors (GDP, unemployment) are less important.'
             '\nGraduates mostly single, attend daytime classes, have educated parents, up-to-date tuition, and pass '
             'curricular units. Scholarships boost success. '
             '\nDropouts: higher among married students, struggle in certain courses, fail early evaluations, and often have overdue tuition.'
             '\nCategory "Enrolled" remains less predictable, with moderate failure rates and no strong defining factors.')
    st.markdown('\n'
                '\nFailure rates increase as students move through their studies.'
                '\nStudents disengage early.')

st.write("######")

st.header('Machine Learning Model. Logistic Regression, Random Forrest Classifiers ')
st.write('By training models on historical data, we developed a system to analyze new student data and predict outcomes, enabling institutions to take proactive steps in student engagement. After testing two ML models, we selected the most accurate one for final predictions based on performance.')
col1, col2, col3, col4 = st.columns([5,5,1,5])

with col1:
    st.subheader("Logistic Regression")
    st.markdown("""
    The model performed well in predicting Graduates with 88% precision and 80% recall. It struggled with Enrolled students, achieving 64% precision and 49% recall, indicating a high number of false positives and false negatives.
    Dropouts were classified with 82% precision and 94% recall, indicating a strong performance in identifying this class.
    The confusion matrix shows that many Enrolled students were misclassified as Dropouts or Graduates.
    """)

with col2:
    st.subheader("The Random Forest")
    st.markdown("""The model performed well in predicting Graduates with 90% precision and 81% recall. Enrolled students were predicted with lower precision 64% and recall 63%, suggesting difficulty in distinguishing them from the other classes.
    Dropouts were classified with 86% precision and 92% recall, indicating that the model is very good at predicting this group.
    The confusion matrix shows some level of misclassifications between Enrolled and the other two categories.""")


with col4:
    st.markdown('Logistic Regression Accuracy')
    st.markdown('<p class="metric-value">80.83%</p>', unsafe_allow_html=True)

    st.markdown('Random Forest Accuracy')
    st.markdown('<p class="metric-value">83.09%</p>', unsafe_allow_html=True)

st.write("#####")

st.header('Feature Importance Analysis')

col1, col2, col3 = st.columns([5,1,10])

with col1:
    st.subheader('Curricular units 2nd sem (approved) - #1 feature to predict success/dropout')
    st.markdown('According to Decision Tree Classifier: if student has 5 or more scores in Curricular units 2nd sem (approved), they are classified as Graduate. If Curricular units 2nd sem (approved) - 0 or 1 score, the student is classified as Dropout. Otherwise, Tuition Fees up to date becomes a deciding factor. If Tuition Fees are up to date, prediction will be Enrolled; if not, prediction will be Dropout.\n'
                '\nCurricular performance (units, grades, evaluations) is the strongest predictor of success — both in machine learning and statistical correlation. '
                'Demographic and financial factors matter but are secondary predictors. There are no major contradictions between the two approaches, making the models reliable for decision-making.')

with col3:
    st.plotly_chart(fig_2)


col1, col2 = st.columns(2)
with col1:
    st.header("Student Success / Dropout Prediction")
    st.write(
            'The prediction is based on the Random Forest model, which achieves 83% accuracy. '
            'An automatic prediction was made for a new dataset. To predict the outcome for a specific student\'s '
            'success, values can be manually adjusted. For this prediction, the top features identified by the model\'s classifiers were used for editing.')

    st.markdown("""
        <style>
        .big-button {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            background-color: #66c2a5; 
            color: white;
            font-size: 16px; 
            font-weight: bold;
            padding: 10px 20px; 
            border-radius: 10px; 
            border: 2px solid #66c2a5; 
            transition: all 0.3s ease-in-out;
            text-decoration: none; 
            width: 50%; 
        }

        .big-button:hover {
            background-color: #559782; /* Hover color */
            border-color: #559782; /* Hover border color */
            color: white;
        }

        .big-button a {
            color: black; 
            text-decoration: none; 
        }
        </style>
        <div style='text-align: center;'>
            <a href="/Predict_Success" class="big-button">
                <span> Full Prediction Tool</span>
            </a>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Load the trained model
    rfc_model = joblib.load("random_forest_model.pkl")

    # Class label mapping
    class_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

    st.subheader("Student-specific Prediction Preview")
    # Preview Inputs (Only a few important ones)
    admission_grade = st.slider("Admission Grade", 95, 190, 130)
    age_at_enrollment = st.slider("Age at Enrollment", 17, 70, 20)
    approved_units_1st = st.slider("1st Semester Approved Units", 0, 26, 1)

    # Define all features with default values
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

    # Default values for missing features
    default_values = {feature: 0 for feature in training_features}
    default_values.update({
        'Admission grade': admission_grade,
        'Age at enrollment': age_at_enrollment,
        'Curricular units 1st sem (approved)': approved_units_1st,
        'Tuition fees up to date': 1,  # Assume tuition is paid
    })


    preview_df = pd.DataFrame([default_values])[training_features]

    # Predict
    predicted_class = rfc_model.predict(preview_df)[0]
    predicted_label = class_mapping[predicted_class]

    # Display Prediction
    st.markdown(f"""
        <div style='text-align: left; font-size: 20px; font-weight: bold; color: black;'>
            Preview Prediction: <span style='color: red;'>{predicted_label}</span>
        </div>
    """, unsafe_allow_html=True)


st.header('Conclusions')
st.subheader ('1. Academic Performance Is The Key')
st.write('EDA and machine learning models confirm that academic performance is the strongest predictor of student success. Correlation analysis and visualizations highlight the impact of curricular units, grades, and evaluations, while machine learning reinforces these findings. From a business perspective, tracking student progress through their educational path is critical. Other factors, including economic conditions, play a lesser role in determining outcomes. Discovering deeper insights into the student path, prioritizing academic engagement analysis, and considering additional support strategies can significantly impact success and serve as a turning point in student retention and graduation rates.')
col1, col2 = st.columns(2)
with col1:
     st.subheader('2. Early Disengagement (1st Semester)')
     st.write('Dropout and graduation are strongly influenced by academic performance. Students disengage early, the data is limited.\n'
              '\nFactors to Watch:\n'
              '- Engagement & Support: Low engagement and lack of academic support contribute to dropout.  Mentors, counselors, and help centers can keep them on track. Attendance, activities, and mental health - potential areas to explore.\n'
              '- Education Intensity & Overload: A demanding curriculum can lead to burnout. Next step is to explore curricular unit completion rates and dropout trends. ')

with col2:
    st.subheader('3. Courses: What Works, What Fails')
    st.write('Some classes succeed. Others lose students fast. Understand reasons is to inform improvements in curriculum design and teaching strategies.\n'
             '\nKey Considerations:\n'
             '- Track Course Popularity: What makes a course work? Analyze courses with high graduation vs. high dropout rates, reveal differences in curriculum structure, teaching methods, or assessment types.\n'
             '- Consider Fix/Eliminate the Failing Ones: Some classes can be changed to keep students engaged. Demand and cost-efficiency weigh into the decision. ')

col1, col2 = st.columns(2)
with col1:
    st.subheader('4. The Enrolled: The Overlooked Group')
    st.write('This group perform well, but doesn’t finish. It makes up 19.5% of students. '
             'Machine learning models can’t distinguish it clearly, predictions are less stable.\n'
             '- Why Focus on it? Students are close to success. Helping them finish costs less than saving a failing student.\n'
             '- Analyse the Difference: Compare them to graduates to improve predictions, and graduation rates.')

with col2:
    st.subheader('5. Key Metrics to Track & Expand')
    st.write('To better understand and prevent dropouts, additional metrics to consider:\n'
             '- Track Performance Early: First semester to investigate deeper.\n'
             '- Watch Student Engagement: Consider incorporate satisfaction surveys, psychological assessments, and engagement metrics to detect warning signs.\n'
             '- Measure Support: Mentors, counselors, academic help, support system — do they make a difference?')

