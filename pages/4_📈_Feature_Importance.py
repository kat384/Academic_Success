import streamlit as st

from source import fig_fi, dt_plot


st.set_page_config(page_title="4_📈_Feature_Importance.py", page_icon="📈", layout="wide")

st.title('Feature Importance for Prediction')

col1, col2, col3 =st.columns([5,1,10])
with col1:
    st.subheader('Feature Importance by Random Forest Model')
    st.write('Curricular performance (units, grades, evaluations) is the strongest predictor of success — both in machine learning and statistical correlation. '
            'Demographic and financial factors matter but are secondary predictors. There are no major contradictions between the two approaches, making the model reliable for decision-making.')
with col3:
    st.plotly_chart(fig_fi)


col1, col2, col3 =st.columns([10,1,5])
with col1:
    st.pyplot(dt_plot)
with col3:
    st.subheader('Feature Importance by Decision Tree')
    st.write('If feature Curricular units 2nd sem (approved) has 5 scores or more, the student is classified as Graduate. If Curricular units 2nd sem (approved) has 0 or 1 score, the student is classified as Dropout. Otherwise, check the feature Tuition Fees up to date. If Tuition Fees up to date, prediction will be Enrolled; if not, prediction will be Dropout.')
    with st.expander('Code'):
        st.write('''
                initial_features = list(df_test.columns)
                dt = DecisionTreeClassifier(max_depth=3)
                dt.fit(df[initial_features], df.Target)
            ''')