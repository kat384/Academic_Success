
import streamlit as st
import plotly.express as px

from source import df


numerical_features = df.columns[[7, 13, 20, 26, 32, 34, 35, 36]]
categorical_features = df.columns.drop(numerical_features).drop('id').drop('Target')


numerical_explanations = {
    "Previous qualification (grade)": "The data is mostly evenly distributed with outliers in each category. Graduates tend to have higher qualification grades, while dropouts have lower grades. Enrolled students have a wider range of grades, slightly lower than the graduates but higher than the dropouts.",
    "Admission grade": "The distribution is evenly spread across categories, with a tendency for dropouts to have slightly lower than average admission grades, although there are some dropouts with higher scores. Graduates tend to have average or above-average scores.",
    "Age at enrollment": "The distribution is right-skewed with outliers in all categories. Older students tend to have a higher dropout rate, while younger students have a higher chance of graduating. Enrolled students tend to be slightly older than graduates.",
    "Curricular units 1st sem (grade)": "59% of dropouts have a 0 grade in the 1st semester. Only 41% of dropouts perform at an average level, closer to graduates. Enrolled students typically have a broader range of grades, but most have average scores, with some having 0s. Graduates tend to perform higher than average, though there are outliers on both ends (extremely good and failed).",
    "Curricular units 2nd sem (grade)": "69% of dropouts have a 0 grade in the 2nd semester. 10% of dropouts who performed at an average level failed with a score of 0. In comparison, graduates show better performance than in the 1st semester, though the proportion failing is similar.",
    "Unemployment rate": "The distribution of unemployment rate is similar across graduates, dropouts, and enrolled students. This suggests that unemployment does not have a significant impact on whether a student graduates, remains enrolled, or drops out.",
    "Inflation rate": "The inflation rate appears consistently across all categories, with no significant difference between graduates, dropouts, and enrolled students. It shows minimal influence on the likelihood of student success.",
    "GDP": "The distribution of GDP is similar for all categories, indicating that GDP does not have a strong influence on academic outcomes or the likelihood of graduation, enrollment, or dropout."
}



categorical_explanations = {
    "Marital status": "Most students belong to a single category—either married or single. Given the mean age of 22, we can assume most are single. Graduates are more common in this group, while Dropouts have a higher representation in the opposite category, presumably married.",
    "Application mode": "The majority of students applied through a particular mode (1), with Graduates being the most represented. The remaining students applied mainly through two other modes (17 and 39), with more than half of Dropouts using mode 39. A smaller portion of students applied through various other modes.",
    "Application order": "No specific insight was provided. You can add details here.",
    "Course": "Certain courses have higher graduation rates, while others show a higher proportion of Dropouts. This could indicate a need for further investigation into the quality of education and its impact on student success.",
    "Daytime/evening attendance": "Most students attend classes in a specific mode (1), with a higher number of Graduates in this category.",
    "Previous qualification": "Most students have a similar previous qualification, with only a small percentage falling into other categories.",
    "Nacionality": "The distribution of students across nationalities mirrors the overall distribution of the Target variable, meaning nationality does not strongly predict whether a student will Graduate, Drop Out, or remain Enrolled.",
    "Mother's qualification": "Parental education levels are spread across four main categories, following a distribution similar to the overall Target variable. However, category 34 for both parents is predominantly associated with Dropouts, suggesting that students from this background are more likely to drop out.",
    "Father's qualification": "Parental education levels are spread across four main categories, following a distribution similar to the overall Target variable. However, category 34 for both parents is predominantly associated with Dropouts, suggesting that students from this background are more likely to drop out.",
    "Mother's occupation": "Students' parents’ occupations are spread across a range of categories, but each occupation group has a similar distribution of Graduates, Dropouts, and Enrolled students as the overall Target variable. This suggests that occupation is not a strong predictor of student outcomes. However, students whose parents fall under category (0, presumably unemployed) have a higher likelihood of dropping out, though this category represents a minor portion of the dataset.",
    "Father's occupation": "Students' parents’ occupations are spread across a range of categories, but each occupation group has a similar distribution of Graduates, Dropouts, and Enrolled students as the overall Target variable. This suggests that occupation is not a strong predictor of student outcomes. However, students whose parents fall under category (0, presumably unemployed) have a higher likelihood of dropping out, though this category represents a minor portion of the dataset.",
    "Displaced": "The dataset contains a mix of displaced and non-displaced students, with a roughly 60/40 ratio. Graduates are more frequent in one group, presumably non-displaced students. Dropout rates, however, do not appear to be significantly affected by this factor.",
    "Educational special needs": "The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis.",
    "Debtor": "Most students fall into a single category. In the other category (presumably debtors), Dropout rates are higher. However, this feature does not seem to have a strong influence on the Target variable.",
    "Tuition fees up to date": "The dataset consists of two categories, with most students belonging to one (presumably those with tuition fees up to date). However, 40% of Dropouts fall into the other category, indicating that overdue tuition fees might be linked to a higher risk of dropping out.",
    "Gender": "The dataset contains a mix of genders, but one gender is more represented. Graduates are more frequent in this category, while Dropouts are equally represented across both genders. This suggests that gender has an influence on graduation rates but not on dropout rates—students of one gender are more likely to Graduate or stay Enrolled, but Dropout rates are similar for both.",
    "Scholarship holder": "Most students belong to one category, with Dropouts and Graduates appearing in similar proportions. However, 40% of Graduates fall into the other category (presumably scholarship holders), suggesting that students with scholarships are more likely to graduate.",
    "International": "The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis.",
    "Curricular units 1st sem (credited)": "7% of Dropouts (2.6% of all students) are still enrolled in the 1st semester, and 8% in the 2nd semester.",
    "Curricular units 1st sem (enrolled)": "30% of Dropouts never took an evaluation, while 60% failed in the 1st semester.",
    "Curricular units 1st sem (evaluations)": "Similarly, 33% of Dropouts never took an evaluation in the 2nd semester, and 70% failed.",
    "Curricular units 1st sem (approved)": "This suggests that many students disengage early, showing warning signs of dropping out even before the end of the 1st semester.",
    "Curricular units 1st sem (without evaluations)": "This suggests that many students disengage early, showing warning signs of dropping out even before the end of the 1st semester.",
    "Curricular units 2nd sem (credited)": "The dataset consists of two categories, with most students belonging to one (presumably those with tuition fees up to date). However, 40% of Dropouts fall into the other category, indicating that overdue tuition fees might be linked to a higher risk of dropping out.",
    "Curricular units 2nd sem (enrolled)": "Most students belong to one category, with Dropouts and Graduates appearing in similar proportions. However, 40% of Graduates fall into the other category (presumably scholarship holders), suggesting that students with scholarships are more likely to graduate.",
    "Curricular units 2nd sem (evaluations)": "The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis.",
    "Curricular units 2nd sem (approved)": "The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis.",
    "Curricular units 2nd sem (without evaluations)": "The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis."
}

    # Set page configuration
st.set_page_config(page_title="Feature Visualization", page_icon="📊", layout="wide")

st.header("Exploratory Data Analysis")

feature_type = st.selectbox('Select feature type for analysis', ['Numerical', 'Categorical'])

if feature_type == 'Numerical':
        selected_feature = st.selectbox('Select Numerical Feature for Analysis', options=numerical_features)

        mean_val = df[selected_feature].mean()
        median_val = df[selected_feature].median()
        std_val = df[selected_feature].std()
        min_val = df[selected_feature].min()
        max_val = df[selected_feature].max()

        col1, col2 = st.columns(2)
        col1.markdown(f"""
        **Summary Statistics for {selected_feature}:**
        - **Mean:** {mean_val:.2f}
        - **Median:** {median_val:.2f}
        - **Standard Deviation:** {std_val:.2f}
        - **Min:** {min_val:.2f}
        - **Max:** {max_val:.2f}
        """)
        col2.markdown(
            f"### **Feature Insight:**\n{numerical_explanations.get(selected_feature, 'No explanation available.')}")


        hist_plot = px.histogram(df, x=selected_feature, color='Target', barmode='group',
                                 opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2)

        violin_plot = px.violin(df, y=selected_feature, color='Target', box=True,
                                points="all", color_discrete_sequence=px.colors.qualitative.Set2)

        st.plotly_chart(hist_plot)
        st.plotly_chart(violin_plot)


elif feature_type == 'Categorical':
        selected_feature = st.selectbox('Select Categorical Feature for Analysis', options=categorical_features)

        col1, col2 = st.columns(2)
        col1.markdown(
            f"### **Feature Insight:**\n{categorical_explanations.get(selected_feature, 'No explanation available.')}")

        selected_plot = px.histogram(df, x=selected_feature, color='Target', barmode='group',
                                     opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2)

        st.plotly_chart(selected_plot)