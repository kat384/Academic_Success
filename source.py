# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib

# Title = ''' Student Success Analysis and Prediction: A Data-Driven Approach'''

# Project Structure
# 1. Load libraries, read data.
#
# 2. Exploratory Data Analysis (EDA):
#
#   2.1. Academic Success Feature 'Target'
#   2.2. Numerical Features vs.'Target'
#   2.3. Categorical Features vs.'Target'
#   2.3. Heatmap & Correlation Analysis
#
# 3. Machine Learning
# 3.1. Split data, Train & Evaluate Models: 
#  - Logistic Regression Classifier
#  - Random Forest Classifier
# 3.3. Feature Importance
# 3.4. Test & Predict
#
# 4. Executive Summary
# 4.1. Key Findings from EDA.
# 4.2. Top Factors Affecting Student Success.
# 4.3. Best ML Model & Insights.
# 4.4. Future Recommendations for Institutions.'''


# ## 1.Load libraries, read data.

df = pd.read_csv('train.csv')
df.head()

df_info = df.info()

df_describe = df.describe(include='all').T

Obs_read = '''Dataset has 38 columns, id column brings no valuable information.
Some features are numerical, some are categorical, encoded.
In some features data distribution is skewed as mean and median significantly different.
The Target feature which is key in our analysis, contains 3 unique classes and is categorical.
There are no NaN values, the dataset is clean.'''

# ## 2. Exploratory Analysis (EDA):

# ### 2.1. Academic success feature 'Target'
# Count, Pie of Academic Success Categories
target_counts = df['Target'].value_counts()

colors = px.colors.qualitative.Set2

fig_target = make_subplots(rows=1, cols=2,
                           subplot_titles=["Count of Academic Success Categories",
                                           "Percentage Distribution of Academic Success"],
                           specs=[[{"type": "bar"}, {"type": "pie"}]])

fig_target.add_trace(go.Bar(
    x=target_counts.index.astype(str),
    y=target_counts.values,
    marker=dict(color=colors),
    text=target_counts.values,
), row=1, col=1)

fig_target.add_trace(go.Pie(
    labels=target_counts.index.astype(str),
    values=target_counts.values,
    marker=dict(colors=colors),
    textinfo='percent+label',
    hole=0.3
), row=1, col=2)

fig_target.update_layout(
    height=460, width=1000,
    title_text="Academic Success Analysis",
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

### 2.2. Numerical features vs 'Target'.

numerical_features = df.columns[[7, 13, 20, 26, 32, 34, 35, 36]]
fig_num, axes = plt.subplots(len(numerical_features), 2, figsize=(14, 3 * len(numerical_features)))

for i, feature in enumerate(numerical_features):

    sns.histplot(data=df, x=feature, palette='Set2', alpha=0.7, hue='Target', linewidth=0.5, ax=axes[i, 0])
    axes[i, 0].set_title(f'{feature} Distribution')
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel('Count')

    sns.violinplot(data=df, x=feature, palette='Set2', alpha=0.7, hue='Target', ax=axes[i, 1])
    axes[i, 1].set_title(f'{feature} Violin Plot')
    axes[i, 1].set_xlabel(feature)
    axes[i, 1].set_ylabel('')

plt.tight_layout()

# ### Observations
num_obs1 = ('\nThe data is mostly evenly distributed with outliers in each category.\n'
            '\n**Previous qualification**\n'
            '\nGraduates tend to have higher qualification grades, while dropouts have lower grades. Enrolled students have a wider range of grades, slightly lower than the graduates but higher than the dropouts.\n\'\n'
            '\n'
            '**Admission Grade:**\n'
            '\nThe distribution is evenly spread across categories, with a tendency for dropouts to have slightly lower '
            'than average admission grades, although there are some dropouts with higher scores. Graduates tend to '
            'have average or above-average scores.\n'
            '\n'
            '**Age at Enrollment:** \n'
            '\nThe distribution is right-skewed with outliers in all categories. Older students tend to have a higher '
            'dropout rate, while younger students have a higher chance of graduating. Enrolled students tend to be slightly older than graduates.\n'
            '\n'
            '**Curricular Units 1st Semester (Grade):** \n'
            '\n59% of dropouts have a 0 grade in the 1st semester. Only 41% of dropouts perform at an average level, closer to graduates. Enrolled students typically have a broader range of grades, but most have average scores, with some having 0s. Graduates tend to perform higher than average, though there are outliers on both ends (extremely good and failed).\n\'\n'
            '\n'
            '**Curricular Units 2nd Semester (Grade):** \n'
            '\n69% of dropouts have a 0 grade in the 2nd semester. 10% of dropouts who performed at an average level failed with a score of 0. In comparison, graduates show better performance than in the 1st semester, though the proportion failing is similar.\n'
            '\n'
            '**Unemployment Rate:** \n'
            '\nThe distribution of unemployment rate is similar across graduates, dropouts, and enrolled students. This suggests that unemployment does not have a significant impact on whether a student graduates, remains enrolled, or drops out.\n'
            '\n'
            '**Inflation Rate, GDP:** \n'
            '\nThe inflation rate, GDP appears consistently across all categories, with not significant difference between graduates, dropouts, and enrolled students. There are some instances showing higher level of Graduates and Dropouts, but seems like these metrics less influential on student success.\n'
            '\n')

# ### 2.3. Catgorical features vs 'Target'.
categorical_features = df.columns.drop(numerical_features).drop('id').drop('Target')

cat = plt.figure(figsize=(14, 1.5 * len(categorical_features)))
for i, feature in enumerate(categorical_features):
    plt.subplot(14, 2, i + 1)
    sns.countplot(data=df, x=feature, palette='Set2', alpha=0.7, hue='Target', edgecolor='black', linewidth=0.7)
    plt.title(f'{feature} by Academic Success')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()

cat_obs = (' **Marital Status:**  \n'
           'Most students belong to a single category—either married or single. Given the mean age of 22, we can assume most are single. Graduates are more common in this group, while Dropouts have a higher representation in the opposite category, presumably married.  \n'
           '\n'
           ' **Application Mode:**  \n'
           ' The majority of students applied through a particular mode (1), with Graduates being the most represented. The remaining students applied mainly through two other modes (17 and 39), with more than half of Dropouts using mode 39. A smaller portion of students applied through various other modes.  \n'
           '\n'
           ' **Course:**  \n'
           ' Certain courses have higher graduation rates, while others show a higher proportion of Dropouts. This could indicate a need for further investigation into the quality of education and its impact on student success.  \n'
           '\n'
           ' **Daytime/Evening Attendance:**  \n'
           ' Most students attend classes in a specific mode (1), with a higher number of Graduates in this category.  \n'
           '\n'
           ' **Previous Qualification:**  \n'
           ' Most students have a similar previous qualification, with only a small percentage falling into other categories.  \n'
           '\n'
           ' **Nationality:**  \n'
           ' The distribution of students across nationalities mirrors the overall distribution of the Target variable, meaning nationality does not strongly predict whether a student will Graduate, Drop Out, or remain Enrolled.  \n'
           '\n'
           ' **Mother\'s and Father\'s Qualification:**  \n'
           ' Parental education levels are spread across four main categories, following a distribution similar to the overall Target variable. However, category 34 for both parents is predominantly associated with Dropouts, suggesting that students from this background are more likely to drop out.  \n'
           '\n'
           ' **Mother\'s and Father\'s Occupation:**  \n'
           ' Students\' parents’ occupations are spread across a range of categories, but each occupation group has a similar distribution of Graduates, Dropouts, and Enrolled students as the overall Target variable. This suggests that occupation is not a strong predictor of student outcomes. However, students whose parents fall under category (0, presumably unemployed) have a higher likelihood of dropping out.  \n'
           '\n'
           ' **Displaced Status:**  \n'
           ' The dataset contains a mix of displaced and non-displaced students, with a roughly 60/40 ratio. Graduates are more frequent in one group, presumably non-displaced students. Dropout rates, however, do not appear to be significantly affected by this factor.  \n'
           '\n'
           ' **Educational Special Needs & International Status:**  \n'
           ' The vast majority of students belong to a single category in these features, meaning they do not provide significant variation for analysis.  \n'
           '\n'
           ' **Debtor Status:**  \n'
           ' Most students fall into a single category. In the other category (presumably debtors), Dropout rates are higher. However, this feature does not seem to have a strong influence on the Target variable.  \n'
           '\n'
           ' **Curricular Units (1st and 2nd Semester):**  \n'
           ' These features capture the student’s academic progress, showing how failure rates increase as students move through their studies.  \n'
           '\n'
           ' - 7% of Dropouts (2.6% of all students) are still enrolled in the 1st semester, and 8% in the 2nd semester. \n'
           ' - 30% of Dropouts never took an evaluation, while 60% failed in the 1st semester.  \n'
           ' - Similarly, 33% of Dropouts never took an evaluation in the 2nd semester, and 70% failed.  \n'
           ' - This suggests that many students disengage early, showing warning signs of dropping out even before the end of the 1st semester.  \n'
           '\n'
           ' **Tuition Fees Up to Date:**  \n'
           ' The dataset consists of two categories, with most students belonging to one (presumably those with tuition fees up to date). However, 40% of Dropouts fall into the other category, indicating that overdue tuition fees might be linked to a higher risk of dropping out. \n'
           '\n'
           ' **Gender:**  \n'
           ' The dataset contains a mix of genders, but one gender is more represented. Graduates are more frequent in this category, while Dropouts are equally represented across both genders. \n'
           '\n'
           ' **Scholarship Holder:**  \n'
           ' Most students belong to one category, with Dropouts and Graduates appearing in similar proportions. However, 40% of Graduates fall into the other category (presumably scholarship holders), suggesting that students with scholarships are more likely to graduate. ')

# ### 2.3. Heatmap & Correlation Analysis.
#Encoding Target feature for using in Heatmap

lab_enc = LabelEncoder()
df['Target'] = lab_enc.fit_transform(df['Target'])

d_corr = df.corr()
fig_heatmap = px.imshow(d_corr, height=600, width=700, color_continuous_scale='tealrose')

obs_heatmap = ('### Correlation to Target\n'
               '\n'
               ' **Strong Positive Correlations:**\n'
               ' - Curricular Units Semesters 1,2 (0.66 - 0.78): Strong correlation with academic success, indicating early academic performance as a key predictor of success.\n'
               '\n**Moderate Positive Correlations:**\n'
               ' - Scholarship Holder (0.39): Scholarship holders are predominantly Graduates (40% of students).\n'
               ' - Tuition Fees Up to date (0.42): No Graduates or Enrolled students have overdue tuition fees.\n'
               ' \n **Moderate Negative Correlations:**\n'
               ' - Age at Enrollment (-0.32): Younger students tend to graduate at higher rates.\n'
               ' - Gender (-0.33): Gender may influence academic success, with Graduates skewing toward one gender.\n'
               ' - Debtor (-0.25): Weak correlation, suggesting some impact but not statistically significant.\n'
               ' - Application Mode (-0.32): Application mode negatively correlates with student success, with certain modes indicating Dropouts or Graduates.\n'
               '\n **Weak or Insignificant Correlations:**\n'
               ' - Other features show weak correlations or insignificant statistical impact. Future investigations can focus on correlations across the dataset for deeper insights and further regression modeling.\n'
               '\n'
               '\n'
               '\n')
obs_heatmap2 =(' ### Correlations across Dataset\n'
               '\n'
               ' **Strong Correlations:**\n'
               ' - Curricular Units (0.95 - 0.45): High correlation between steps in the curriculum, showing that success in one step affects the next. Suggests potential multicollinearity.\n'
               ' - Age at Enrollment: Strong correlations with marital status (0.55), application mode (0.59), and attendance mode (-0.49).\n'
               ' - Parents\' Occupations (0.89): Strong positive correlation, indicating that couples often share similar occupations.\n'
               ' - International Status (0.82): Strong correlation with nationality.\n'
               ' - Previous Qualification (0.56): Positive correlation with admission grade.\n'
               ' - Course and Curricular Units (0.61 - 0.25): Correlation between course selection and academic progress.\n'
               '\n**Moderate and Weak Correlations:**\n'
               ' - Marital Status (-0.32): Correlates negatively with time of attendance.\n'
               ' - Displaced: Correlates with application mode (-0.32), order (0.35), and age (-0.37).\n'
               ' - Debtor and Tuition Fees (-0.47): Suggests possible multicollinearity.\n'
               '\n**Economical Features:**\n'
               ' - GDP, Inflation, Unemployment Rate (0.1 or lower): These have weak correlations, implying minimal impact on academic success.')


# ### 2.4. EDA Conclusions

eda_concl = ('\n'
             '### EDA Summary'
             '\n'
             '\n**Academic Performance:** '
             '\nHigher grades and completed curricular units strongly predict graduation. Dropouts often have failing grades.\n'
             '\n**Demographics:** '
             '\nYounger students graduate more; older students drop out more. Marital status and gender influence graduation rates.\n'
             '\n**Financial Factors:** '
             '\nScholarship holders graduate more, while overdue tuition correlates with dropout risk.\n'
             '\n**Course & Application Mode:** '
             '\nCertain courses and application modes have higher dropout or graduation rates.\n'
             '\n**Economic Indicators:** '
             '\nUnemployment, inflation, and GDP have minimal impact on student success.\n'
             '\nMost findings in Heatmap and through feature analysis align well, but parental education/occupation appears weaker in correlation analysis than in categorical analysis. '
             'This suggests other socioeconomic factors (not captured in the heatmap) may influence student success.')



# ## 3. Machine Learning:

# ### 3.1. Split data, Train & Evaluate Models: 
# ### Logistic Regression Classifier 

X = df.drop(['Target', 'id'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

logit = LogisticRegression(multi_class='ovr')
logit.fit(X_train, y_train)

y_pred_logit = logit.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_logit)
print(f"Accuracy: {accuracy}")

cr = classification_report(y_test, y_pred_logit)
print(f"Classification Report:\n{cr}")

cm = confusion_matrix(y_test, y_pred_logit)
print(f"Confusion Matrix:\n{cm}")

cv_scores = cross_val_score(logit, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Random Forest Classifier¶

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

joblib.dump(rfc, "random_forest_model.pkl")

rfc_model = joblib.load("random_forest_model.pkl")
y_pred_rfc = rfc.predict(X_test)

# Confusion Matrix
cm_ = confusion_matrix(y_pred_rfc, y_test)
print("Confusion Matrix:")
print(cm_)

# Classification Report
cr_ = classification_report(y_pred_rfc, y_test)
print("\nClassification Report:")
print(cr_)

# Accuracy Score
score_ = accuracy_score(y_pred_rfc, y_test)
print(f"\nAccuracy Score: {score_:.4f}")


val_score = cross_val_score(rfc, X_train, y_train, cv=5)
print(f"Cross-validation scores: {val_score}")
print(f"\nAverage Cross-Validation Score: {val_score.mean()}")

# ### 3.2. Feature importance by Random Forest Classifier

feature_names = df.drop(['Target', 'id'], axis=1).columns
feature_importances = rfc.feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_features = importance_df.head(15)
fig_fi = px.bar(top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Important Features in Predicting Student Outcomes",
                labels={'Importance': 'Feature Importance Score', 'Feature': 'Feature'},
                color='Importance',
                color_continuous_scale='sunset',
                height=450)

fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})


# ### 3.3. Test Data, Predict Academic Success 

df_test = pd.read_csv('test.csv')
df_test.info()

test = df_test.drop(columns='id')

preds = rfc.predict(test)
preds_labels = lab_enc.inverse_transform(preds)

result = pd.DataFrame({
    'id': df_test.id,
    'Target': preds
})
result['Target'] = preds_labels
result


#Decision Tree

initial_features = list(df_test.columns)
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(df[initial_features], df.Target)

dt_plot=plt.figure(figsize=(16, 6))
plot_tree(dt, feature_names=initial_features, class_names=lab_enc.classes_, fontsize=7, impurity=False, filled=True, ax=plt.gca())





#Executive Summary Plots
#Feature Importance Comparison

features = ["Curricular units 2nd sem (approved)", "Tuition Fees Up to date", "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)", "Curricular units 2nd sem (grade)", "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (evaluations)", "Admission grade", "Previous qualification (grade)",
            "Scholarship Holder", "Age at enrollment", "Course"]

decision_tree_values = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
random_forest_values = [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
eda_values = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]

fig_2 = go.Figure()

fig_2.add_trace(go.Scatterpolar(
    r=decision_tree_values,
    theta=features,
    fill='toself',
    name='Decision Tree',
    line=dict(color='#9363b6')
))

fig_2.add_trace(go.Scatterpolar(
    r=random_forest_values,
    theta=features,
    fill='toself',
    name='Random Forest',
    line=dict(color='#f1d343')
))

fig_2.add_trace(go.Scatterpolar(
    r=eda_values,
    theta=features,
    fill='toself',
    name='EDA',
    line=dict(color='#f66465')
))


fig_2.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=True,
    title="Comparison of Feature Importance Across Models",
    height=500
)

#Funnel

all_classes_count = len(df)
enrolled_count = len(df[df['Target'] == 1])
dropout_1st_sem = len(df[(df['Target'] == 0) & (df['Curricular units 1st sem (approved)'] == 0)])
dropout_2nd_sem = len(df[(df['Target'] == 0) & (df['Curricular units 2nd sem (approved)'] == 0)])

continued_after_1st = all_classes_count - dropout_1st_sem
continued_after_2nd = continued_after_1st - dropout_2nd_sem
graduate_count = len(df[df['Target'] == 2])

funnel_data = {
    'Stage': ['Start', 'After Semester 1', 'After Semester 2', 'Graduated'],
    'Count': [all_classes_count, continued_after_1st, continued_after_2nd, graduate_count],
    'Frame': ['Start', 'After Semester 1', 'After Semester 2', 'Graduated']
}


funnel_df = pd.DataFrame(funnel_data)

Figure_f = px.funnel(funnel_df,
                     y='Count',
                     x='Stage',
                     title='Graduates flow',
                     color_discrete_sequence=['#559bbe'],  # Apply green color to all stages
                     height=425)

Figure_f.update_layout(
    plot_bgcolor='white',  # Set background of the plot area to white
    paper_bgcolor='white',  # Set background of the entire figure to white
    xaxis_title="Number of Students",
    yaxis_title="Stages",
    font=dict(size=14),
    showlegend=False  # Hide legend since all stages have the same color
)


