import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="2_🤖_Machine_Learning_Model.py", page_icon="🤖", layout="wide")

logit_accuracy = 0.8083
logit_cr = """precision    recall  f1-score   support
           0       0.88      0.80      0.84      2501
           1       0.64      0.49      0.55      1461
           2       0.82      0.94      0.87      3690

    accuracy                           0.81      7652
   macro avg       0.78      0.74      0.75      7652
weighted avg       0.80      0.81      0.80      7652"""
logit_cm = np.array([[2001, 150, 350], [400, 720, 341], [120, 102, 3468]])
logit_cv_scores = [0.7989, 0.7991, 0.7967, 0.8017, 0.7987]
logit_cv_avg = 0.7990

# Random Forest Results
rf_accuracy = 0.8309
rf_cr = """precision    recall  f1-score   support
           0       0.90      0.81      0.85      2501
           1       0.64      0.63      0.63      1461
           2       0.86      0.92      0.89      3690

    accuracy                           0.83      7652
   macro avg       0.80      0.79      0.79      7652
weighted avg       0.83      0.83      0.83      7652"""
rf_cm = np.array([[2025, 140, 336], [385, 921, 155], [101, 185, 3404]])
rf_cv_scores = [0.8279, 0.8244, 0.8263, 0.8274, 0.8204]
rf_cv_avg = 0.8253

st.title("Machine Learning Model")

code = '''# Prepare Features & Target
X = df.drop(['Target', 'id'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Fit the Model - Logistic Regression Classifier
logit = LogisticRegression(multi_class='ovr')
logit.fit(X_train, y_train)
y_pred_logit = logit.predict(X_test)

# Model Performance Metrics
accuracy = accuracy_score(y_test, y_pred_logit)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred_logit)
cm = confusion_matrix(y_test, y_pred_logit)

#Cross-validation
cv_scores = cross_val_score(logit, X_train, y_train, cv=5)

#Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

# Model Performance Metrics
cm_ = confusion_matrix(y_pred_rfc, y_test)
cr_ = classification_report(y_pred_rfc, y_test)
score_ = accuracy_score(y_pred_rfc, y_test)

#Cross-validation
val_score = cross_val_score(rfc, X_train, y_train, cv=5)
'''

with st.expander('Prepare Features, \'Target\', split Dataset for test:'):
    st.code(code, language="python")

col1, col2, col3 = st.columns([8, 1, 8])

# Logistic Regression Results
with col1:
    # Accuracy
    st.subheader("Logistic Regression Classifier")
    st.write(f"**Accuracy:** {logit_accuracy:.4f}")
    st.markdown("""
    - The Logistic Regression model achieved an accuracy of **80.83%** on the test set.
    - The model performed well in predicting **Graduates (0)** with **88% precision** and **80% recall**.
    - It struggled with **Enrolled students (1)**, achieving **64% precision** and **49% recall**, indicating a high number of false positives and false negatives.
    - **Dropouts (2)** were classified with **82% precision** and **94% recall**, indicating a strong performance in identifying this class.
    - The confusion matrix shows that many **Enrolled students (1)** were misclassified as **Dropouts (2)** or **Graduates (0)**.
    """)

    st.subheader("Logistic Regression - Classification Report")
    st.text(logit_cr)

    st.subheader("Logistic Regression - Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(logit_cm, annot=True, fmt='d', cmap="Set2", xticklabels=["Graduates", "Enrolled", "Dropouts"], yticklabels=["Graduates", "Enrolled", "Dropouts"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Logistic Regression - Cross-validation Scores")
    st.write(f"Cross-validation scores: {logit_cv_scores}")
    st.write(f"**Average cross-validation score**: {logit_cv_avg:.4f}")

# Random Forest Results
with col3:
    st.subheader("Random Forest Classifier")
    st.write(f"**Accuracy:** {rf_accuracy:.4f}")
    st.markdown("""
    - The Random Forest model achieved an accuracy of **83.09%** on the test set.
    - It performed well in predicting **Graduates (0)** with **90% precision** and **81% recall**.
    - **Enrolled students (1)** were predicted with lower precision (**64%**) and recall (**63%**), suggesting difficulty in distinguishing them from the other classes.
    - **Dropouts (2)** were classified with **86% precision** and **92% recall**, indicating that the model is very good at predicting this group.
    - The confusion matrix shows misclassifications, especially between **Enrolled** and the other two categories.
    """)

    st.subheader("Random Forest - Classification Report")
    st.text(rf_cr)

    st.subheader("Random Forest - Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap="Set2", xticklabels=["Graduates", "Enrolled", "Dropouts"], yticklabels=["Graduates", "Enrolled", "Dropouts"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Random Forest - Cross-validation Scores")
    st.write(f"Cross-validation scores: {rf_cv_scores}")
    st.write(f"**Average cross-validation score**: {rf_cv_avg:.4f}")