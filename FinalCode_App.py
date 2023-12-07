# %% [markdown]
# Esha Navaneethakrishnan
# Final Assignment - Programming II
# 12 December 2023

# %%
#Q1.
import streamlit as st
import pandas as pd
import numpy as np
s = pd.read_csv('social_media_usage.csv')

#To display Streamlit app
st.title("Social Media Usage Data")
#Displaying dataframe
show_raw_data = st.checkbox("Show Raw Data")

if show_raw_data:
    # Display the raw data
    st.write("Raw data", s)

# Display dimensions of the dataframe
st.write("Data shape:", s.shape)

#Q2.
def clean_sm(x):
    return np.where(x==1,1,0)
s['sm_li'] = clean_sm(s['web1h'])    #Replacing existing column with the actual column sm_li
#Q3.
import os
import matplotlib.pyplot as plt
import seaborn as sns

#copy of dataframe
ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'sm_li']].copy()


#numeric values to drop for age and education level
ss = ss[~ss['age']>=98]
ss = ss[~ss['educ2'] >= 8]
# Dropping any missing values from the "ss" dataframe
ss.dropna(subset=['income', 'par', 'marital', 'gender', 'sm_li'], inplace=True)

#Streamlit App
st.title("Exploratory Data Analysis")

# Display Cleaned up Data
# Creating a Button on Streamlit to indicate data cleaning
if st.button("Clean Data"): 
    ss_cleaned = ss.dropna(subset=['income', 'par', 'marital', 'gender', 'sm_li']).copy()
    st.write("Data Cleaned Successfully!")
    # Display the cleaned data
    st.write(ss_cleaned)
else:
    st.write("Click the 'Clean Data' button to perform the cleaning operation.")

# Exploratory Analysis
# Pairplot to visualize relationships between features and the target
sns.set(style="ticks")
st.title("Pairplot for Social Media Usage")
# Set seaborn style
sns.set(style="ticks")
# Create pairplot
fig = sns.pairplot(ss, hue='sm_li', diag_kind="kde", markers=["o", "s"], palette="husl", height= 2.5)
# Display the pairplot in Streamlit
st.pyplot(fig)

# %%
#Q4. Creating a Target vector(y) and a feature set(x)
y = ss['sm_li'] #Target variable is whether the individual uses Linkedin
x = ss.drop('sm_li', axis = 1)
st.title("Target Vector and Feature Set")

# Display the Target vector
st.write("Target Vector (y):", y.head())

# Display the Feature Set
st.write("Feature Set (x):", x.head())

# %%
#Q5. Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 50)
st.title("Train-Testing Data Split")
# Display the shapes of the split datasets
st.write("Shape of x_train:", x_train.shape)
st.write("Shape of x_test:", x_test.shape)
st.write("Shape of y_train:", y_train.shape)
st.write("Shape of y_test:", y_test.shape)

# %%
#Q6.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
#Logistic Regression Model Code
lr = LogisticRegression(class_weight = 'balanced', random_state = 50)
lr.fit(x_train, y_train)
y_predicted = lr.predict(x_test)
#Model Evaluation
accuracy = accuracy_score(y_test, y_predicted) #Yields accuracy score
classificationreport = classification_report(y_test, y_predicted) #Yields a classification report


# Logistic Regression Model Results in Streamlit App
st.title("Logistic Regression Model Evaluation with Training Data")
st.write("Logistic Regression Model Results:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.text(classificationreport)

# %%
#Q7. Model evaluation with testing data
from sklearn.metrics import confusion_matrix
y_predicted= lr.predict(x_test)
#Accuracy calculation with testing data
accuracyscore = accuracy_score(y_test, y_predicted)
print(accuracyscore)
#Confusion Matrix generation 
confusionmatrix = confusion_matrix(y_test, y_predicted)

st.write("Confusion Matrix:")
sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", square=True, cbar=False)
st.pyplot()

# %%
#Q8. Creating confusion matrix  as a dataframe
confusionmatrix_df = pd.DataFrame(confusionmatrix, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])




#Generation of ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)
#Generating an ROC Curve on streamlit
st.write("ROC Curve:")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
st.pyplot(fig)


# %%
#Q9. 
from sklearn.metrics import precision_recall_curve
#Precision
precision = confusionmatrix[1,1] / (confusionmatrix [1,1] + confusionmatrix[0,1])
#Recall
recall = confusionmatrix[1,1] / (confusionmatrix [1,1] + confusionmatrix[1,0])
#F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)
# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, lr.predict_proba(x_test)[:, 1])

# Create an interactive precision-recall curve using Plotly
fig = px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall Curve")
st.plotly_chart(fig)
# %%

#Q10.
#Feature Definition
#High income, high education, non-parent, married, female, 42 years old
features_person1 = [8, 7, 0, 1, 1, 42] 
#High income, high education, non-parent, married, female, 82 years old
features_person2 = [8, 7, 0, 1, 1, 82] 

#Reshape features to a 2D array
features_person1 = np.array(features_person1).reshape(1, -1)
features_person2 = np.array(features_person2).reshape(1, -1)
st.title("Feature Visualization for Two Individuals")

# Plot the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, df_features_normalized.loc[:, 'Person 1'].values, 'b', alpha=0.1)
ax.fill(angles, df_features_normalized.loc[:, 'Person 2'].values, 'r', alpha=0.1)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, feature_names)
ax.set_title("Feature Comparison")

# Display the radar chart in Streamlit
st.pyplot(fig)
#Probability Prediction
probabilities_person1 = lr.predict_proba(features_person1)[:, 1]
probabilities_person2 = lr.predict_proba(features_person2)[:, 1]

# Streamlit app
st.title("Probability Predictions for Social Media Usage")

# Display probability predictions
st.write("Probability Prediction for Person 1:", probabilities_person1)
st.write("Probability Prediction for Person 2:", probabilities_person2)




