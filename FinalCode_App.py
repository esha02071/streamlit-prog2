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
#show_raw_data = st.toggle("Show Raw Data")

if show_raw_data:
    # Display the raw data
   # st.write("Raw data", s)

# Display dimensions of the dataframe
#st.write("Data shape:", s.shape)

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
ss = ss[~((ss['age'] > 98) | (ss['educ2'] >= 8))]
# Dropping any missing values from the "ss" dataframe
ss.dropna(subset=['income', 'par', 'marital', 'gender', 'sm_li'], inplace=True)

#Streamlit App
#st.title("Exploratory Data Analysis")

# Display Cleaned up Data
# Creating a Button on Streamlit to indicate data cleaning
if st.button("Clean Data"): 
    ss.dropna(subset=['income', 'par', 'marital', 'gender', 'sm_li']).copy()
   # st.write("Data Cleaned Successfully!")
    # Display the cleaned data
   # st.dataframe(ss)
else:
   # st.write("Click the 'Clean Data' button to perform the cleaning operation.")

# Exploratory Analysis
# Pairplot to visualize relationships between features and the target
#sns.set(style="ticks")
#st.title("Pairplot for Social Media Usage")
# Set seaborn style
#sns.set(style="ticks")
# Create pairplot
#fig = sns.pairplot(ss, hue='sm_li', diag_kind="kde", markers=["o", "s"], palette="husl", height= 2.5)


# %%
# Q4. Creating a Target vector(y) and a feature set(x)
y = ss['sm_li']  # Target variable is whether the individual uses Linkedin
x = ss.drop('sm_li', axis=1)
#st.title("Target Vector and Feature Set")

# Display the Target vector
#st.write("Target Vector (y) Shape:", y.shape)
#st.write("Target Vector (y) Head:", y.head())

# Display the Feature Set
#st.write("Feature Set (x) Shape:", x.shape)
#st.write("Feature Set (x) Head:", x.head())

# %%
#Q5. Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 50)
st.title("Train-Testing Data Split")
# Display the shapes of the split datasets
#st.write("Shape of x_train:", x_train.shape)
#st.write("Shape of x_test:", x_test.shape)
#st.write("Shape of y_train:", y_train.shape)
#st.write("Shape of y_test:", y_test.shape)

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
#st.title("Logistic Regression Model Evaluation with Training Data")
#st.write("Logistic Regression Model Results:")
#st.write(f"Accuracy: {accuracy:.2f}")
#st.write("Classification Report:")
#st.text(classificationreport)

# %%
#Q7. Model evaluation with testing data
from sklearn.metrics import confusion_matrix
y_predicted= lr.predict(x_test)
#Accuracy calculation with testing data
accuracyscore = accuracy_score(y_test, y_predicted)
print(accuracyscore)
#Confusion Matrix generation 
confusionmatrix = confusion_matrix(y_test, y_predicted)

# %%
#Q8. Creating confusion matrix  as a dataframe
confusionmatrix_df = pd.DataFrame(confusionmatrix, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])


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


# Display the radar chart in Streamlit
#Probability Prediction
probabilities_person1 = lr.predict_proba(features_person1)[:, 1]
probabilities_person2 = lr.predict_proba(features_person2)[:, 1]

# Streamlit app
#st.title("Probability Predictions for Social Media Usage")

# Display probability predictions
#st.write("Probability Prediction for Person 1:", probabilities_person1)
#st.write("Probability Prediction for Person 2:", probabilities_person2)

income_category_options = {
    "Less than $10,000",
    "10 to under $20,000",
    "20 to under $30,000",
    "30 to under $40,000",
    "40 to under $50,000",
    "50 to under $75,000",
    "75 to under $100,000",
    "100 to under $150,000",
    "$150,000 or more"
}

income_mapping = {
    "Less than $10,000": 1,
    "10 to under $20,000": 2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000": 6,
    "75 to under $100,000": 7,
    "100 to under $150,000": 8,
    "$150,000 or more": 9
}
    
education_labels = {
    1: "Elementary School",
    2: "Middle School",
    3: "High School",
    4: "Associate's Degree",
    5: "Bachelor's Degree",
    6: "Master's Degree",
    7: "Doctorate"
}

# Streamlit App
st.title("Feature Definition")

# Allow users to input their own features
selected_income_category = st.selectbox("Select Income Category", income_category_options)
education = st.selectbox("Education Level:", options=list(range(1, 8)), format_func=lambda x: education_labels[x])
parent_status = st.checkbox("Is a Parent?")
marital_status = st.checkbox("Is Married?")
gender = st.radio("Gender:", options=["Male", "Female"])
age = st.number_input("Age:", min_value=1, max_value=None, step=1)


# Display the defined features
st.title("User Profile:")
st.write(f"Income Category: {selected_income_category}")
st.write(f"Education Level: {education_labels[education]}")
st.write(f"Is a Parent: {parent_status}")
st.write(f"Is Married: {marital_status}")
st.write(f"Gender: {gender}")
st.write(f"Age: {age}")

if age > 98:
    st.error("Error: Age cannot be over 98.")
    st.title("Age Distribution")
# Convert selected_income_category to numerical using income_mapping
income = income_mapping[selected_income_category]

# Convert gender to numerical (1 for Female, 0 for Male)
gender_numeric = 1 if gender == "Female" else 0

# Create a numpy array with the user's features
user_features = np.array([income, education, int(parent_status), int(marital_status), gender_numeric, age]).reshape(1, -1)

# Probability Prediction
probabilities_user = lr.predict_proba(user_features)[:, 1]

# Streamlit app
st.title("Probability Prediction for User")

# Display probability prediction for the user
st.write("Probability Prediction for User:", probabilities_user)


# Set a probability threshold (you can adjust this value)
threshold = 0.5

# Determine the user's classification based on the threshold
user_classification = "LinkedIn User" if probabilities_user > threshold else "Non-LinkedIn User"

# Display the user's classification
st.write("User Is A:", user_classification)
