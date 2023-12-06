# %% [markdown]
# Esha Navaneethakrishnan
# Final Assignment - Programming II
# 12 December 2023

# %%
#Q1.
import pandas as pd
import numpy as np
s = pd.read_csv('social_media_usage.csv')
print(s)
print(s.shape)

# %%
#Q2.
def clean_sm(x):
    return np.where(x==1,1,0)

#Creation of toy dataframe

toydataframe_data = {'Field 1': [0,1,0], 'Field 2': [1,0,1]}
toydf = pd.DataFrame(toydataframe_data)
toydf = toydf.apply(clean_sm)

print(toydf)

s['sm_li'] = clean_sm(s['web1h'])  # Replace 'your_existing_column' with the actual column



#dropping 8 and 9 numbers
    #clean and drop NAs, set up the target, sm_

# %%
#Q3.
import matplotlib.pyplot as plt
import seaborn as sns

#copy of dataframe
ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'sm_li']].copy()


drop1 = [98] #numeric values to drop for age
ss = ss[~ss['age'].isin(drop1)]
ss = ss[~ss['educ2'] <= 8]
# Dropping any missing values from the "ss" dataframe
ss.dropna(subset=['income', 'par', 'marital', 'gender', 'sm_li'], inplace=True)

#def clean_sm(x):
   # return np.where(x==1,1,0)

#ss['web1h'] = clean_sm(s['web1h'])


# Exploratory Analysis
# Pairplot to visualize relationships between features and the target
sns.pairplot(ss, hue='sm_li')
plt.show()

# %%
#Q4. Creating a Target vector(y) and a feature set(x)
y = ss['sm_li'] #Target variable is whether the individual uses Linkedin
x = ss.drop('sm_li', axis = 1)
y.head()
x.head()

# %%
#Q5. Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 50)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
#Q6.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
lr = LogisticRegression(class_weight = 'balanced', random_state = 50)
lr.fit(x_train, y_train)
y_predicted = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_predicted)
classificationreport = classification_report(y_test, y_predicted)
print(accuracy)
print(classificationreport)

# %%
#Q7. Model evaluation with testing data
from sklearn.metrics import confusion_matrix
y_predicted= lr.predict(x_test)
#Accuracy calculation with testing data
accuracyscore = accuracy_score(y_test, y_predicted)
print(accuracyscore)
#Confusion Matrix generation 
confusionmatrix = confusion_matrix(y_test, y_predicted)
print(confusionmatrix)

# %%
#Q8. Creating confusion matrix  as a dataframe
confusionmatrix_df = pd.DataFrame(confusionmatrix, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])
print(confusionmatrix_df)

# %%
#Q9. 
#Precision
precision = confusionmatrix[1,1] / (confusionmatrix [1,1] + confusionmatrix[0,1])
#Recall
recall = confusionmatrix[1,1] / (confusionmatrix [1,1] + confusionmatrix[1,0])
#F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)
#Calculated Metrics
print(precision)
print(recall)
print(f1_score)

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

#Probability Prediction
probabilities_person1 = lr.predict_proba(features_person1)[:, 1]
probabilities_person2 = lr.predict_proba(features_person2)[:, 1]

#Probability display
print(probabilities_person1[0])
print(probabilities_person2[0])





