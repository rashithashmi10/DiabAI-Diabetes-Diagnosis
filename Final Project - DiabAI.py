#!/usr/bin/env python
# coding: utf-8

# Importing Dependencies

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# In[2]:


#loading the dataset to a pandas df
df = pd.read_csv('diabetes.csv')


# In[3]:


#printing the first 5 rows
df.head()


# In[4]:


#no of rows and cols
df.shape


# In[5]:


#getting the statistical measures of the df
df.describe()


# In[6]:


#no of diabetics and non-diabetics
df['Outcome'].value_counts()

#0 = Non-Diabetic

#1 = Diabetic


# Data Cleaning

# In[7]:


#Drop duplicates


# In[8]:


print('Before dropping duplicates: ', df.shape)
df = df.drop_duplicates()
print('After dropping duplicates: ', df.shape)


# In[9]:


#Check for NULL value


# In[10]:


df.isnull().sum()


# In[11]:


#Check for missing values


# In[12]:


print('No of missing values in Glucose: ', df[df['Glucose'] == 0].shape[0])
print('No of missing values in BloodPressure: ', df[df['BloodPressure'] == 0].shape[0])
print('No of missing values in SkinThickness: ', df[df['SkinThickness'] == 0].shape[0])
print('No of missing values in Insulin: ', df[df['Insulin'] == 0].shape[0])
print('No of missing values in BMI: ', df[df['BMI'] == 0].shape[0])


# In[13]:


#Replace missing values with mean


# In[14]:


df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

df.describe()


# Data Visualisation

# In[15]:


#Count plot


# In[16]:


#Count plot

f, ax = plt.subplots(1,2,figsize=(10,5))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],shadow=True)
ax[0].set_title('Outcome')
ax[0].set_ylabel('')


sns.countplot(x='Outcome', data=df, ax=ax[1]) # Pass 'Outcome' as the 'x' argument explicitly
ax[1].set_title('Outcome')
N, P = df['Outcome'].value_counts()
print('Negative(0) ->', N)
print('Positive(1) ->', P)

plt.grid()
plt.show()


# In[17]:


#Histogram (data is balanced or skewed)

df.hist(bins=10,figsize=(10,10))
plt.show()


# In[18]:


#Correlation analysis

#get correlations of each feature in the dataset
corr_mat = df.corr()
top_corr_features = corr_mat.index
plt.figure(figsize=(10,10))
#plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# Split data into x and y

# In[19]:


#separating the independent and dependent variables
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
print(X.head())
print(y.head())


# In[20]:


# Data Standardisation - Feature Scaling

scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)
print(standardised_data)

X = standardised_data
y = df.Outcome
print(X)
print(y)


# Split data into training and testing data

# In[21]:


#80% is train, 20% is test
#random state is used to ensure a specific split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

print(X.shape, X_train.shape, X_test.shape)


# In[22]:


#Classification Models


# Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver='liblinear', multi_class='ovr')
lr_model.fit(X_train, y_train)


# In[24]:


lr_preds = lr_model.predict(X_test)
print(classification_report(y_test, lr_preds))


# K Neighbours Classifier

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)


# In[26]:


knn_preds = knn_model.predict(X_test)
print(classification_report(y_test, knn_preds))


# Support Vector Machine(SVM)

# In[27]:


from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)


# In[28]:


svm_preds = svm_model.predict(X_test)
print(classification_report(y_test, svm_preds))


# Random Forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(criterion='entropy')
rf_model.fit(X_train, y_train)


# In[30]:


rf_preds = rf_model.predict(X_test)
print(classification_report(y_test, rf_preds))


# Decision tree

# In[31]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[32]:


dt_preds = dt_model.predict(X_test)
print(classification_report(y_test, dt_preds))


# Accuracies of all models

# In[33]:


lr_preds = lr_model.predict(X_test)

knn_preds = knn_model.predict(X_test)

svm_preds = svm_model.predict(X_test)

dt_preds = dt_model.predict(X_test)

rf_preds = rf_model.predict(X_test)


# In[34]:


#get the accuracy of the models
print('Accuracy score of Logistic Regression:', round(accuracy_score(y_test, lr_preds) * 100, 2))
print('Accuracy score of KNN:', round(accuracy_score(y_test, knn_preds) * 100, 2))
print('Accuracy score of SVM:', round(accuracy_score(y_test, svm_preds) * 100, 2))
print('Accuracy score of Decision Tree:', round(accuracy_score(y_test, dt_preds) * 100, 2))
print('Accuracy score of Random Forest:', round(accuracy_score(y_test, rf_preds) * 100, 2))


# In[35]:


from sklearn.metrics import classification_report

# Generate classification reports for each model
print("Logistic Regression:")
print(classification_report(y_test, lr_preds))

print("KNN:")
print(classification_report(y_test, knn_preds))

print("SVM:")
print(classification_report(y_test, svm_preds))

print("Decision Tree:")
print(classification_report(y_test, dt_preds))

print("Random Forest:")
print(classification_report(y_test, rf_preds))


# Hyperparameter Tuning

# In[43]:


from sklearn.model_selection import GridSearchCV


# In[61]:


# Hyperparameter tuning for SVM
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print("Best parameters for SVM:", grid_svm.best_params_)
svm_preds = grid_svm.predict(X_test)
print('Accuracy score of SVM (tuned):', round(accuracy_score(y_test, svm_preds) * 100, 2))


# In[68]:


# Hyperparameter tuning for Decision Tree
param_grid_dt = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)
print("Best parameters for Decision Tree:", grid_dt.best_params_)
dt_preds = grid_dt.predict(X_test)
print('Accuracy score of Decision Tree (tuned):', round(accuracy_score(y_test, dt_preds) * 100, 2))


# In[69]:


# Hyperparameter tuning for Logistic Regression
param_grid_lr = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_lr = GridSearchCV(LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", grid_lr.best_params_)
lr_preds = grid_lr.predict(X_test)
print('Accuracy score of Logistic Regression (tuned):', round(accuracy_score(y_test, lr_preds) * 100, 2))


# In[70]:


# Hyperparameter tuning for KNN
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
print("Best parameters for KNN:", grid_knn.best_params_)
knn_preds = grid_knn.predict(X_test)
print('Accuracy score of KNN (tuned):', round(accuracy_score(y_test, knn_preds) * 100, 2))


# In[71]:


# Hyperparameter tuning for Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(criterion='entropy'), param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print("Best parameters for Random Forest:", grid_rf.best_params_)
rf_preds = grid_rf.predict(X_test)
print('Accuracy score of Random Forest (tuned):', round(accuracy_score(y_test, rf_preds) * 100, 2))


# Save the Model

# In[42]:


# Save the Model with the Highest Accuracy using pickle

import pickle
pickle.dump(rf_model, open('rf_model.pkl', 'wb')) #svm has the highest accuracy
pickle.dump(scaler, open('scaler.pkl', 'wb')) #save the std scaler too

