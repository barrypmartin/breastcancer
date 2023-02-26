#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries 
import streamlit as st
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
import altair as alt
from urllib.error import URLError
# %matplotlib inline


# In[2]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

@st.cache
def get_cancer_data():
  # CANCER _BUCKET_URL = cancer
    df = pd.read_csv( "/breast_cancer.csv")
    return df.set_index("target")
try:
    df = load_breast_cancer()
    target = st.multiselect(
        "Choose Benign/Malignant", list(df.feature_names), ['mean radius' 'mean texture']
    )
    if not target:
        st.error("Please select at least one target.")
    else:
        data = df.loc[target]
        data /= 1000000.0
        st.write("### Gross Agricultural Production ($B)", data.sort_index())

        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["index"]).rename(
            columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
        )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.3)
            .encode(
                x="year:T",
                y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                color="Region:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )


# In[3]:


cancer


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


print(cancer['target_names'])


# In[7]:


print(cancer['target'])


# In[8]:


print(cancer['feature_names'])


# In[9]:


print(cancer['data'])


# In[10]:


cancer['data'].shape


# In[11]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))


# In[12]:


df_cancer.head()


# In[13]:


df_cancer.tail()


# In[14]:


x = np.array([1,2,3])
x.shape


# In[15]:


Example = np.c_[np.array([1,2,3]), np.array([4,5,6])]
Example.shape


# In[16]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# In[17]:


sns.countplot(df_cancer['target'], label = "Count") 


# In[18]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[19]:


# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# In[20]:


# Let's drop the target label coloumns
X = df_cancer.drop(['target'],axis=1)


# In[21]:


X


# In[22]:


y = df_cancer['target']
y


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[24]:


X_train.shape


# In[25]:


from sklearn.svm import SVC


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


svc_model = SVC()


# In[28]:


svc_model.fit(X_train, y_train)


# In[29]:


y_predict = svc_model.predict(X_test)


# In[30]:


y_predict


# In[31]:


cm = confusion_matrix(y_test, y_predict)


# In[32]:


sns.heatmap(cm, annot=True)


# In[33]:


print(classification_report(y_test, y_predict))


# In[34]:


min_train = X_train.min()
min_train


# In[35]:


range_train = (X_train - min_train).max()


# In[36]:


X_train_scaled = (X_train - min_train)/range_train


# In[37]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[38]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[39]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[40]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[41]:


print(classification_report(y_test,y_predict))


# In[42]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[43]:


from sklearn.model_selection import GridSearchCV


# In[44]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[45]:


grid.fit(X_train_scaled,y_train)


# In[46]:


grid.best_params_


# In[47]:


grid.best_estimator_


# In[48]:


grid_predictions = grid.predict(X_test_scaled)


# In[49]:


cm = confusion_matrix(y_test, grid_predictions)


# In[50]:


sns.heatmap(cm, annot=True)


# In[51]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




