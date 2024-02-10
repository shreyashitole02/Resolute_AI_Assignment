#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

#data_url = 'https://docs.google.com/spreadsheets/d/1hyPUqH6mHm_La8bx9M7DTlWStBfUqmJmfAOlKM7foRw/edit#gid=1330760030'
data = pd.read_csv('train - train.csv')


# In[2]:


data.head()


# In[3]:


data.info()


# In[4]:


label_encoder = LabelEncoder() #encode target column values into numerical labels
data['target_label'] = label_encoder.fit_transform(data['target'])


# In[5]:


# Selecting only the feature columns (T1 to T18)
features = data.drop(['target', 'target_label'], axis=1)


# In[6]:


from umap import UMAP

umap = UMAP(n_components=2, random_state=42)
components = umap.fit_transform(features)


# In[7]:


# Apply PCA to reduce dimensionality to 2 components
#pca = PCA(n_components=2)
#components = pca.fit_transform(features)


# In[8]:


# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(features)


# In[9]:


# Assign clusters to each data point
data['cluster'] = kmeans.labels_


# In[10]:


# Function to identify the cluster for a given data point
def identify_cluster(data_point):
    # Predict the cluster for the given data point
    cluster_index = kmeans.predict([data_point])[0]
    return cluster_index


# In[11]:


# Streamlit app
st.title("Resolute_AI_Assignment")


# In[12]:


# Task 1: Clustering
st.subheader("Task 1: Clustering")


# In[13]:


# User input for all 18 values
st.subheader("Enter values for T1 to T18:")
values = []
for i in range(18):
    values.append(st.number_input(f'T{i+1}', step=1))

# Convert user input to a data point (list)
data_point = values


# In[14]:


# Identify the cluster for the data point
cluster_index = identify_cluster(data_point)
st.write(f'The data point belongs to Cluster {cluster_index}')

# Plot clusters
plt.figure(figsize=(8, 5))


# In[15]:


# Scatter plot each cluster
for cluster_number in range(5):
    cluster_indices = data[data['cluster'] == cluster_number].index
    plt.scatter(components[cluster_indices, 0], components[cluster_indices, 1], label=f'Cluster {cluster_number}')


# In[16]:


# Plot user input
plt.scatter(components[-1, 0], components[-1, 1], color='blue', label='User Input', marker='+', s=100)
plt.annotate('User Input', xy=(components[-1, 0], components[-1, 1]), xytext=(-20, 20),
             textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.title('Clustering')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.tight_layout()


# In[17]:


# Show plot
st.pyplot(plt)


# In[18]:


# Load predictions cloumn
predictions_data = data["target"]


# In[19]:


# Task 2: Classification
st.subheader("Task 2: Classification")


# In[20]:


# Display predictions cloumn
st.write("Predictions of target values for the test set:")
st.write(predictions_data)


# In[21]:


# Load the output.csv file
output_data = pd.read_csv('assignment_output.csv')


# In[22]:


# Add a subheader for Task 3
st.subheader("Task 3: Python")

# Display the DataFrame
st.write("Results from rawdata are:")
st.write(output_data)


# In[ ]:




