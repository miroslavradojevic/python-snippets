#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../Pierian_Data_Logo.PNG"></a>
# <strong><center>Copyright by Pierian Data Inc.</center></strong> 
# <strong><center>Created by Jose Marcial Portilla.</center></strong>
# # Deep Learning for Image Classification Assessment
# # SOLUTION
# 
# Welcome to your assessment! Follow the instructions in bold below to complete the assessment.
# 
# If you get stuck, check out the solutions video and notebook. (Make sure to run the solutions notebook before posting a question to the QA forum please, thanks!)
# 
# ------------
# 
# ## The Challenge
# 
# **Your task is to build an image classifier with Keras and Convolutional Neural Networks for the Fashion MNIST dataset. This data set includes 10 labels of different clothing types with 28 by 28 *grayscale* images. There is a training set of 60,000 images and 10,000 test images.**
# 
#     Label	Description
#     0	    T-shirt/top
#     1	    Trouser
#     2	    Pullover
#     3	    Dress
#     4	    Coat
#     5	    Sandal
#     6	    Shirt
#     7	    Sneaker
#     8	    Bag
#     9	    Ankle boot
#     
#  

# ## The Data
# 
# **TASK 1: Run the code below to download the dataset using Keras.**

# In[1]:


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# ## Visualizing the Data
# 
# **TASK 2: Use matplotlib to view an image from the data set. It can be any image from the data set.**

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x_train[0]


# In[4]:


plt.imshow(x_train[0])


# In[5]:


y_train[0]


# ## Preprocessing the Data
# 
# **TASK 3: Normalize the X train and X test data by dividing by the max value of the image arrays.**

# In[6]:


x_train.max()


# In[7]:


x_train = x_train/255


# In[8]:


x_test = x_test/255


# **Task 4: Reshape the X arrays to include a 4 dimension of the single channel. Similar to what we did for the numbers MNIST data set.**

# In[9]:


x_train.shape


# In[10]:


x_train = x_train.reshape(60000,28,28,1)


# In[11]:


x_test = x_test.reshape(10000,28,28,1)


# **TASK 5: Convert the y_train and y_test values to be one-hot encoded for categorical analysis by Keras.**

# In[12]:


from tensorflow.keras.utils import to_categorical


# In[13]:


y_train


# In[14]:


y_cat_train = to_categorical(y_train)


# In[15]:


y_cat_test = to_categorical(y_test)


# ## Building the Model
# 
# **TASK 5: Use Keras to create a model consisting of at least the following layers (but feel free to experiment):**
# 
# * 2D Convolutional Layer, filters=32 and kernel_size=(4,4)
# * Pooling Layer where pool_size = (2,2)
# 
# * Flatten Layer
# * Dense Layer (128 Neurons, but feel free to play around with this value), RELU activation
# 
# * Final Dense Layer of 10 Neurons with a softmax activation
# 
# **Then compile the model with these parameters: loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']**

# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[17]:


model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[18]:


model.summary()


# ### Training the Model
# **TASK 6: Train/Fit the model to the x_train set. Amount of epochs is up to you.**

# In[21]:


model.fit(x_train,y_cat_train,epochs=10)


# ### Evaluating the Model
# 
# **TASK 7: Show the accuracy,precision,recall,f1-score the model achieved on the x_test data set. Keep in mind, there are quite a few ways to do this, but we recommend following the same procedure we showed in the MNIST lecture.**

# In[22]:


model.metrics_names


# In[23]:


model.evaluate(x_test,y_cat_test)


# In[24]:


from sklearn.metrics import classification_report


# In[25]:


predictions = model.predict_classes(x_test)


# In[26]:


y_cat_test.shape


# In[27]:


y_cat_test[0]


# In[28]:


predictions[0]


# In[29]:


y_test


# In[30]:


print(classification_report(y_test,predictions))


# # Great Job!
