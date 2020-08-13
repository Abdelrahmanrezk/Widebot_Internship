
# coding: utf-8

# ### Imports Libraries

# In[2]:


# Main libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import csv

# Text Pre-Processing
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import re

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Keras 
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras.utils import plot_model


# ### Data Preprocessing
# 
# Because reading data as one string and some of other issues related to the data in the file, I just read each line of the file to handle the cases and replace like [;,] and for each line retrive the data its contain, then make a new file that contain a cleaned csv file.

# In[3]:



def csv_file(file_path):
    '''
    Argument:
        file_path the directory of the file we need to read
    return:
        list of all rows in the file
    '''
    df_data = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # escape header
        for line in reader:
            # before preprocess
            line = ",".join(line) 
            line = line.replace(';', ' ')
            line = line.replace(',', ' ')
            line = line.split(' ')
            # after preprocess the file
            df_data.append(line)
            
    return df_data
df_training_data = csv_file('csv_files/training.csv')
df_validation_data = csv_file('csv_files/validation.csv')


# In[4]:


# After pre-process the lines of the file make a pre-defined cols names
cols = []
for i in range(len(df_training_data[0])-4):
    cols.append("variable" + str(i+1))
cols.append('classLabel')
cols.append('target1')
cols.append('target2')
cols.append('target3')
df_training_data = pd.DataFrame(df_training_data, columns=cols)
df_validation_data = pd.DataFrame(df_validation_data, columns=cols)


# ## Note !
# 
# - drop the class label because it has no meaning when we train the model
# - while its have no meaning maybe cause to miss leading our training to the wrong way

# In[5]:



df_training_data = df_training_data.drop(['classLabel'], axis=1)
df_validation_data = df_validation_data.drop(['classLabel'], axis=1)
print(len(df_training_data))
print(len(df_validation_data))


# ### Target Class
# I noticed that the last three columns are the target variables and they like to complete each other so I will follow these steps of pre process the target variables:
# - use the second one of them as our main target variables
# - if its empty or Nan value:
#     - check with the last columns if has the value
#     - if not I will take the value of first columns
#     - if still not I will take the prvious value and applied to it

# In[6]:


training_targets = list(df_training_data[['target1', 'target2', 'target3']].values)
validation_targets = list(df_validation_data[['target1', 'target2', 'target3']].values)
training_targets[0] # each index is three values as 'target1', 'target2', 'target3'


# In[7]:


def handle_target_class(targets):
    '''
    Argument:
        targets: list of list each of them are 3 values
    reuturn:
        output_class: One list that contain all of our classes and fill the empty or nan values
    '''
    output_class = []
    for i, target in enumerate(targets):
        if target[1] == '"no."' or target[1] == 0 or target[1] == '0':
            output_class.append(0)
            target[1] = 0
        elif target[1] == '"yes."' or target[1] == 1 or target[1] == '1':
            output_class.append(1)
        elif target[2] == '"no."' or target[2] == 0 or target[2] == '0':
                output_class.append(0)
        elif target[2] == '"yes."' or target[2] == 1 or target[2] == '1':
            output_class.append(1)
        elif target[0] == '"no."' or target[0] == 0 or target[0] == '0':
                output_class.append(0)
        elif target[0] == '"yes."' or target[0] == 1 or target[0] == '1':
            output_class.append(1)
        else:
            output_class.append(output_class[i-1])
    return output_class

y_train = handle_target_class(training_targets) # The target classes
y_test = handle_target_class(validation_targets) # The target classes


# ### Nan Values
# 
# Some of our features represent continuous values and beside of that some of these values are **Nan** values so I replace the nan values  by the mean static value or by the most frequent value based on that if the unique values > 100 then replaced with the mean else replace by the most frequent value of the unique values in this feature, then I will apply the features scaling for these features to make the values of our data in some ranges bwtween [0 - 1] or [-1 - 1], which help the model to learn well than of different ranges, and this also avoid the overfitting.
# 
# Some of other features take the same process but replacd by the most frequent.
# 
# - Extract continuous features
# - replace nan values with static mean
# - features scaling

# ## Note !
# 
# some of the features have values at same time of some chars so I will replace chars with nan value.
# 
#  then when apply the Imputer these values will change to mean or most frequent.

# In[8]:


# def handle_char_cell(col):
#     '''
#     Try to check where its float number if not then give it a nan value
#     then using the Imputer_missing functoin will change nan to the mean or most frequent values.
#     '''
#     for i in range(len(col)):
#         try:
#             float(col[i])
#         except:
#             col[i] = np.nan
#     return col

# def handle_num_cell(col):
#     for i in range(len(col)):
#         try:
#             float(col[i])
#             col[i] = np.nan
#         except:
#             pass
#     return col


# In[9]:


def Imputer_missing(features, missing_value, strategy):
    '''
    Argument:
    features: The columns features you will apply the missing values on
    missing_value: which is string [nan or string of missing values you need to replace ]
    strategy: Replace nan values with mean or with most_frequent values
    
    Return:
    features: The columns with all missing are filled
        
    '''
    imp_mean = SimpleImputer(missing_values=missing_value, strategy=strategy)
    for i in range(features.shape[1]):
        imp = features[:, i]
        imp = np.array(imp).reshape(-1, 1)
        imp_mean.fit(imp)
        imp = imp_mean.transform(imp)
        features[:, i] = imp.reshape(-1)
    return features


# ## Note !
# 
# After what we have done with continuos values we need to deal with categorial features so we use label encoder from sklearn to encode the different discrete values to some index based on number of classes in the category we are in, and it takes a sequence of index from 0 to number of classes.
# 

# In[10]:


def label_encoder(categircal_features):
    '''
    Argument:
    categircal_features: the columns that contain categircal features
    '''
    le = preprocessing.LabelEncoder()
    for i in range(categircal_features.shape[1]):
        categircal_feature = categircal_features[:, i]
        le.fit(categircal_feature)
        categircal_feature = le.transform(categircal_feature)
        categircal_features[:, i] = categircal_feature.reshape(-1)
    return categircal_features


# ## Note !
# 
# After what we have done our function now its time to apply the process of pre-process using the functions we have implemented and check with training model.

# In[11]:


for col in df_training_data.columns:
    print("The number of unique values of column " + str(col) + " " + str(len(pd.unique(df_training_data[col]))))


# In[12]:


# Tr Pre-process

cols1 = ['variable5', 'variable16', 'variable17', 'variable18']

cols2 = ['variable1', 'variable2', 'variable3', 'variable4', 'variable6', 'variable7', 'variable8','variable9',
       'variable10', 'variable11', 'variable12', 'variable13', 'variable14', 'variable15']

# impute the categorical for training data
tr_categorical_features = np.array(list(df_training_data[cols2].values)).reshape(-1,14)
tr_categorical_features = label_encoder(tr_categorical_features)
tr_categorical_features = Imputer_missing(tr_categorical_features, np.nan, "most_frequent")

# impute the categorical for testing data
va_categorical_features = np.array(list(df_validation_data[cols2].values)).reshape(-1,14)
va_categorical_features = label_encoder(va_categorical_features)
va_categorical_features = Imputer_missing(va_categorical_features, np.nan, "most_frequent")

# impute the continuous for training data
tr_con_features_mean = np.array(list(df_training_data[cols1].values)).reshape(-1,4)
tr_con_features_mean = label_encoder(tr_con_features_mean)
tr_con_features_mean = Imputer_missing(tr_con_features_mean, np.nan, "mean")

# impute the continuous for testing data
va_con_features_mean = np.array(list(df_validation_data[cols1].values)).reshape(-1,4)
va_con_features_mean = label_encoder(va_con_features_mean)
va_con_features_mean = Imputer_missing(va_con_features_mean, np.nan, "mean")


tr_con_features_mean = tr_con_features_mean.astype('float')
va_con_features_mean = va_con_features_mean.astype('float')


# scaling
scale = preprocessing.MinMaxScaler().fit(tr_con_features_mean)
tr_con_features_mean = scale.transform(tr_con_features_mean)

# scaling
scale = preprocessing.MinMaxScaler().fit(va_con_features_mean)
va_con_features_mean = scale.transform(va_con_features_mean)



# In[13]:


# continuous  features
tr_con_features_mean = pd.DataFrame(tr_con_features_mean)
va_con_features_mean = pd.DataFrame(va_con_features_mean)

# categorical features
va_categorical_features = pd.DataFrame(va_categorical_features)

tr_categorical_features = pd.DataFrame(tr_categorical_features)
# Concat continuous and categorical features in one dataframe for training and testing
df_training_data = pd.concat([tr_con_features_mean, tr_categorical_features], axis= 1)
df_validation_data = pd.concat([va_con_features_mean, va_categorical_features], axis= 1)


# In[14]:


# Now Display and save the new CSV file after we have done the pre-process Approach
df_training_data.head()


# In[15]:


df_validation_data.head()


# In[16]:


print(len(df_training_data))


# ## Note !
# 
# Now we have done all of the pre process to our data, next we need to save our new form of the data in csv file along with the Traget labels we have also done above.
# 
# Beofre of save the new cleaned files I will shuffle the data to be randomies when the model start to learn from.

# In[17]:


df_training_data['target'] = y_train
df_validation_data['target'] = y_test


# In[18]:


# fea = features
cols = []
for i in range(len(df_training_data.columns)-1):
    cols.append("fea" + str(i+1))
cols.append('target')
print(cols)


# In[19]:


# Shuffle before save as csv
df_training_data = df_training_data.sample(frac=1).reset_index(drop=True) # shuffel
df_validation_data = df_validation_data.sample(frac=1).reset_index(drop=True) # shuffel

#Save as CSV
df_training_data.to_csv('csv_files/pre_process_training.csv', header=cols, index=False)
df_validation_data.to_csv('csv_files/pre_process_validation.csv', header=cols, index=False)


# In[20]:


df_training_data = pd.read_csv('csv_files/pre_process_training.csv')
df_validation_data = pd.read_csv('csv_files/pre_process_validation.csv')


# ## Display The Data after The Pre-processing 

# In[21]:


df_training_data.head()


# In[22]:


df_validation_data.head()


# In[23]:


# Training data
training_data = df_training_data.iloc[:, :-1]
y_train = df_training_data.iloc[:, -1]

# Testing data
testing_data = df_validation_data.iloc[:, :-1]
y_test = df_validation_data.iloc[:, -1]
training_data.head()


# In[24]:


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


# ## Train with Multinomial nive bayes Model

# In[25]:


clf_MultinomialNB = MultinomialNB()
model = clf_MultinomialNB.fit(training_data, y_train)
predict = model.predict(training_data)
print("F1 score of our training data is: ", f1_score(y_train, predict, average='micro'))
predict = model.predict(testing_data)
print("F1 score of our testing data is: ", f1_score(y_test, predict, average='micro'))


# ## Train with Logistic Regression Model

# In[26]:


clf_LogisticRegression = LogisticRegression(penalty='l2', solver='sag')
logistic_model = clf_LogisticRegression.fit(training_data, y_train)
predict = logistic_model.predict(training_data)
print("F1 score of our training data is: ", f1_score(y_train, predict, average='micro'))
predict = model.predict(testing_data)
print("F1 score of our testing data is: ", f1_score(y_test, predict, average='micro'))


# ## Training With Keras

# In[27]:


def tensor_model(x_train,y_trains, x_val, y_val, input_shape):
  # Sequential model

    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])
    
    history = model.fit(x_train,
                        y_trains,
                        epochs=25,
                        batch_size=1024,
                        validation_data=(x_val, y_val))
    return history


# In[28]:


history = tensor_model(training_data, y_train, testing_data,y_test, 18)


# In[29]:


history_dict = history.history
history_dict.keys()


acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[30]:


plt.clf()   # clear figure
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# ## Conclusion
# 
# As we can see that the model is overfitting on the training data while not doing well on the testing data, and the accuracy is less that 50% which not good and this return to some paramters:
# 
# - The number of training data is so small
# - The features of the data has  no meaning to know which feature can affect the training or not
# - The data contain a lot of empty values
# - The data contain different values as number and chars and string in some features
# - The data contain a lot of Nan values
# 
# 
# **So Some of the ways we can improve our model result can be:**
# 
# - Increase the size of the data
# - applying meaning full features as meta-data
# - Label the data by some expertise
# - Applying more pre-processing 
# - trying to combine features together
