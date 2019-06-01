#importing the libraries
import pandas as pd

#import the dataset
dataset  = pd.read_csv('heart.csv') #Here "heart.csv" is the dataset which you need to import. You can find it on kaggle
X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,13].values


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.15, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#import libraries for network
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

#Initialising first layer and hidden layer
classifier.add(Dense(units = 6, activation='relu',kernel_initializer='uniform',input_dim = 12))

#adding another hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer= 'uniform'))

#adding the output layer
classifier.add(Dense(units = 1, activation='sigmoid',kernel_initializer='uniform'))

#compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics= ['accuracy'])

#Fitting the classifier
classifier.fit(X,y,batch_size=10,epochs = 200)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
