import pandas as pd 
import numpy as np
import math
print('loaded dependencies')
pathTrain = 'C:\\Users\\Ben\\Desktop\\Sandboc\\Untitled Folder\\chapter 3\\Problem 3\\train.csv'
pathTest = 'C:\\Users\\Ben\\Desktop\\Sandboc\\Untitled Folder\\chapter 3\\Problem 3\\test.csv'

trainData = pd.read_csv(pathTrain)
testData = pd.read_csv(pathTest)


def clean_Data(inDF):
	trainDataTrimmed = inDF.drop(['Name','Ticket', 'PassengerId','Cabin'], axis = 1) # Most Cabins are undefined, so it's not wroth convoluting the data
	trainDataTrimmed = trainDataTrimmed.dropna(subset = ['Fare'])
	# Make the sex of each passenger a number
	trainDataTrimmed['Sex'] = trainDataTrimmed['Sex'] .mask(trainDataTrimmed['Sex'] == 'male' , 1)
	trainDataTrimmed['Sex'] = trainDataTrimmed['Sex'] .mask(trainDataTrimmed['Sex'] == 'female' , 2)
	# Make the place they left from a value as well
	trainDataTrimmed['Embarked'] = trainDataTrimmed['Embarked'].mask(trainDataTrimmed['Embarked'] == 'S' , 0)
	trainDataTrimmed['Embarked'] = trainDataTrimmed['Embarked'].mask(trainDataTrimmed['Embarked'] == 'C' , 1)
	trainDataTrimmed['Embarked'] = trainDataTrimmed['Embarked'].mask(trainDataTrimmed['Embarked'] == 'G' , 2)
	trainDataTrimmed['Embarked'] = trainDataTrimmed['Embarked'].mask(trainDataTrimmed['Embarked'] == 'Q' , 3)
	trainDataTrimmed['Embarked'] = trainDataTrimmed['Embarked'].mask(trainDataTrimmed['Embarked'].isnull() , 4) #There are only two null values, so it isn't a big deal
	# Next we convert the cabin to a number so we can make predictions based on that

	# Next we make a prediciction of the age using a classifier. We make the
	trainDataAgeDefined = trainDataTrimmed[trainDataTrimmed['Age'].notnull()]
	trainDataAgeUnDefined = trainDataTrimmed[trainDataTrimmed['Age'].isnull()]


	from sklearn.neighbors import KNeighborsClassifier
	knn_clf_age = KNeighborsClassifier(weights = 'uniform', n_neighbors = 2 )
	knn_clf_age.fit((trainDataAgeDefined.drop('Age',axis = 1)), (trainDataAgeDefined['Age']).astype(int) ) #Labels must be either ints or strings
	trainDataAgeUnDefined['Age'] = knn_clf_age.predict(trainDataAgeUnDefined.drop('Age',axis = 1))
	return pd.concat([trainDataAgeUnDefined , trainDataAgeDefined  ])




"""
# Here I was messing with some parameters to check how well the classifier predicted ages. Instead of scoring how often it predicted correctly, it's based on hoe close it was to the correct val
sizeOfTrain = math.ceil( len(trainDataAgeDefined) * 3/4 )
trainDataAgeDefined_train , trainDataAgeDefined_test = trainDataAgeDefined[:sizeOfTrain] , trainDataAgeDefined[sizeOfTrain:]

knn_clf_age.fit(trainDataAgeDefined_train.drop('Age',axis = 1), (trainDataAgeDefined_train['Age']).astype(int) ) #Labels must be either ints or strings

preds = knn_clf_age.predict(trainDataAgeDefined_test.drop('Age',axis = 1) )

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print( mean_absolute_percentage_error (trainDataAgeDefined_test['Age'] , preds) )
"""
from sklearn.neighbors import KNeighborsClassifier

DataFrameMain_Train = clean_Data(trainData)

knn_clf_survival = KNeighborsClassifier()
knn_clf_survival.fit(DataFrameMain_Train.drop('Survived',axis = 1) ,  DataFrameMain_Train['Survived'])

DataFrameMain_Test = clean_Data(testData)
score = knn_clf_survival.predict( DataFrameMain_Test )
print(score)

