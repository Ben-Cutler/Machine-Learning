from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
image, target = mnist['data'] , mnist['target']

#We'll want some data visualization tools, such as the option to look at a given number

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
def plotAnImage(img, sz = 28):
    
    reshaped_img = img.reshape(sz, sz)

    plt.imshow(reshaped_img, cmap = matplotlib.cm.binary,interpolation="nearest")
    plt.show()   
    
#N = 20000
#plotAnImage(image[N])
#print(target[N])

# This dataset comes pre-split, so there's no use to use a StratifiedShuffleSplit
sizeOfTrain = 60000

image_train, image_test, target_train, target_test = image[:sizeOfTrain], image[sizeOfTrain:], target[:sizeOfTrain], target[sizeOfTrain:]
import numpy as np
#Shuffles the Data
shuffle_index = np.random.permutation(sizeOfTrain)
image_train, target_train = image_train[shuffle_index], target_train[shuffle_index]

##KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
"""
for i in range(10):
	# Comparison of scores depending n_neighbors parameter
	# 96.91 , .9627 , .9705 , .99682 , .9688 , .9677 , .9694 , .967 , .9659 , .9665
	knn_clf = KNeighborsClassifier(weights = 'uniform', n_neighbors = i+1 )
	knn_clf.fit(image_train, target_train)

	score = knn_clf.score(image_test,target_test)
	print(score)
"""
for i in range(10):
	#Scores by 'i'
	# .9691 , .9691 , .9717 , .9714 , .9691 , .9709 , .97 , .9706 , .9673 , .9684
	knn_clf = KNeighborsClassifier(weights = 'distance', n_neighbors = i+1 )
	knn_clf.fit(image_train, target_train)

	score = knn_clf.score(image_test,target_test)
	print(score)