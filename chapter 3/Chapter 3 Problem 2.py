from sklearn.datasets import fetch_mldata
import numpy as np

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
def plotAnImage(img, sz = 28):
    
    reshaped_img = img.reshape(sz, sz)
    plt.imshow(reshaped_img, cmap = matplotlib.cm.binary,interpolation="nearest")
    plt.show()   
    

def shiftAnImage(inArr,sz,direction):
	#Shifts an image by 1 pixel in a given direction, and returns the flattened array
    val = inArr.reshape(sz,sz)
    if direction == 'down':
        return np.append( [0 for i in range(sz)], val[:-1])
    elif direction == 'up':
        return np.append( val[1:], [0 for i in range(sz)])
    elif direction == 'left':
        return np.ravel( np.append( val[:, 1:], [ [0 ]for i in range(sz)], axis = 1))
    elif direction == 'right':
        return np.ravel( np.append([ [0 ]for i in range(sz)], val[:, :-1], axis = 1) )

    return -1

def shiftAllImages(arrArr , sz , direction):
    return [shiftAnImage(arrArr[i] , sz , direction) for i in range( len(arrArr))  ]


mnist = fetch_mldata('MNIST original')
image, target = mnist['data'] , mnist['target']
dirs = ['up','down' , 'left', 'right']

sizeOfTrain = 60000 #Pre-Split Dataset

image_train_main, image_test_main, target_train_main, target_test_main = image[:sizeOfTrain], image[sizeOfTrain:], target[:sizeOfTrain], target[sizeOfTrain:]


for direction in dirs:
	shiftImage = shiftAllImages (image, 28 , direction)
	image_train, image_test, target_train, target_test = np.array(shiftImage[:sizeOfTrain]), np.array(shiftImage[sizeOfTrain:]), target[:sizeOfTrain], target[sizeOfTrain:]

	
	#Image 
	image_train_main = np.append(image_train_main , image_train,axis = 0)
	image_test_main = np.append(image_test_main , image_test,axis = 0)
	# Target
	target_train_main = np.append(target_train_main,target_train,axis = 0)
	target_test_main = np.append(target_test_main , target_test,axis = 0)


# This dataset comes pre-split, so there's no use to use a StratifiedShuffleSplit

#Shuffles the Data
shuffle_index = np.random.permutation(sizeOfTrain*5)
image_train_main, target_train_main = image_train_main[shuffle_index], target_train_main[shuffle_index]

##KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

for i in range(10):
	#It took my computer ~15 hours to run through one iteration of this
	knn_clf = KNeighborsClassifier(weights = 'distance', n_neighbors = i+1 )
	knn_clf.fit(image_train_main, target_train_main)

	score = knn_clf.score(image_test_main,target_test_main)
	print(score)
