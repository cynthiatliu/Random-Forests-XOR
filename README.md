# Random-Forests-XOR
# Can a computer learn XOR using random forests?

#Main model techniques in scikit-learn
#some-model-name.fit()
#some-model-name.predict()
#some-model-name.score()

import numpy as np
import matplotlib.pyplot as plot
import random as rd
from sklearn.ensemble import RandomForestClassifier

#Makes a plot of data with the decision boundary
def createPlot(clf, dataMat):
    # create a mesh to plot in
    h = .01
    x_min, x_max = dataMat[:, 0].min() - 1, dataMat[:, 0].max() + 1
    y_min, y_max = dataMat[:, -1].min() - 1, dataMat[:, -1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plot.contourf(xx, yy, Z, cmap=plot.cm.Paired)
    
    plot.title("Decision Boundaries")
    plot.scatter(dataMat[:,0],dataMat[:,1],color=(0,1,0))
    #plot.show()

#Generates random training coordinates, as well as h_even and k_even, within bounds
#Assumes that relevant area is a square
def genCoords(lowerLim, upperLim):
    ranCoords = open("randomCoordinates.txt", 'w')
    for i in range(200):
        j = rd.randrange(lowerLim, upperLim+1, 1)
        m = rd.randrange(lowerLim, upperLim+1, 1)
        n = 0 #the target value, xor
        if ((j%2) or (m%2)) and (not ((j%2) and (m%2))): #Exclusive or
            n = 1
        ranCoords.write("{0}\t{1}\t{2}\t{3}\t{4}".format(j,m,j%2,m%2,n))
        
    ranCoords.close()
    coordBase = open("randomCoordinates.txt", 'r')
        
    trainingCoords = []; testingCoords = []
    for i in range(100):
        line = coordBase.readline().split("\t")
        trainingCoords.append(line)
    for i in range(100):
        line = coordBase.readline().split("\t")
        testingCoords.append(line)
        
    trCoords = np.array(trainingCoords); testCoords = np.array(testingCoords)
        
    return trCoords, testCoords
    
def main():
    X_even = np.array([[0,0], [0,1], [1,0], [1,1]]) #all possible combinations of h_even, k_even
    Y_even = np.array([0, 1, 1, 0])
    
    rawTrainingData, rawTestData = genCoords(-10, 10)
    Y = rawTrainingData[:,-1] #target values
    X = rawTrainingData[:,0:4] #everything ese
    clf2d = RandomForestClassifier(n_estimators=200)
    clf2d = clf2d.fit(X,Y)
    createPlot(clf2d,X)
    
    X_test = rawTestData[:,0:4]
    Y_test=clf2d.predict(X_test)
    plot.scatter(X_test[:,0],X_test[:,1],color=(1,0,0))
    plot.show()
    
main()
