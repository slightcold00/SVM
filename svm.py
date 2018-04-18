import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

#calculate kernel value
def kernelvalue(vector_x, vector_y, kernel0):
    kntype = kernel0[0]
    knvalue = 0

    if kntype == 'linear':
        knvalue = np.dot(vector_x , vector_y)
    if kntype == 'polynomial':
        p = kernel0[2]
        knvalue = (np.dot(vector_x , vector_y) + 1) ** p
    if kntype == 'rbf':
        sigma = kernel0[1]
        diff = vector_x - vector_y
        knvalue = np.exp(np.dot(diff, diff)/ (-2.0 * sigma **2))
    return knvalue

#calculate kernel matrix given train set and kernel type
def prearray(train_x, train_t, kernel0):
    numsamples = train_x.shape[0]
    knarray = np.array(np.zeros((numsamples,numsamples)))
    for i in range(numsamples):
        for j in range(numsamples):
            knarray[i,j] = train_t[i] * train_t[j] * kernelvalue(train_x[i,],train_x[j,],kernel0)
    return knarray

#define a struct for shorting variavles and data
class SVMStruct:
    def __init__(self, inputs, targets, C, kernel0):
        self.train_x = inputs #each row stands for a sample
        self.train_t = targets #corresponding label
        self.C = C            #slack variable
        self.N = inputs.shape[0]    #number of samples
        self.alpha = np.array(np.zeros((self.N,1)))   #Lagrange factors for all samples
        self.calb = np.array(np.zeros((self.N,1)))
        self.b = 0
        self.kernel0 = kernel0
        self.premat = prearray(self.train_x, self.train_t, self.kernel0)
        self.Supportvector = inputs
        self.Svindex = targets
        self.nonzero_alpha = np.array(np.zeros((self.N,1))) 

#The main training procedure
def trainSVM(inputs, targets, C, kernelO):
    svm = SVMStruct(inputs, targets, C, kernelO)
    #optimize the dual problem
    #define the objective function
    def objective(a):
        objarr = np.array(np.zeros((svm.N,svm.N)))
        for i in range(svm.N):
            for j in range(svm.N):
                objarr[i,j] = a[i] * a[j] * svm.premat[i,j]
        obj = (1/2) * np.sum(objarr) - np.sum(a)
        return obj

    #define zerofun
    def zerofun(a):
        return np.sum(a * svm.train_t)

    ret = minimize (objective , np.array(np.zeros((svm.N,1))),\
          bounds = [(0, C) for b in range(svm.N)], constraints={'type':'eq', 'fun':zerofun})
    svm.alpha = np.array(ret['x'])
    svm.nonzero_alpha = svm.alpha

    #extract the non-zero alpha value
    for i in range(svm.N-1,-1,-1):
        if svm.alpha[i,] < 10 ** (-5):
            svm.nonzero_alpha = np.delete(svm.nonzero_alpha, i ,0)
            svm.Supportvector = np.delete(svm.Supportvector, i ,0)
            svm.Svindex = np.delete(svm.Svindex, i ,0)
    print(svm.Supportvector)
    print(svm.Svindex)
    
    #calculate b
    for j in range(len(svm.Supportvector)):
        calb = np.array(np.zeros((len(svm.Supportvector),1)))
        for i in range(len(svm.Supportvector)):
            calb[i,] = svm.nonzero_alpha[i] * svm.Svindex[i] * \
            kernelvalue(svm.Supportvector[j,], svm.Supportvector[i,], svm.kernel0)
        svm.b = svm.b + np.sum(calb) - svm.Svindex[j,]
    svm.b = svm.b / (len(svm.Supportvector))    
        
    return svm
    

#indicator function
def indicator(svm, x, y):
    vector_x = (x, y)
    vector_x = np.array(vector_x)
    np.transpose(vector_x)
    calin = np.array(np.zeros((len(svm.Supportvector),1)))
    for i in range(len(svm.Supportvector)):
        calin[i,] = svm.nonzero_alpha[i] * svm.Svindex[i] * \
        kernelvalue(vector_x, svm.Supportvector[i,], svm.kernel0)
    ind = np.sum(calin) - svm.b
   
    return ind

#generate test data
np.random.seed(100)
classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5],np.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = np.random.randn(20,2)*0.2+[0.0,-0.5]
inputs = np.concatenate((classA,classB))
targets = np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]

#train the SVM
C = None
kernelO = ('polynomial', 1, 2)  
svm = trainSVM(inputs, targets, C, kernelO)   

#draw the picture
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.plot([p[0] for p in svm.Supportvector], [p[1] for p in svm.Supportvector], 'yo')
plt.axis('equal')

xgrid = np.linspace(-5,5)
ygrid = np.linspace(-4,4)

grid = np.array([[indicator(svm,x,y) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors = ('red', 'black', 'blue'), linewidths = (1, 1, 1))

print(np.array([indicator(svm,p[0],p[1]) for p in svm.Supportvector]))

plt.savefig('svmplot_poly_.pdf')
plt.show()
