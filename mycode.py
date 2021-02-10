# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split



full_data=pd.read_csv("titanic_data.csv")

full_output=full_data['Survived']


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"




#Question 1 
#Using the RMS Titanic data, how accurate would a prediction be that none of the passengers survived?
##########################
predictions = pd.Series(np.zeros(len(full_output), dtype = int))
print("Death Prediction")
print(accuracy_score(full_output, predictions))

##########################


#Question 2
#How accurate would a prediction be that all female passengers survived and the remaining passengers did not survive?
#######################
survived=(full_data[full_data.Survived>0])

survived_female=full_data[(full_data.Survived>0) &(full_data.Sex=="female")]

print(len(survived))
print(len(survived_female))


print("woman survival prediction")    
print((float(len(survived_female))/len(survived))*100)

##########################


#Question 3
#How accurate would a prediction be that all female passengers and all male passengers younger than 10 survived?
#Hint: Run the code cell below to see the accuracy of this prediction.
############################
print("children under age of 10 chance of survival")
print((float(len(full_data[(full_data.Age<10) & (full_data.Survived>0) ]))/len(full_data[full_data.Age<10]))*100)

############################

#Question 4
#Describe the steps you took to implement the final prediction model so that it got an accuracy of at least 80%. What features did you look at? Were certain features more informative than others? Which conditions did you use to split the survival outcomes in the data? How accurate are your predictions?
for i in range (0,len(full_data)):
    if(full_data.Sex[i]=="male"):
        full_data.Sex[i]=0
    else:
        full_data.Sex[i]=1
# Age and Gender are the most effective features 
x = np.array(full_data[['Sex', 'Age']])
y = np.array(full_data['Survived'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

classifier = SVC(kernel = 'rbf', gamma = 200)
classifier.fit(X_train,y_train)
classifier.predict(X_test,y_test)
