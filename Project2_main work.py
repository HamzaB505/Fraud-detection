import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# importing the data and showing the head and tail of the dataframe
df=pd.read_csv('creditcard.csv',sep=',')
df.head(10)



#information
df.info()
df.count()
df.isnull().values.any() 
df.Class.value_counts()



Class_col=df[["Class"]]
df_filtred=df.drop(df[["Class"]], axis="columns")                      
Class_col

#descriptive statistics
df_filtred.describe()

#plotting the data to explore it

sns.boxplot(x = df.Amount, y = df.Class)
sns.distplot(df.Time)
sns.distplot(df.Amount)
df.Class.value_counts()

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter( df[['Time']], df[['Amount']], color='r')
ax.set_xlabel('Time')
ax.set_ylabel('Amount payed')
ax.set_title('scatter plot')
plt.legend('Amount', 'Class')
plt.show()


#percentage
fraud_percentage = (fraud_cases/len(df))*100
nofraud_percentage = (nofraud_cases/len(df))*100

print('the percentage of fraud cases in the data is =', fraud_percentage)
print('the percentage of non fraud transactions in the data is = ', nofraud_percentage)


plt.figure(figsize=(7.5, 9))        
df.boxplot( column = 'Amount', by = 'Class')
plt.xlabel('Class'); plt.ylabel('Amount')
plt.legend()
plt.show()

# plotting all the variables
columns = df.columns
fig = plt.figure(figsize = (15, 12))
i = 0
for column in columns :
    i=i+1
    plt.subplot(7,7,i) #7 rows and 7 columns for the graphe matrix
    plt.plot(df_filtred[[column]])


#matrice de correlation
mat_corr = df.corr()
mat_corr
plt.figure(figsize=(9.5,10))
heat_map = sns.heatmap(mat_corr[['Class']], square = True, cmap = plt.cm.Greens, linecolor = 'white', annot = True )
plt.title('Correlation matrix heatmap')
plt.show()

#number of cases
fraud_cases = df.Class.value_counts()[1]
nofraud_cases = df.Class.value_counts()[0]
print('the number of fraud cases in the data is =', fraud_cases)
print('the number of non fraud transactions in the data is = ', nofraud_cases)
#seperating the data
df_fraud = df.loc[df['Class'] != 0]
df_nofraud = df.loc[df['Class'] != 1]


#sampling new data to balance the ratio of fraud and no fraud data
New_nofraud = df_nofraud.sample(fraud_cases)
New_nofraud

#merging data to create new dataframe
New_df = pd.concat([df_fraud, New_nofraud])
New_df = New_df.sample(frac=1).reset_index(drop=True) #shuffling
Class_vect = New_df[['Class']]
# boxplot of amount by class
plt.figure(figsize=(7.5, 9))        
New_df.boxplot( column = 'Amount', by = 'Class')
plt.xlabel('Class'); plt.ylabel('Amount')
plt.legend()
plt.show()

mat_corr = New_df.corr()
Class_corr = mat_corr[['Class']] #correlation between the y = Class and other variables
## New viz
heat_map = sns.heatmap(Class_corr, square = True, cmap = plt.cm.Greens, linecolor = 'white', annot = True )
plt.title('Correlation matrix heatmap')
plt.show()
#highest correlated variables
corr1 = Class_corr[Class_corr.Class< -0.5]
corr2 = Class_corr[Class_corr.Class > 0.5]
corr1
corr2
# Splitting data to train and to test
New_df = New_df.drop('Class', axis =1)
X = New_df
Y = Class_vect
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #train 79,9% Test 20% 
len(X_train)
len(Y_train)
len(X_test)
len(Y_test)

#Machine learning models used :

#Logistic regression 
model_LR = LogisticRegression()
model_LR.fit(X_train, Y_train.values.ravel())

pred_LR= model_LR.predict(X_test)

#visualising the predicted data vs the real data
sns.heatmap(confusion_matrix(Y_test, pred_LR), annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("confusion matrix : results evaluation")
plt.tight_layout()
plt.ylabel('Real data')
plt.xlabel('model predicted result')
plt.show()


# K-Nearest Neighbor
model_KNN = KNeighborsClassifier()

model_KNN.fit(X_train , Y_train.values.ravel())
pred_KNN = model_KNN.predict(X_test)
#Result visualisation KNN model : 
sns.heatmap(confusion_matrix(Y_test, pred_KNN), annot=True, cbar=None, cmap="Greens", fmt = 'g')
plt.title("confusion matrix : results evaluation")
plt.tight_layout()
plt.ylabel('Real data')
plt.xlabel('model predicted result')
plt.show()

#Random forest Classifier : 
model_RF = RandomForestClassifier()

model_RF.fit(X_train, Y_train.values.ravel())
pred_RF = model_RF.predict(X_test)

#Result visualisation Random forest model : 
sns.heatmap(confusion_matrix(Y_test, pred_RF), annot=True, cbar=None, cmap="Greys", fmt = 'g')
plt.title("confusion matrix : results evaluation")
plt.tight_layout()
plt.ylabel('Real data')
plt.xlabel('model predicted result')
plt.show()

#Cross validation scores #generates 5 cross validation scores by default
cross_val_LR= cross_val_score(model_LR, X_train, Y_train)
cross_val_KNN = cross_val_score(model_KNN, X_train, Y_train)
cross_val_RF= cross_val_score(model_RF, X_train, Y_train)

Result = [cross_val_LR.mean(), cross_val_KNN.mean(), cross_val_RF.mean()]


metrics.accuracy_score(Y_test, pred_LR) #Logistic Regression accuracy score
metrics.accuracy_score(Y_test, pred_KNN) #K-Nearest-Neighbor accuracy score
metrics.accuracy_score(Y_test, pred_RF) #Random Forest accuracy score

#Best model is Random Forest, based on result visualisation, Cross validation and Accuracy Score
