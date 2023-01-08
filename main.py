import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image
import pydotplus


#----------------------------------------------------------------------------------------------------------
#Column Name
col_names = ['PassengerId', 'Pclass' ,'Sex','Age','SibSp','Parch','Survived','label']

df = pd.read_csv("data.csv", names=col_names).iloc[1:]

print(df.head())

#----------------------------------------------------------------------------------------------------------
features = ['PassengerId', 'Pclass' ,'Sex','Age','SibSp','Parch','Survived']
X = df[features]
y = df.label

#----------------------------------------------------------------------------------------------------------
#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Initialising the Decision Tree Model
clf = DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(X_train,y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))

#----------------------------------------------------------------------------------------------------------

dot_data = StringIO()
export_graphviz( clf , out_file=dot_data , filled = True, rounded = True, special_characters=True, feature_names=features , class_names = ["0","0"])

print(dot_data.getvalue())

#----------------------------------------------------------------------------------------------------------
Plot = pydotplus.graph_from_dot_data( dot_data.getvalue() )

Plot.write_png("diabaties.png")

Image(Plot.create_png())