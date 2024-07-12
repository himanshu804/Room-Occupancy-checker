import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib




data = pd.read_csv('trainingdataset.csv')
X = data[['Temperature','CO2','Light']]

y = data['Occupancy']


x_train, x_test , y_train,y_test = train_test_split(X, y , test_size= 0.3 , random_state=42 )

rfc = RandomForestClassifier()
rfc.fit(x_train , y_train)       # fitting the model on our train dataset 

joblib.dump(rfc,'trained_model.pkl')
