import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Load Dataset
passengers = sns.load_dataset("titanic")
passengers['sex'] = passengers['sex'].map({'female':1, 'male':0})
passengers['age'].fillna(value=passengers['age'].mean(), inplace=True)
passengers['FirstClass'] = passengers['pclass'].apply(lambda x: 1 if x==1 else 0)
passengers['SecondClass'] = passengers['pclass'].apply(lambda x: 1 if x==2 else 0)

features = passengers[['sex', 'age', 'FirstClass', 'SecondClass']] #feature를 한 데 묶어서 사용하겠다
survival = passengers['survived']

#Feature setting
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)

#optional, 성능을 높이기 위한 방법
#ex) 0~200까지를 0~1로 바꾸어 주는 등
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#Training
model = LogisticRegression()
model.fit(train_features, train_labels)
print(model.score(train_features, train_labels))
print(model.score(test_features, test_labels))

#coef 뽑아 주기
print(model.coef_)

#Testing
Jack = np.array([0.0, 20.0, 0.0, 0.0])
sample_passengers = np.array([Jack])
sample_passengers = scaler.transform(sample_passengers)

print(model.predict(sample_passengers))
