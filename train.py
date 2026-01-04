import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib



df = pd.read_csv(r"C:\Users\Admin\Downloads\Healthcare Symptoms–Disease Classification DatasetHealthcare Symptoms–Disease Classification Dataset\Healthcare.csv")

df['Symptoms_List'] = df['Symptoms'].str.split(',')
all_symptoms = sorted(set(sum(df['Symptoms_List'],[])))
for s in all_symptoms:
    df[s] = df['Symptoms_List'].apply(lambda x: int(s in x))

X = df[['Age'] + list(all_symptoms)] 
y = df['Disease']

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size= 0.2,random_state= 45,stratify=y)



models = {
    'knn': KNeighborsClassifier(n_neighbors=6,metric= 'minkowski'),
    'svm': SVC(kernel = 'rbf',gamma= 'scale'),
    'LR': LogisticRegression(max_iter= 1000,solver='lbfgs')
}
results = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    results[name] = acc
   
    print(name, "Accuracy:", acc)
best_model_name = max(results,key=results.get)
best_model = models[best_model_name]
print("Best_model:",best_model_name)

joblib.dump(best_model,'Disease_model.pkl')
joblib.dump(all_symptoms,'Symptom_List.pkl')

print('Model saved Successfully')








                         













