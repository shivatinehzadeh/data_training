from sklearn.model_selection import train_test_split
from data_load_and_cleaning import data_cleaning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


df=data_cleaning()


def training():
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
        
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
                }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1 Score": f1_score(y_test, y_pred, average="weighted")
            })
        return results

