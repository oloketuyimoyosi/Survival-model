import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset (Ensure train.csv is in your directory)
# Dataset source: Kaggle Titanic
df = pd.read_csv("C:\\Users\\oloke\\OneDrive\Desktop\\Pytorch\\Titanic-Dataset.csv")

# 2. Preprocessing
# Fill missing values: Age (median), Embarked (mode), Fare (median)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Drop columns irrelevant for this simple baseline (Name, Ticket, Cabin, PassengerId)
df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

# Encode Categorical Variables (Sex, Embarked)
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex']) # male:1, female:0

le_emb = LabelEncoder()
df['Embarked'] = le_emb.fit_transform(df['Embarked'])

# 3. Split Data
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Baseline XGBoost Model
baseline_model = xgb.XGBClassifier( eval_metric='logloss', random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)

print("--- Baseline Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")

# 5. Hyperparameter Tuning (GridSearchCV)
# Tuning n_estimators, max_depth, learning_rate
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. Evaluate Tuned Model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\n--- Tuned Model Performance ---")
print(f"Best Params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

# Save the model for Week 8
best_model.save_model("C:\\Users\\oloke\\OneDrive\Desktop\\Pytorch\\titanic_xgboost_model.json")
print("\nModel saved as 'titanic_xgboost_model.json'")