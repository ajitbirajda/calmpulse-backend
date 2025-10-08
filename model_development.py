import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ==============================================================================
# Model Development for Employee Dataset
# ==============================================================================

try:
    cleaned_employee_file = 'cleaned_employeedataset_advanced.csv'
    if not os.path.exists(cleaned_employee_file):
        raise FileNotFoundError(f"Cleaned file '{cleaned_employee_file}' not found. Please run the updated data_cleaning.py first.")

    df_employee = pd.read_csv(cleaned_employee_file)
    print("--- Model Development for Employee Dataset (Target: Stress_Level) ---")

    # Define features (X) and target (y)
    features = df_employee.drop('Stress_Level', axis=1)
    target = df_employee['Stress_Level']

    # One-hot encode the remaining categorical features
    features_encoded = pd.get_dummies(features, columns=['Job_Role', 'Satisfaction_with_Remote_Work'], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest model parameters found: {grid_search.best_params_}")

    # Make predictions and evaluate the best model
    y_pred_employee = best_model.predict(X_test)
    accuracy_employee = accuracy_score(y_test, y_pred_employee)
    
    print("\nEmployee Model Performance with Best Parameters:")
    print(f"Accuracy: {accuracy_employee:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_employee))
   

    # Visualize the Confusion Matrix
    cm_employee = confusion_matrix(y_test, y_pred_employee)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_employee, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix for Employee Stress Level')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    # Save the final trained model for deployment
    with open('employee_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\nEmployee model saved to 'employee_model.pkl'")


except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred with the employee dataset: {e}")

print("\n" + "="*80 + "\n")

# ==============================================================================
# Model Development for Student Dataset (unchanged)
# ==============================================================================

try:
    cleaned_student_file = 'cleaned_studentdataset.csv'
    if not os.path.exists(cleaned_student_file):
        raise FileNotFoundError(f"Cleaned file '{cleaned_student_file}' not found. Please run data_cleaning.py first.")

    df_student = pd.read_csv(cleaned_student_file)
    print("--- Model Development for Student Dataset (Target: stress_level) ---")

    # Define features (X) and target (y)
    features = df_student.drop('stress_level', axis=1)
    target = df_student['stress_level']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model_student = RandomForestClassifier(n_estimators=100, random_state=42)
    model_student.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred_student = model_student.predict(X_test)
    accuracy_student = accuracy_score(y_test, y_pred_student)
    
    print("\nStudent Model Performance:")
    print(f"Accuracy: {accuracy_student:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_student))

    # Visualize the Confusion Matrix
    cm_student = confusion_matrix(y_test, y_pred_student)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_student, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Student Stress Level')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
      # Save the final trained model for deployment
    with open('student_model.pkl', 'wb') as f:
        pickle.dump(model_student, f)
    print("\nStudent model saved to 'student_model.pkl'")
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred with the student dataset: {e}")