import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# EDA for cleaned_employeedataset.csv
# ==============================================================================

try:
    cleaned_employee_file = 'cleaned_employeedataset.csv'
    if not os.path.exists(cleaned_employee_file):
        raise FileNotFoundError(f"Cleaned file '{cleaned_employee_file}' not found. Please run data_cleaning.py first.")
    
    df_employee = pd.read_csv(cleaned_employee_file)
    print("--- Exploratory Data Analysis for Employee Dataset ---")

    # Univariate Analysis: Descriptive statistics and distributions
    print("\nDescriptive Statistics for Numerical Columns:")
    numerical_cols = ['Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating', 
                      'Stress_Level', 'Access_to_Mental_Health_Resources', 'Company_Support_for_Remote_Work', 
                      'Physical_Activity', 'Sleep_Quality']
    print(df_employee[numerical_cols].describe())
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_employee['Hours_Worked_Per_Week'], kde=True)
    plt.title('Distribution of Hours Worked Per Week')
    plt.subplot(1, 2, 2)
    sns.histplot(df_employee['Number_of_Virtual_Meetings'], kde=True)
    plt.title('Distribution of Number of Virtual Meetings')
    plt.tight_layout()
    plt.show()

    # Bivariate Analysis: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_employee[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Employee Metrics')
    plt.show()

    # Bivariate Analysis: Categorical vs. Numerical Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Job_Role', y='Hours_Worked_Per_Week', data=df_employee)
    plt.title('Hours Worked Per Week by Job Role')
    plt.xticks(rotation=45)
    plt.show()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred with the employee dataset: {e}")

print("\n" + "="*80 + "\n")

# ==============================================================================
# EDA for cleaned_studentdataset.csv
# ==============================================================================

try:
    cleaned_student_file = 'cleaned_studentdataset.csv'
    if not os.path.exists(cleaned_student_file):
        raise FileNotFoundError(f"Cleaned file '{cleaned_student_file}' not found. Please run data_cleaning.py first.")
    
    df_student = pd.read_csv(cleaned_student_file)
    print("--- Exploratory Data Analysis for Student Dataset ---")

    # Univariate Analysis: Descriptive statistics
    print("\nDescriptive Statistics for Student Data:")
    print(df_student.describe())

    # Bivariate Analysis: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_student.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Student Metrics')
    plt.show()

    # Bivariate Analysis: Pair Plot for key features
    important_features = ['anxiety_level', 'depression', 'academic_performance', 'sleep_quality', 'stress_level']
    sns.pairplot(df_student[important_features])
    plt.suptitle('Pairwise Relationships of Key Student Metrics', y=1.02)
    plt.show()

    # Bivariate Analysis: Box plot of key features vs. stress_level
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='stress_level', y='anxiety_level', data=df_student)
    plt.title('Anxiety Level by Stress Level')
    plt.subplot(1, 2, 2)
    sns.boxplot(x='stress_level', y='depression', data=df_student)
    plt.title('Depression by Stress Level')
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred with the student dataset: {e}")