import flask
import pickle
import pandas as pd
import csv
import os
import numpy as np
from flask import request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime
# Removed: import pymysql (Not needed for PostgreSQL)


# Initialize Flask app and extensions
app = flask.Flask(__name__)
CORS(app)


# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "connect_args": {"sslmode": "require"}
}

db = SQLAlchemy(app)
# ----------------------------------------------------------------------------------
# 🔥 CRITICAL: CREATE TABLES (MUST BE DONE ON FIRST DEPLOY ONLY) 🔥
# This code creates the tables in your Render PostgreSQL database if they don't exist.
# REMOVE THIS BLOCK AFTER THE FIRST SUCCESSFUL DEPLOYMENT!
# ----------------------------------------------------------------------------------
with app.app_context():
    db.create_all()

# --- Feature Order Definitions (CRITICAL FOR PREDICTION) ---
# ... (rest of your code is the same for brevity) ...

# --- Database Models (Must match current DB schema) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    contact = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), nullable=True)
    is_new_user = db.Column(db.Boolean, default=True)
    history = db.relationship('StressHistory', backref='user', lazy=True)

class StressHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stress_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now())
    # Employee Inputs
    job_role = db.Column(db.String(100), nullable=True)
    working_hours = db.Column(db.Integer, nullable=True)
    virtual_meetings = db.Column(db.Integer, nullable=True)
    work_life_balance = db.Column(db.Integer, nullable=True)
    access_to_mental_health = db.Column(db.String(10), nullable=True)
    satisfaction_with_remote_work = db.Column(db.String(20), nullable=True)
    company_support = db.Column(db.Integer, nullable=True)
    physical_activity = db.Column(db.String(20), nullable=True)
    sleep_quality = db.Column(db.String(20), nullable=True)
    # Student Inputs
    anxiety_level = db.Column(db.Integer, nullable=True)
    depression = db.Column(db.Integer, nullable=True)
    academic_performance = db.Column(db.Integer, nullable=True)
    study_load = db.Column(db.Integer, nullable=True)
    teacher_student_relationship = db.Column(db.Integer, nullable=True)
    future_career_concerns = db.Column(db.Integer, nullable=True)
    social_support = db.Column(db.Integer, nullable=True)
    peer_pressure = db.Column(db.Integer, nullable=True)
    extracurricular_load = db.Column(db.Integer, nullable=True)

# Load the machine learning models
try:
    with open('employee_model.pkl', 'rb') as f:
        employee_model = pickle.load(f)
    with open('student_model.pkl', 'rb') as f:
        student_model = pickle.load(f)
except FileNotFoundError:
    employee_model, student_model = None, None
    print("Error: Model files not found. Check the file paths.")

# --- Corrected Preprocessing Functions ---
def preprocess_employee_data_for_prediction(df):
    # Synchronize mappings with data_prep.py (case-sensitive)
    access_to_mental_health_mapping = {'Yes': 1, 'No': 0}
    physical_activity_mapping = {'None': 0, 'Weekly': 1, 'Daily': 2}
    sleep_quality_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    satisfaction_mapping = {'Unsatisfied': 0, 'Satisfied': 1}

    # Extract Job Role and drop it from the main DF
    job_role_input = df['job_role'].iloc[0]

    if pd.isna(job_role_input) or job_role_input is None or str(job_role_input).strip() == '':
        job_role_input = 'Unknown'
    
    df = df.drop(columns=['job_role'], errors='ignore')

    # Apply Categorical Mappings (Conversions)
    df['access_to_mental_health'] = df['access_to_mental_health'].map(access_to_mental_health_mapping).fillna(0).astype(int)
    df['physical_activity'] = df['physical_activity'].map(physical_activity_mapping).fillna(0).astype(int)
    df['sleep_quality'] = df['sleep_quality'].map(sleep_quality_mapping).fillna(0).astype(int)
    df['satisfaction_with_remote_work'] = df['satisfaction_with_remote_work'].map(satisfaction_mapping).fillna(0).astype(int)
    
    # --- Outlier Handling ---
    for col in ['working_hours', 'virtual_meetings']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # Feature Engineering (meetings_per_hour)
    df['meetings_per_hour'] = df['virtual_meetings'] / df['working_hours']
    df['meetings_per_hour'] = df['meetings_per_hour'].replace([np.inf, -np.inf, np.nan], 0)
    
    # --- CRITICAL FIX: One-Hot Encoding (OHE) Simulation ---
    # 1. Create a temporary DataFrame for OHE with all possible columns set to 0
    ohe_df = pd.DataFrame(0, index=df.index, columns=EMPLOYEE_OHE_COLS)
    
    # 2. Identify the correct OHE column name based on user input 
    # The training script used `pd.get_dummies` which creates `job_role_Job Role Name`.
    ohe_col_name = f'job_role_{str(job_role_input).replace(" ", "_")}'
    
    # 3. Set the specific OHE column for this prediction to 1, ONLY if it's a known column
    if ohe_col_name in ohe_df.columns:
        ohe_df[ohe_col_name] = 1
    
    # 4. Merge the main features and the OHE features
    df = pd.concat([df, ohe_df], axis=1)

    # 5. Enforce final feature order
    df = df[EMPLOYEE_FEATURES_ORDER]
    
    return df

def preprocess_student_data_for_prediction(df):
    # --- CRITICAL FIX: Implement Categorical Mappings from data_prep.py ---
    sleep_quality_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    
    # Apply mapping for the 'sleep_quality' feature, which is required as a number by the model
    df['sleep_quality'] = df['sleep_quality'].map(sleep_quality_mapping).fillna(0).astype(int)
    
    # --- FIX: Rename feature to match the trained model's expectation ---
    if 'extracurricular_load' in df.columns:
        df.rename(columns={'extracurricular_load': 'extracurricular_activities'}, inplace=True)
    
    # Drop metadata columns after mapping is done
    df = df.drop(columns=['role', 'timestamp', 'user_id'], errors='ignore') 
    
    # --- FINAL FIX: Enforce column order to match the trained model's fit order ---
    df = df[STUDENT_FEATURES_ORDER]
    
    return df

# --- CORRECTLY PLACED generate_suggestions function ---
def generate_suggestions(stress_score, input_data, user_role):
    suggestions = []

    if user_role == 'employee':
        if stress_score == 3:
            suggestions.append("⚠️ High Stress Alert: It's crucial to address your stress level immediately. Consider talking to your manager about adjusting your workload or taking a mental health day. Prioritize a a relaxing evening routine to help you wind down.")
        elif stress_score == 2:
            suggestions.append("📈 Moderate Stress: Pay close attention to your work-life balance. Try techniques like setting clear boundaries between work and personal time. A short break or a walk during the workday can make a big difference.")
        elif stress_score == 1:
            suggestions.append("✅ Low Stress: You're managing your stress effectively! Keep up the good work by maintaining a healthy lifestyle, including regular exercise and proper nutrition.")

        # Additional suggestions based on specific factors
        if input_data.get('working_hours') and input_data['working_hours'] > 45:
            suggestions.append("⏳ Long Hours: Your working hours are a major concern. Ensure you're not working through your lunch and dinner breaks. Try to disconnect from work completely after hours.")
        if input_data.get('workload') and input_data['workload'] == 'Heavy':
            suggestions.append("💼 Heavy Workload: Try to break down large tasks into smaller, more manageable steps. Using project management tools can help you track your progress and feel more in control.")
        if input_data.get('sleep_quality') and input_data['sleep_quality'] == 'Poor':
            suggestions.append("😴 Poor Sleep: Your sleep quality is a key factor. Aim for 7-9 hours of sleep per night. Avoid caffeine and screen time before bed to improve your rest.")
    
    elif user_role == 'student':
        if stress_score == 2:
            suggestions.append("🚨 High Stress: It's time to take action. Talk to your academic advisor or a university counselor to get support. Consider temporarily reducing extracurricular activities to focus on your well-being.")
        elif stress_score == 1:
            suggestions.append("📉 Moderate Stress: Your stress is at a manageable level. Continue to stay organized, but remember to schedule time for hobbies and social activities to avoid burnout.")
        elif stress_score == 0:
            suggestions.append("🎉 Low Stress: Great job balancing your academics! Continue to prioritize your health, get enough sleep, and take part in activities you enjoy.")

        # Additional suggestions based on specific factors
        if input_data.get('academic_performance') and input_data['academic_performance'] < 3:
            suggestions.append("📚 Academic Struggles: Try new study techniques like the Pomodoro method to improve focus. Consider joining a study group or seeking help from a tutor.")
        if input_data.get('anxiety_level') and input_data['anxiety_level'] > 15:
            suggestions.append("😨 High Anxiety: This can significantly impact your health. Look into campus resources for mental health support, or try mindfulness and deep breathing exercises.")
        if input_data.get('financial_stress') and input_data['financial_stress'] == 'High':
            suggestions.append("💸 Financial Worries: Don't face this alone. Look into financial aid, scholarships, or part-time job opportunities provided by the university.")
            
    return suggestions
    
# --- API Endpoints ---
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({"error": "Missing email or password."}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(
        first_name=data.get('first_name'),
        last_name=data.get('last_name'),
        contact=data.get('contact'),
        email=email,
        password_hash=hashed_password
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created. Please log in to complete your profile."}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        response_data = {
            "message": "Login successful!",
            "user_id": user.id,
            "is_new_user": user.is_new_user,
            "role": user.role
        }
        return jsonify(response_data), 200
    return jsonify({"error": "Invalid email or password."}), 401

@app.route('/profile', methods=['POST'])
def save_profile():
    data = request.get_json()
    user_id = data.get('user_id')
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404
    user.first_name = data.get('first_name')
    user.last_name = data.get('last_name')
    user.age = data.get('age')
    user.gender = data.get('gender')
    user.contact = data.get('contact')
    user.role = data.get('mode')
    user.is_new_user = False
    db.session.commit()
    return jsonify({"message": "Profile updated successfully."}), 200

def safe_int_cast(value):
    """Safely converts a value to an integer, returning 0 on failure."""
    if value is None:
        return 0
    try:
        # Tries to convert string/float/int directly
        return int(value)
    except (ValueError, TypeError):
        # Catches cases like empty string "" or malformed text
        return 0

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id')
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    # --- DEBUG: RAW INCOMING DATA ---
    print("\n--- DEBUG: RAW INCOMING JSON PAYLOAD ---")
    print(data)
    print("---------------------------------------\n")
    # --- END DEBUG ---

    # 1. Prepare input data for CSV and DB (FIXED: Using safe_int_cast for robust input handling)
    input_data = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'role': user.role,
        'job_role': data.get('job_role'), 
        'working_hours': safe_int_cast(data.get('working_hours')),
        'virtual_meetings': safe_int_cast(data.get('virtual_meetings')),
        'work_life_balance': safe_int_cast(data.get('work_life_balance')),
        'access_to_mental_health': data.get('access_to_mental_health'), 
        'satisfaction_with_remote_work': data.get('satisfaction_with_remote_work'),
        'company_support': safe_int_cast(data.get('company_support')),
        'physical_activity': data.get('physical_activity'),
        'sleep_quality': data.get('sleep_quality'),
        'anxiety_level': safe_int_cast(data.get('anxiety_level')),
        'depression': safe_int_cast(data.get('depression')),
        'academic_performance': safe_int_cast(data.get('academic_performance')),
        'study_load': safe_int_cast(data.get('study_load')),
        'teacher_student_relationship': safe_int_cast(data.get('teacher_student_relationship')),
        'future_career_concerns': safe_int_cast(data.get('future_career_concerns')),
        'social_support': safe_int_cast(data.get('social_support')),
        'peer_pressure': safe_int_cast(data.get('peer_pressure')),
        'extracurricular_load': safe_int_cast(data.get('extracurricular_load'))
    }

    # 2. Write to CSV
    csv_file_path = 'user_inputs.csv'
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(input_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_data)

    # 3. Preprocess and Predict
    df_features = pd.DataFrame([input_data])
    model = None
    
    try:
        if user.role == 'employee':
            # Drop unnecessary columns first
            df_features = df_features.drop(columns=['role', 'timestamp', 'user_id', 'anxiety_level', 'depression', 'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_load'], errors='ignore')
            # Preprocess the relevant features for the model (includes OHE simulation and final ordering)
            df_features = preprocess_employee_data_for_prediction(df_features)
            if employee_model:
                model = employee_model
        elif user.role == 'student':
            # Corrected drop: Remove employee features and metadata, KEEP all student features including sleep_quality
            df_features = df_features.drop(columns=[
                'role', 'timestamp', 'user_id', 
                'job_role', 'working_hours', 'virtual_meetings', 'work_life_balance', 
                'access_to_mental_health', 'satisfaction_with_remote_work', 
                'company_support', 'physical_activity' 
            ], errors='ignore')
            # Preprocess the relevant features for the model (maps sleep_quality and enforces ordering)
            df_features = preprocess_student_data_for_prediction(df_features)
            if student_model:
                model = student_model
        else:
            return jsonify({"error": "Invalid role."}), 400

        if model is None:
            return jsonify({"error": "Model not loaded."}), 500
        
        # Prediction
        prediction = model.predict(df_features)
        stress_score = float(prediction[0])
        
        # 4. Save to Database
        history_data = {key: value for key, value in input_data.items() if key not in ['role']}
        history_data['stress_score'] = stress_score # Add the predicted score
        new_history = StressHistory(**history_data)
        
        db.session.add(new_history)
        db.session.commit()
        
        # CORRECTED FUNCTION CALL: Pass the user's role
        suggestions = generate_suggestions(stress_score, input_data, user.role)
        
        return jsonify({"stress_score": stress_score, "suggestions": suggestions}), 200
    except Exception as e:
        print(f"Prediction Error: {e}") 
        return jsonify({"error": f"Prediction failed due to server error: {e}"}), 400

@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    history_records = StressHistory.query.filter_by(user_id=user_id).order_by(StressHistory.timestamp).all()
    history_data = [{
        "stress_score": record.stress_score,
        "timestamp": record.timestamp.isoformat(),
        "factors": {
            "sleep_quality": record.sleep_quality,
            "workload": record.working_hours if record.working_hours is not None else record.study_load
        }
    } for record in history_records]
    return jsonify(history_data)

# No __name__ == '__main__' block here
# Render will use Gunicorn to run the app
