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
import psycopg2

# Initialize Flask app and extensions
app = flask.Flask(__name__)
CORS(app)

# Database Configuration (Update with your MySQL credentials)
app.config['SQLALCHEMY_DATABASE_URI'] =postgresql://clampulsedb_user:k11oWfFOCyyhhMai9TgcKmfPgyCCDldm@dpg-d3n3s23e5dus73ejq740-a/clampulsedb
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- Database Models ---
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

# Create the database and tables (Run this file once)
with app.app_context():
    db.create_all()

# Load the machine learning models
try:
    with open('employee_model.pkl', 'rb') as f:
        employee_model = pickle.load(f)
    with open('student_model.pkl', 'rb') as f:
        student_model = pickle.load(f)
except FileNotFoundError:
    employee_model, student_model = None, None
    print("Error: Model files not found. Check the file paths.")

# --- Preprocessing Functions ---
def preprocess_employee_data(df):
    df = df.rename(columns={
        'Working_Hours': 'Hours_Worked_Per_Week',
        'Virtual_Meetings': 'Number_of_Virtual_Meetings'
    })
    
    df['Access_to_Mental_Health_Resources'] = df['Access_to_Mental_Health_Resources'].replace({'Yes': 1, 'No': 0})
    
    physical_activity_mapping = {'None': 0, 'Weekly': 1, 'Daily': 2}
    sleep_quality_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    
    df['Physical_Activity'] = df['Physical_Activity'].map(physical_activity_mapping)
    df['Sleep_Quality'] = df['Sleep_Quality'].map(sleep_quality_mapping)
    
    df['meetings_per_hour'] = df['Number_of_Virtual_Meetings'] / df['Hours_Worked_Per_Week']
    df['meetings_per_hour'] = df['meetings_per_hour'].replace([np.inf, -np.inf], 0)
    
    return df

def preprocess_student_data(df):
    # Map Sleep Quality from 1-5 to a different scale if needed by the model
    # For example, 1-5 scale can be used directly, so no mapping is needed here
    # The rest of your student inputs are already numerical
    return df

@app.route('/')
def home():
    return jsonify({"message": "CalmPulse Flask backend is running successfully on Render!"})

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id')
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    # Prepare input data for CSV and DB
    input_data = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'role': user.role,
        'job_role': data.get('job_role'), 'working_hours': data.get('working_hours'),
        'virtual_meetings': data.get('virtual_meetings'), 'work_life_balance': data.get('work_life_balance'),
        'access_to_mental_health': data.get('access_to_mental_health'), 'satisfaction_with_remote_work': data.get('satisfaction_with_remote_work'),
        'company_support': data.get('company_support'), 'physical_activity': data.get('physical_activity'),
        'sleep_quality': data.get('sleep_quality'),
        'anxiety_level': data.get('anxiety_level'), 'depression': data.get('depression'),
        'academic_performance': data.get('academic_performance'), 'study_load': data.get('study_load'),
        'teacher_student_relationship': data.get('teacher_student_relationship'),
        'future_career_concerns': data.get('future_career_concerns'), 'social_support': data.get('social_support'),
        'peer_pressure': data.get('peer_pressure'), 'extracurricular_load': data.get('extracurricular_load')
    }

    # Write to CSV
    csv_file_path = 'user_inputs.csv'
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(input_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_data)

    # Preprocess data for the model
    df_features = pd.DataFrame([input_data])
    model = None
    if user.role == 'employee':
        df_features = preprocess_employee_data(df_features)
        if employee_model:
            model = employee_model
    elif user.role == 'student':
        df_features = preprocess_student_data(df_features)
        if student_model:
            model = student_model
    else:
        return jsonify({"error": "Invalid role."}), 400

    try:
        if model is None:
            return jsonify({"error": "Model not loaded."}), 500
        
        prediction = model.predict(df_features)
        stress_score = float(prediction[0])
        
        new_history = StressHistory(**input_data, stress_score=stress_score)
        db.session.add(new_history)
        db.session.commit()
        
        suggestions = generate_suggestions(stress_score, input_data)
        
        return jsonify({"stress_score": stress_score, "suggestions": suggestions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    history_records = StressHistory.query.filter_by(user_id=user_id).order_by(StressHistory.timestamp).all()
    history_data = [{
        "stress_score": record.stress_score,
        "timestamp": record.timestamp.isoformat(),
        "factors": {
            "sleep_quality": record.sleep_quality,
            "workload": record.workload if record.workload else record.study_load
        }
    } for record in history_records]
    return jsonify(history_data)
def generate_suggestions(stress_score, input_data):
    suggestions = []
    
    # --- Overall Stress Score Based Suggestions ---
    if stress_score >= 3.0:
        suggestions.append("You are experiencing a high level of stress. Take a break and try some relaxation techniques.")
    else:
        suggestions.append("Your stress level is low. Keep up the good work and maintain a healthy lifestyle.")
    
    # --- Employee Specific Suggestions ---
    if input_data.get('working_hours') is not None and input_data['working_hours'] > 45:
        suggestions.append("Long working hours are a major contributor to stress. Try to set boundaries and take regular breaks to avoid burnout.")
    
    if input_data.get('virtual_meetings') is not None and input_data['virtual_meetings'] > 5:
        suggestions.append("Too many virtual meetings can be draining. Try to consolidate meetings or suggest alternative communication methods.")

    if input_data.get('work_life_balance') is not None and input_data['work_life_balance'] <= 2:
        suggestions.append("Your work-life balance is low. Try to disconnect from work after hours and schedule time for personal activities.")

    if input_data.get('physical_activity') == 'None':
        suggestions.append("Physical activity is a great stress reliever. Try to incorporate a daily walk or some form of exercise into your routine.")
        
    if input_data.get('sleep_quality') in ['Poor', 1]:
        suggestions.append("Poor sleep quality contributes to stress. Try to establish a regular sleep schedule and create a relaxing bedtime routine.")

    # --- Student Specific Suggestions ---
    if input_data.get('anxiety_level') is not None and input_data['anxiety_level'] > 15:
        suggestions.append("Your anxiety level is high. Consider talking to a professional or using mindfulness apps to manage your thoughts.")
    
    if input_data.get('depression') is not None and input_data['depression'] > 15:
        suggestions.append("Depression seems to be a significant factor. Seeking professional help from a therapist or counselor is highly recommended.")
        
    if input_data.get('academic_performance') is not None and input_data['academic_performance'] <= 2:
        suggestions.append("Low academic performance can be a major stressor. Try time management and study hacks to improve your focus.")

    if input_data.get('study_load') is not None and input_data['study_load'] >= 4:
        suggestions.append("Your study load is high. Break down large tasks into smaller, more manageable ones with short breaks in between.")
    
    if input_data.get('peer_pressure') is not None and input_data['peer_pressure'] >= 4:
        suggestions.append("Peer pressure may be a major stressor. Focus on your goals and don't compare yourself to others.")
    
    if input_data.get('social_support') is not None and input_data['social_support'] <= 2:
        suggestions.append("Social support is key to managing stress. Try to connect with friends and family more often.")
    
    # --- General Resources ---
    suggestions.append("For more tips and helpful videos, visit the Stress Management Forum page on the app.")
    
    return suggestions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
