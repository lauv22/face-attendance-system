from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import base64
import pandas as pd
from datetime import datetime, date

from models.face_detector import FaceDetector
from models.face_recognizer import recognizer
from models.similarity import matcher
from database.db_queries import (
    add_person, save_embedding, mark_attendance,
    get_todays_attendance, get_dashboard_stats,
    get_all_persons, get_attendance_records,
    delete_person_by_employee_id, log_event,
    get_all_embeddings, get_person_by_id
)

app = Flask(__name__)
detector = FaceDetector()

TOTAL_PHOTOS = 20
registration_buffer = {}

# ====================== PAGES ======================
@app.route('/')
def index():
    stats = get_dashboard_stats()
    recent = get_todays_attendance()[:5]
    return render_template('index.html', stats=stats, recent=recent)

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/persons')
def persons_page():
    persons = get_all_persons()
    return render_template('persons.html', persons=persons)

@app.route('/records')
def records_page():
    records = get_attendance_records()
    return render_template('records.html', records=records)

# ====================== REGISTRATION ======================
@app.route('/register_capture', methods=['POST'])
def register_capture():
    try:
        data = request.get_json()
        image_base64 = data['image'].split(',')[1]
        name = data['name'].strip()
        employee_id = data['employee_id'].strip()
        department = data['department'].strip()

        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = detector.detect_faces(frame)
        if not detections:
            return jsonify({"status": "no_face", "message": "No face detected"})

        x, y, w, h = detections[0]['box']
        face_crop = frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))

        embedding = recognizer.get_embedding(face_crop)
        if embedding is None:
            return jsonify({"status": "error", "message": "Could not generate embedding"})

        if employee_id not in registration_buffer:
            registration_buffer[employee_id] = []
        registration_buffer[employee_id].append(embedding)

        captured = len(registration_buffer[employee_id])

        if captured >= TOTAL_PHOTOS:
            avg_embedding = recognizer.average_embeddings(registration_buffer[employee_id])
            person_id = add_person(name, employee_id, department)
            save_embedding(person_id, avg_embedding.tolist())
            del registration_buffer[employee_id]
            log_event("REGISTRATION", f"Completed: {name} ({employee_id})")
            return jsonify({"status": "success", "message": "Registration completed!"})

        return jsonify({"status": "capturing", "captured": captured, "total": TOTAL_PHOTOS})

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/cancel_registration', methods=['POST'])
def cancel_registration():
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        if employee_id and employee_id in registration_buffer:
            del registration_buffer[employee_id]
        return jsonify({"status": "cancelled"})
    except Exception:
        return jsonify({"status": "error"})

# ====================== LIVE ATTENDANCE ======================
@app.route('/load_embeddings', methods=['POST'])
def load_embeddings():
    try:
        embeddings_list = get_all_embeddings()
        matcher.load_embeddings_from_db(embeddings_list)
        return jsonify({"status": "success", "count": len(embeddings_list)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_base64 = data['image'].split(',')[1]

        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = detector.detect_faces(frame)
        results = []

        for det in detections:
            x, y, w, h = det['box']
            face_crop = frame[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (160, 160))

            embedding = recognizer.get_embedding(face_crop)
            if embedding is None:
                continue

            person_id, score = matcher.find_match(embedding, threshold=0.65)

            if person_id:
                now = datetime.now()
                status = "Late" if (now.hour > 9) or (now.hour == 9 and now.minute > 0) else "Present"
                marked = mark_attendance(person_id, status, score)

                person = get_person_by_id(person_id)
                name = person['name'] if person else "Unknown"

                results.append({
                    "name": name,
                    "status": status if marked else "Already Marked",
                    "confidence": round(score, 3),
                    "box": [x, y, w, h]
                })
            else:
                results.append({
                    "name": "Unknown",
                    "status": "Unknown",
                    "confidence": round(score, 3) if score else 0,
                    "box": [x, y, w, h]
                })

        return jsonify({"detections": results})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_today_attendance')
def get_today_attendance_route():
    return jsonify(get_todays_attendance())

# ====================== DELETE ======================
@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        if request.is_json:
            employee_id = request.get_json().get('employee_id')
        else:
            employee_id = request.form.get('employee_id')
        if delete_person_by_employee_id(employee_id):
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Person not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ====================== EXPORT ======================
@app.route('/export_csv')
def export_csv():
    records = get_attendance_records()
    df = pd.DataFrame([{
        'Name': r['name'], 'Employee ID': r['employee_id'],
        'Department': r['department'], 'Date': r['date'],
        'Time': r['time'], 'Status': r['status'],
        'Confidence': r['confidence_score']
    } for r in records])
    csv_path = "attendance_records.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route('/export_excel')
def export_excel():
    records = get_attendance_records()
    df = pd.DataFrame([{
        'Name': r['name'], 'Employee ID': r['employee_id'],
        'Department': r['department'], 'Date': r['date'],
        'Time': r['time'], 'Status': r['status'],
        'Confidence': r['confidence_score']
    } for r in records])
    excel_path = "attendance_records.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

# ====================== STARTUP ======================
if __name__ == '__main__':
    from database.db_setup import create_tables
    create_tables()
    print("🚀 Face Attendance System Started!")
    print("   Dashboard      → http://127.0.0.1:5000/")
    print("   Register       → http://127.0.0.1:5000/register")
    print("   Live Attendance → http://127.0.0.1:5000/attendance")
    app.run(host='0.0.0.0', debug=True, port=5000)
