from flask import Flask, render_template, request, jsonify, send_file
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
    delete_person_by_employee_id, log_event
)
from database.db_config import get_db_connection

app = Flask(__name__)
detector = FaceDetector()

TOTAL_PHOTOS = 20
registration_buffer = {}

# ====================== MAIN PAGES ======================
@app.route('/')
def index():
    stats = get_dashboard_stats()
    recent = get_todays_attendance()[:5]  # last 5 records
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
            return jsonify({"status": "error", "message": "No face detected"})

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
            return jsonify({"status": "success", "message": f"Registration completed with {TOTAL_PHOTOS} photos!"})

        return jsonify({"status": "success", "captured": captured})

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
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT p.id as person_id, e.embedding FROM persons p JOIN face_embeddings e ON p.id = e.person_id")
        data = cur.fetchall()
        cur.close()
        conn.close()

        embeddings_list = [(row['person_id'], np.array(row['embedding'], dtype=np.float32)) for row in data]
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
                status = "Late" if now.hour >= 9 and now.minute > 0 else "Present"
                marked = mark_attendance(person_id, status, score)

                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT name FROM persons WHERE id = %s", (person_id,))
                name_row = cur.fetchone()
                name = name_row['name'] if name_row else "Unknown"
                cur.close()
                conn.close()

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
                    "confidence": round(score, 3),
                    "box": [x, y, w, h]
                })

        return jsonify({"detections": results})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_today_attendance')
def get_today_attendance_route():
    return jsonify(get_todays_attendance())

# ====================== DELETE PERSON ======================
@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
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
        'Name': r['name'],
        'Employee ID': r['employee_id'],
        'Department': r['department'],
        'Date': r['date'],
        'Time': r['time'],
        'Status': r['status'],
        'Confidence': r['confidence_score']
    } for r in records])
    
    csv_path = "attendance_records.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route('/export_excel')
def export_excel():
    records = get_attendance_records()
    df = pd.DataFrame([{
        'Name': r['name'],
        'Employee ID': r['employee_id'],
        'Department': r['department'],
        'Date': r['date'],
        'Time': r['time'],
        'Status': r['status'],
        'Confidence': r['confidence_score']
    } for r in records])
    
    excel_path = "attendance_records.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    print("🚀 Full Face Attendance System Started!")
    print("   Dashboard     → http://127.0.0.1:5000/")
    print("   Register      → http://127.0.0.1:5000/register")
    print("   Live Attendance → http://127.0.0.1:5000/attendance")
    app.run(debug=True, port=5000)