from database.db_config import get_db_connection
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)

def log_event(event_type: str, message: str):
    """Central logging function used everywhere."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO system_logs (event_type, message)
            VALUES (%s, %s)
        """, (event_type, message))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")
    finally:
        cur.close()
        conn.close()

def add_person(name: str, employee_id: str, department: str) -> int:
    """Create new person and return person_id."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO persons (name, employee_id, department)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (name.strip(), employee_id.strip(), department.strip()))
        person_id = cur.fetchone()['id']
        conn.commit()
        log_event("REGISTRATION", f"New person registered: {name} ({employee_id})")
        return person_id
    except Exception as e:
        conn.rollback()
        if "unique constraint" in str(e).lower():
            raise ValueError(f"Employee ID '{employee_id}' already exists!")
        raise
    finally:
        cur.close()
        conn.close()

def save_embedding(person_id: int, embedding: list):
    """Save 512-dim embedding."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO face_embeddings (person_id, embedding)
            VALUES (%s, %s)
        """, (person_id, embedding))
        conn.commit()
        log_event("EMBEDDING", f"Embedding saved for person_id {person_id}")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def mark_attendance(person_id: int, status: str, confidence: float):
    """Mark attendance (prevents duplicate on same day)."""
    today = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM attendance 
            WHERE person_id = %s AND date = %s
        """, (person_id, today))
        if cur.fetchone():
            return False  # already marked today

        cur.execute("""
            INSERT INTO attendance (person_id, date, time, status, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (person_id, today, datetime.now().time(), status, confidence))
        conn.commit()
        log_event("ATTENDANCE", f"Marked {status} for person_id {person_id}")
        return True
    except Exception as e:
        conn.rollback()
        print(f"❌ Attendance error: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def get_todays_attendance():
    """Return today's attendance for the live log."""
    today = date.today()
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT p.name, p.employee_id, a.time, a.status, a.confidence_score
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = %s
            ORDER BY a.time DESC
        """, (today,))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

def delete_person_by_employee_id(employee_id: str) -> bool:
    """Delete person + all embeddings + attendance (used later)."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            DELETE FROM persons 
            WHERE employee_id = %s
            RETURNING name
        """, (employee_id.strip(),))
        result = cur.fetchone()
        conn.commit()
        if result:
            log_event("DELETION", f"Person deleted: {result['name']} ({employee_id})")
            return True
        return False
    except Exception as e:
        conn.rollback()
        print(f"❌ Delete error: {e}")
        return False
    finally:
        cur.close()
        conn.close()