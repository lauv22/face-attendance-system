from database.db_config import get_db_connection
import psycopg2

def create_tables():
    """Create all tables exactly as per your schema (idempotent)."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Table 1: persons
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                employee_id VARCHAR(50) UNIQUE NOT NULL,
                department VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 2: face_embeddings (512-dim for FaceNet512)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                embedding REAL[],          -- 512 float values
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 3: attendance
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                date DATE NOT NULL,
                time TIME NOT NULL,
                status VARCHAR(20) CHECK (status IN ('Present', 'Late', 'Unknown')),
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 4: system_logs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(50),
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        print("✅ All tables created successfully in PostgreSQL!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating tables: {e}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    create_tables()