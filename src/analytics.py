import sqlite3
import datetime
from typing import Dict, Any, List

DB_PATH = "memory.db"

def init_analytics_db():
    """Initializes analytics and feedback tables in SQLite with automatic schema migration."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tickets and Triage log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticket_id TEXT,
        thread_id TEXT,
        query TEXT,
        department TEXT,
        intent TEXT,
        sentiment TEXT,
        priority TEXT,
        approval_required INTEGER,
        final_response TEXT,
        latency_sec REAL DEFAULT 0.0,
        token_count INTEGER DEFAULT 0
    )
    """)
    
    # Safe schema migration for existing databases
    cursor.execute("PRAGMA table_info(ticket_logs)")
    columns = [col[1] for col in cursor.fetchall()]
    if "latency_sec" not in columns:
        cursor.execute("ALTER TABLE ticket_logs ADD COLUMN latency_sec REAL DEFAULT 0.0")
    if "token_count" not in columns:
        cursor.execute("ALTER TABLE ticket_logs ADD COLUMN token_count INTEGER DEFAULT 0")
    
    # Feedback log for MLOps and continuous evaluation
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        ticket_id TEXT,
        rating INTEGER,
        comment TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def log_ticket_event(
    ticket_id: str,
    thread_id: str,
    query: str,
    department: str,
    intent: str,
    sentiment: str,
    priority: str,
    approval_required: bool,
    final_response: str,
    latency_sec: float = 0.0,
    token_count: int = 0
):
    """Logs a completed query for real-time monitoring and analytics."""
    try:
        init_analytics_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO ticket_logs (
            timestamp, ticket_id, thread_id, query, department, intent,
            sentiment, priority, approval_required, final_response,
            latency_sec, token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticket_id,
            thread_id,
            query,
            department or "General",
            intent or "General",
            sentiment or "Neutral",
            priority or "Medium",
            1 if approval_required else 0,
            final_response,
            round(float(latency_sec), 2),
            int(token_count)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging ticket event: {e}")

def log_feedback(ticket_id: str, rating: int, comment: str = ""):
    """Stores user feedback (+1 for positive, 0 for negative)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO ticket_feedback (timestamp, ticket_id, rating, comment)
        VALUES (?, ?, ?, ?)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticket_id,
            rating,
            comment
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging feedback: {e}")

def get_analytics_metrics() -> Dict[str, Any]:
    """Retrieves operational KPIs and distribution data for the dashboard."""
    init_analytics_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM ticket_logs")
    total_tickets = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM ticket_logs WHERE approval_required = 1")
    escalated_count = cursor.fetchone()[0]
    
    # Latency and token metrics
    cursor.execute("SELECT AVG(latency_sec) FROM ticket_logs WHERE latency_sec > 0")
    avg_latency_row = cursor.fetchone()
    avg_latency = round(avg_latency_row[0], 2) if avg_latency_row and avg_latency_row[0] is not None else 0.0
    
    cursor.execute("SELECT SUM(token_count) FROM ticket_logs")
    total_tokens_row = cursor.fetchone()
    total_tokens = total_tokens_row[0] if total_tokens_row and total_tokens_row[0] is not None else 0
    
    cursor.execute("SELECT AVG(token_count) FROM ticket_logs WHERE token_count > 0")
    avg_tokens_row = cursor.fetchone()
    avg_tokens = int(avg_tokens_row[0]) if avg_tokens_row and avg_tokens_row[0] is not None else 0
    
    # Department breakdown
    cursor.execute("SELECT department, COUNT(*) FROM ticket_logs GROUP BY department")
    departments = dict(cursor.fetchall())
    
    # Sentiment breakdown
    cursor.execute("SELECT sentiment, COUNT(*) FROM ticket_logs GROUP BY sentiment")
    sentiments = dict(cursor.fetchall())
    
    # Priority breakdown
    cursor.execute("SELECT priority, COUNT(*) FROM ticket_logs GROUP BY priority")
    priorities = dict(cursor.fetchall())
    
    # Feedback stats
    cursor.execute("SELECT COUNT(*), SUM(rating) FROM ticket_feedback")
    feedback_row = cursor.fetchone()
    total_feedback = feedback_row[0] or 0
    positive_feedback = feedback_row[1] or 0
    satisfaction_rate = int((positive_feedback / total_feedback) * 100) if total_feedback > 0 else 100
    
    # Recent tickets with latency and token telemetry
    cursor.execute("""
    SELECT timestamp, ticket_id, department, sentiment, priority, approval_required, latency_sec, token_count, query
    FROM ticket_logs ORDER BY id DESC LIMIT 10
    """)
    recent = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_tickets": total_tickets,
        "escalated_count": escalated_count,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "departments": departments,
        "sentiments": sentiments,
        "priorities": priorities,
        "total_feedback": total_feedback,
        "satisfaction_rate": satisfaction_rate,
        "recent_tickets": recent
    }

