import sqlite3
import datetime
from typing import Dict, Any, List

DB_PATH = "memory.db"

def init_analytics_db():
    """Initializes analytics and feedback tables in SQLite."""
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
        final_response TEXT
    )
    """)
    
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
    final_response: str
):
    """Logs a completed query for real-time monitoring and analytics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO ticket_logs (
            timestamp, ticket_id, thread_id, query, department, intent,
            sentiment, priority, approval_required, final_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            final_response
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
    
    # Recent tickets
    cursor.execute("""
    SELECT timestamp, ticket_id, department, sentiment, priority, approval_required, query
    FROM ticket_logs ORDER BY id DESC LIMIT 10
    """)
    recent = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_tickets": total_tickets,
        "escalated_count": escalated_count,
        "departments": departments,
        "sentiments": sentiments,
        "priorities": priorities,
        "total_feedback": total_feedback,
        "satisfaction_rate": satisfaction_rate,
        "recent_tickets": recent
    }
