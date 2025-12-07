#!/usr/bin/env python3
"""Script to stamp the database with the current Alembic revision."""
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.database import get_database_url

def stamp_database():
    """Stamp the database with the latest revision."""
    db_url = get_database_url()
    # Extract path from sqlite:///path
    db_path = db_url.replace("sqlite:///", "")

    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create alembic_version table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL,
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
        )
    """)

    # Check if there's already a version
    cursor.execute("SELECT version_num FROM alembic_version")
    existing = cursor.fetchone()

    if existing:
        print(f"Current version: {existing[0]}")
        # Update to latest
        cursor.execute("UPDATE alembic_version SET version_num = ?", ("003",))
        print("Updated to version 003")
    else:
        # Insert latest version
        cursor.execute("INSERT INTO alembic_version (version_num) VALUES (?)", ("003",))
        print("Stamped database with version 003")

    conn.commit()
    conn.close()
    print("Done!")

if __name__ == "__main__":
    stamp_database()
