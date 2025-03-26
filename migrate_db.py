import sqlite3
import os

def migrate_database():
    """Add the reasoning column to the moderation_results table."""
    
    print("Starting database migration...")
    
    # Connect to the database - try different possible paths
    possible_paths = [
        'users.db',
        'instance/users.db',  # Flask default SQLite location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'users.db')
    ]
    
    db_path = None
    for path in possible_paths:
        if os.path.exists(path):
            db_path = path
            print(f"Found database at: {db_path}")
            break
    
    if not db_path:
        print("Database file not found in any of the expected locations.")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if reasoning column already exists
        cursor.execute("PRAGMA table_info(moderation_results)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'reasoning' not in column_names:
            print("Adding 'reasoning' column to moderation_results table...")
            cursor.execute("ALTER TABLE moderation_results ADD COLUMN reasoning TEXT")
            conn.commit()
            print("Column added successfully.")
        else:
            print("The 'reasoning' column already exists. No changes needed.")
        
        return True
    
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        return False
    
    finally:
        conn.close()

if __name__ == "__main__":
    success = migrate_database()
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed or was not needed.") 