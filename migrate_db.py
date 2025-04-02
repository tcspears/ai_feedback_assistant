import sqlite3
import os

def migrate_database():
    """Add macro categories table and update feedback_macros table."""
    
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
        # Create macro_categories table
        print("Creating macro_categories table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rubric_id INTEGER NOT NULL,
            name VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (rubric_id) REFERENCES rubrics (id)
        )
        """)

        # Check if category_id column exists in feedback_macros
        cursor.execute("PRAGMA table_info(feedback_macros)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'category_id' not in column_names:
            print("Adding category_id column to feedback_macros table...")
            # First, rename the existing category column
            cursor.execute("ALTER TABLE feedback_macros RENAME TO feedback_macros_old")
            
            # Create new table with updated schema
            cursor.execute("""
            CREATE TABLE feedback_macros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rubric_id INTEGER NOT NULL,
                criteria_id INTEGER,
                name VARCHAR(255) NOT NULL,
                text TEXT NOT NULL,
                category_id INTEGER,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (rubric_id) REFERENCES rubrics (id),
                FOREIGN KEY (criteria_id) REFERENCES rubric_criteria (id),
                FOREIGN KEY (category_id) REFERENCES macro_categories (id)
            )
            """)
            
            # Copy data from old table to new table
            cursor.execute("""
            INSERT INTO feedback_macros (id, rubric_id, criteria_id, name, text, created_at)
            SELECT id, rubric_id, criteria_id, name, text, created_at
            FROM feedback_macros_old
            """)
            
            # Drop old table
            cursor.execute("DROP TABLE feedback_macros_old")
            
            print("Successfully migrated feedback_macros table")
        
        conn.commit()
        print("Migration completed successfully")
        return True
    
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        return False
    
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_database() 