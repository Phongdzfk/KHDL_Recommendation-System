"""
Database module for storing user history and ratings
SQLite-based storage for user interactions
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

class UserHistoryDB:
    def __init__(self, db_path='data/user_history.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.db_dir = Path(db_path).parent
        
        # Create directory if not exists
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                game_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, game_id)
            )
        ''')
        
        # Recommendations viewed table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations_viewed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                game_id INTEGER NOT NULL,
                recommendation_type TEXT,
                clicked INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Search history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                search_query TEXT,
                results_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_game ON ratings(game_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendations_viewed(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id)')
        
        conn.commit()
        conn.close()
    
    def create_new_user(self, user_id=None):
        """Create a new user and return user_id
        
        Args:
            user_id: Optional user_id. If None, generates a new UUID.
        """
        provided_user_id = user_id  # Save original to check if it was provided
        if user_id is None:
            import uuid
            user_id = str(uuid.uuid4())[:8]  # Short UUID
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, created_at, last_active)
                VALUES (?, ?, ?)
            ''', (user_id, datetime.now(), datetime.now()))
            conn.commit()
        except sqlite3.IntegrityError:
            # If user_id was provided and exists, just return it (user already exists)
            if provided_user_id is not None:
                user_id = provided_user_id  # Return the provided user_id
            else:
                # If user_id was auto-generated, generate a new one
                import uuid
                user_id = str(uuid.uuid4())[:8]
                try:
                    cursor.execute('''
                        INSERT INTO users (user_id, created_at, last_active)
                        VALUES (?, ?, ?)
                    ''', (user_id, datetime.now(), datetime.now()))
                    conn.commit()
                except sqlite3.IntegrityError:
                    # If still fails, try one more time
                    user_id = str(uuid.uuid4())[:8]
                    cursor.execute('''
                        INSERT INTO users (user_id, created_at, last_active)
                        VALUES (?, ?, ?)
                    ''', (user_id, datetime.now(), datetime.now()))
                    conn.commit()
        finally:
            conn.close()
        
        return user_id
    
    def delete_user(self, user_id):
        """Delete a user and all their data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete ratings
            cursor.execute('DELETE FROM ratings WHERE user_id = ?', (user_id,))
            # Delete recommendations viewed
            cursor.execute('DELETE FROM recommendations_viewed WHERE user_id = ?', (user_id,))
            # Delete search history
            cursor.execute('DELETE FROM search_history WHERE user_id = ?', (user_id,))
            # Delete user
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def add_rating(self, user_id, game_id, rating):
        """Add or update a rating"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update or insert rating
        cursor.execute('''
            INSERT OR REPLACE INTO ratings (user_id, game_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, game_id, float(rating), datetime.now()))
        
        # Update user last_active
        cursor.execute('''
            UPDATE users SET last_active = ? WHERE user_id = ?
        ''', (datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    def get_user_ratings(self, user_id):
        """Get all ratings for a user"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT game_id, rating, timestamp
            FROM ratings
            WHERE user_id = ?
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        
        return df
    
    def get_user_rating_for_game(self, user_id, game_id):
        """Get rating for a specific game by user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT rating FROM ratings
            WHERE user_id = ? AND game_id = ?
        ''', (user_id, game_id))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_all_user_ratings(self):
        """Get all ratings from all users"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT user_id, game_id, rating, timestamp
            FROM ratings
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def log_recommendation(self, user_id, game_id, recommendation_type='game-based', clicked=0):
        """Log that a recommendation was shown/clicked"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if recommendation already exists
        cursor.execute('''
            SELECT id, clicked FROM recommendations_viewed
            WHERE user_id = ? AND game_id = ? AND recommendation_type = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (user_id, game_id, recommendation_type))
        
        existing = cursor.fetchone()
        
        if existing:
            # If already exists and clicked=0, update to clicked=1 if requested
            if clicked == 1 and existing[1] == 0:
                cursor.execute('''
                    UPDATE recommendations_viewed
                    SET clicked = 1, timestamp = ?
                    WHERE id = ?
                ''', (datetime.now(), existing[0]))
        else:
            # Insert new recommendation
            cursor.execute('''
                INSERT INTO recommendations_viewed (user_id, game_id, recommendation_type, clicked, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, game_id, recommendation_type, clicked, datetime.now()))
        
        # Update user last_active
        cursor.execute('''
            UPDATE users SET last_active = ? WHERE user_id = ?
        ''', (datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    def update_recommendation_clicked(self, user_id, game_id, recommendation_type='game-based'):
        """Update recommendation clicked status to 1"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the most recent unclicked recommendation
        cursor.execute('''
            SELECT id FROM recommendations_viewed
            WHERE user_id = ? AND game_id = ? AND recommendation_type = ?
            AND clicked = 0
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (user_id, game_id, recommendation_type))
        
        result = cursor.fetchone()
        if result:
            # Update the most recent recommendation
            cursor.execute('''
                UPDATE recommendations_viewed
                SET clicked = 1, timestamp = ?
                WHERE id = ?
            ''', (datetime.now(), result[0]))
        else:
            # If no unclicked recommendation found, create a new one with clicked=1
            cursor.execute('''
                INSERT INTO recommendations_viewed (user_id, game_id, recommendation_type, clicked, timestamp)
                VALUES (?, ?, ?, 1, ?)
            ''', (user_id, game_id, recommendation_type, datetime.now()))
        
        # Update user last_active
        cursor.execute('''
            UPDATE users SET last_active = ? WHERE user_id = ?
        ''', (datetime.now(), user_id))
        
        conn.commit()
        conn.close()
    
    def log_search(self, user_id, search_query, results_count):
        """Log search queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_history (user_id, search_query, results_count, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, search_query, results_count, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_recommendation_stats(self, user_id):
        """Get statistics about recommendations for a user (realtime)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total unique games recommended (count distinct game_id)
        cursor.execute('''
            SELECT COUNT(DISTINCT game_id) FROM recommendations_viewed
            WHERE user_id = ?
        ''', (user_id,))
        total_recommendations = cursor.fetchone()[0]
        
        # Total unique games clicked (count distinct game_id where clicked=1)
        cursor.execute('''
            SELECT COUNT(DISTINCT game_id) FROM recommendations_viewed
            WHERE user_id = ? AND clicked = 1
        ''', (user_id,))
        total_clicks = cursor.fetchone()[0]
        
        # Total recommendation sessions (count distinct timestamps grouped by date)
        # This counts how many times user got recommendations
        cursor.execute('''
            SELECT COUNT(DISTINCT DATE(timestamp)) FROM recommendations_viewed
            WHERE user_id = ?
        ''', (user_id,))
        total_sessions = cursor.fetchone()[0]
        
        # Click rate (percentage of unique games that were clicked)
        click_rate = (total_clicks / total_recommendations * 100) if total_recommendations > 0 else 0
        
        conn.close()
        
        return {
            'total_recommendations': total_recommendations,  # Unique games recommended
            'total_clicks': total_clicks,  # Unique games clicked
            'click_rate': round(click_rate, 2),  # Percentage
            'total_sessions': total_sessions  # Number of recommendation sessions
        }
    
    def get_user_history_summary(self, user_id):
        """Get summary of user activity"""
        ratings = self.get_user_ratings(user_id)
        stats = self.get_recommendation_stats(user_id)
        
        return {
            'total_ratings': len(ratings),
            'avg_rating': ratings['rating'].mean() if len(ratings) > 0 else 0,
            'recommendation_stats': stats
        }

