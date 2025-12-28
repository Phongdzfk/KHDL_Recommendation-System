"""
Streamlit App for Steam Games Recommendation System
Demo app based on khdl-game.ipynb notebook
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.database import UserHistoryDB

# Page configuration
st.set_page_config(
    page_title="Steam Games Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for beautiful design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #667eea !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(102, 126, 234, 0.2);
    }
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: white !important;
    }
    .game-card * {
        color: white !important;
    }
    .game-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #1f2937;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.05);
        color: white !important;
    }
    /* Ensure all text in game cards is white */
    .game-card p, .game-card strong {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'games_df' not in st.session_state:
    st.session_state.games_df = None
if 'db' not in st.session_state:
    st.session_state.db = UserHistoryDB()
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None

# HybridRecommendationSystem class (from notebook)
class HybridRecommendationSystem:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.games_df = None
        self.game_id_to_index = {}
        self.index_to_game_id = {}
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self.content_model = None
        self.collab_model = None
        self.tfidf_vectorizer = None
        self.feature_vectors = None
        self.item_user_matrix = None
        self.user_item_matrix_transposed = None
    
    def load_from_pickle(self, model_data):
        """Load model from pickle data"""
        self.alpha = model_data.get('alpha', 0.5)
        self.games_df = model_data['games_df']
        self.game_id_to_index = model_data['game_id_to_index']
        self.index_to_game_id = model_data['index_to_game_id']
        self.user_id_to_index = model_data.get('user_id_to_index', {})
        self.index_to_user_id = model_data.get('index_to_user_id', {})
        self.content_model = model_data['content_model']
        self.collab_model = model_data['collab_model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.feature_vectors = model_data['feature_vectors']
        self.item_user_matrix = model_data['item_user_matrix']
        
        # Reconstruct user_item_matrix_transposed if not present
        if 'user_item_matrix_transposed' in model_data:
            self.user_item_matrix_transposed = model_data['user_item_matrix_transposed']
        else:
            self.user_item_matrix_transposed = self.item_user_matrix.T.tocsr()
    
    def get_game_info(self, game_id):
        """Get game information by ID"""
        game_info = self.games_df[self.games_df['game_id'] == game_id]
        if len(game_info) > 0:
            return game_info.iloc[0].to_dict()
        return None
    
    def get_user_ratings_from_model(self, user_id):
        """Get user ratings from model's user-item matrix"""
        if user_id not in self.user_id_to_index:
            return {}
        
        user_idx = self.user_id_to_index[user_id]
        user_row = self.user_item_matrix_transposed[user_idx]
        
        # Get non-zero ratings from sparse matrix
        ratings_dict = {}
        for game_idx, rating in zip(user_row.indices, user_row.data):
            if rating > 0:
                game_id = self.index_to_game_id[game_idx]
                ratings_dict[game_id] = float(rating)
        
        return ratings_dict
    
    def recommend_by_game(self, game_id, top_n=10):
        """Recommend games similar to a given game"""
        if game_id not in self.game_id_to_index:
            return []
        
        game_idx = self.game_id_to_index[game_id]
        
        # Get candidates from Content
        dists_cont, indices_cont = self.content_model.kneighbors(
            self.feature_vectors[game_idx], n_neighbors=top_n+1)
        
        # Get candidates from Collaborative
        dists_collab, indices_collab = self.collab_model.kneighbors(
            self.item_user_matrix[game_idx], n_neighbors=top_n+1)
        
        # Combine Scores
        scores = {}
        
        # Process Content Recommendations
        for i, neighbor_idx in enumerate(indices_cont[0]):
            if neighbor_idx == game_idx:
                continue
            rec_id = self.index_to_game_id[neighbor_idx]
            sim_score = 1 - dists_cont[0][i]
            scores[rec_id] = scores.get(rec_id, 0) + (1 - self.alpha) * sim_score
        
        # Process Collaborative Recommendations
        for i, neighbor_idx in enumerate(indices_collab[0]):
            if neighbor_idx == game_idx:
                continue
            rec_id = self.index_to_game_id[neighbor_idx]
            sim_score = 1 - dists_collab[0][i]
            scores[rec_id] = scores.get(rec_id, 0) + self.alpha * sim_score
        
        # Sort and return
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]
    
    def recommend_by_user_realtime(self, user_ratings_dict, top_n=10):
        """Recommend games for a user based on their ratings (real-time, no user_id needed)"""
        if not user_ratings_dict:
            return []
        
        # Get game IDs from ratings
        rated_game_ids = list(user_ratings_dict.keys())
        rated_indices = [self.game_id_to_index[gid] for gid in rated_game_ids if gid in self.game_id_to_index]
        
        if not rated_indices:
            return []
        
        # Collect candidates
        candidate_scores = defaultdict(float)
        
        for game_id, rating in user_ratings_dict.items():
            if game_id not in self.game_id_to_index:
                continue
            
            game_idx = self.game_id_to_index[game_id]
            weight = (rating / 5.0) ** 2  # Exponential weight
            
            # Find Content neighbors
            d_cont, i_cont = self.content_model.kneighbors(
                self.feature_vectors[game_idx], n_neighbors=15)
            
            # Find Collab neighbors
            d_coll, i_coll = self.collab_model.kneighbors(
                self.item_user_matrix[game_idx], n_neighbors=15)
            
            # Add content-based candidates
            for dist, neighbor_idx in zip(d_cont[0], i_cont[0]):
                if neighbor_idx in rated_indices:
                    continue
                gid = self.index_to_game_id[neighbor_idx]
                sim = 1 - dist
                candidate_scores[gid] += sim * (1-self.alpha) * weight
            
            # Add collaborative candidates
            for dist, neighbor_idx in zip(d_coll[0], i_coll[0]):
                if neighbor_idx in rated_indices:
                    continue
                gid = self.index_to_game_id[neighbor_idx]
                sim = 1 - dist
                candidate_scores[gid] += sim * self.alpha * weight
        
        # Sort and return
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def recommend_by_user_with_filters(self, user_id, top_n=10,
                                       min_year=None,
                                       max_price=None,
                                       required_genres=None,
                                       exclude_genres=None,
                                       diversity_weight=0.0):
        """
        User recommendations with filters
        
        Parameters:
        - min_year: Only recommend games from this year onwards
        - max_price: Only recommend games under this price
        - required_genres: List of genres that must be present (e.g., ['Action', 'RPG'])
        - exclude_genres: List of genres to avoid (e.g., ['Casual', 'Puzzle'])
        - diversity_weight: Diversity penalty (0-1)
        """
        # Get base recommendations
        if user_id not in self.user_id_to_index:
            return []
        
        user_idx = self.user_id_to_index[user_id]
        user_row = self.user_item_matrix_transposed[user_idx]
        rated_indices = user_row.indices
        ratings = user_row.data
        
        # Get high-rated games
        high_rated_mask = ratings >= 3.5
        if np.sum(high_rated_mask) > 0:
            target_indices = rated_indices[high_rated_mask]
            target_ratings = ratings[high_rated_mask]
        else:
            target_indices = rated_indices
            target_ratings = ratings
        
        # Collect candidates
        candidate_scores = defaultdict(float)
        
        for idx, rating in zip(target_indices, target_ratings):
            weight = (rating / 5.0) ** 2
            game_id = self.index_to_game_id[idx]
            
            # Find Content neighbors
            d_cont, i_cont = self.content_model.kneighbors(
                self.feature_vectors[idx], n_neighbors=15)
            
            # Find Collab neighbors
            d_coll, i_coll = self.collab_model.kneighbors(
                self.item_user_matrix[idx], n_neighbors=15)
            
            # Add candidates
            for dist, neighbor_idx in zip(d_cont[0], i_cont[0]):
                if neighbor_idx in rated_indices:
                    continue
                gid = self.index_to_game_id[neighbor_idx]
                sim = 1 - dist
                candidate_scores[gid] += sim * (1-self.alpha) * weight
            
            for dist, neighbor_idx in zip(d_coll[0], i_coll[0]):
                if neighbor_idx in rated_indices:
                    continue
                gid = self.index_to_game_id[neighbor_idx]
                sim = 1 - dist
                candidate_scores[gid] += sim * self.alpha * weight
        
        base_recs = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 3]
        
        # Apply filters
        filtered_recs = []
        
        for game_id, score in base_recs:
            game_info = self.get_game_info(game_id)
            if not game_info:
                continue
            
            # Year filter
            if min_year and game_info.get('year', 0) < min_year:
                continue
            
            # Price filter
            if max_price is not None and game_info.get('price', float('inf')) > max_price:
                continue
            
            # Genre filters
            genres = str(game_info.get('genres', '')).lower()
            
            if required_genres:
                if not any(g.lower() in genres for g in required_genres):
                    continue
            
            if exclude_genres:
                if any(g.lower() in genres for g in exclude_genres):
                    continue
            
            filtered_recs.append((game_id, score))
            
            if len(filtered_recs) >= top_n:
                break
        
        return filtered_recs

@st.cache_resource(show_spinner=False, max_entries=1)
def load_model():
    """Load pre-trained model from pickle file or URL"""
    import sys
    import traceback
    
    model_path = Path('models/recommendation_model.pkl')
    
    # Model URL (Google Drive)
    MODEL_URL = "https://drive.google.com/uc?export=download&id=180S_9i5886cn9l9qKCeAmft0otJpEN64&confirm=t"
    
    # If model doesn't exist locally, download from URL
    if not model_path.exists():
        st.info("📥 Downloading model from cloud storage... This may take a few minutes.")
        try:
            import requests
            
            # Try using gdown first (better for Google Drive large files)
            use_gdown = False
            try:
                import gdown
                use_gdown = True
            except ImportError:
                pass
            
            if use_gdown:
                try:
                    file_id = "180S_9i5886cn9l9qKCeAmft0otJpEN64"
                    gdrive_url = f"https://drive.google.com/uc?id={file_id}"
                    
                    # Create models directory if not exists
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("📥 Downloading with gdown...")
                    
                    # Download with gdown (handles Google Drive better)
                    gdown.download(gdrive_url, str(model_path), quiet=False)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Validate downloaded file
                    if not model_path.exists():
                        raise FileNotFoundError("File was not downloaded")
                    
                    file_size = model_path.stat().st_size
                    if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
                        model_path.unlink()
                        raise ValueError(f"Downloaded file is too small ({file_size / (1024*1024):.1f} MB). Expected > 100MB.")
                    
                    # Check pickle header
                    with open(model_path, 'rb') as f:
                        first_bytes = f.read(10)
                        if not (first_bytes.startswith(b'\x80') or first_bytes.startswith(b'PK') or 
                                first_bytes[0] in [0x02, 0x03, 0x04, 0x05]):
                            f.seek(0)
                            first_100 = f.read(100)
                            if b'<html' in first_100.lower():
                                model_path.unlink()
                                raise ValueError("Downloaded HTML instead of pickle file")
                    
                    st.success(f"✅ Model downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
                except Exception as gdown_error:
                    # If gdown fails, try requests method
                    if model_path.exists():
                        model_path.unlink()
                    st.warning(f"⚠️ gdown failed: {gdown_error}. Trying requests method...")
                    use_gdown = False  # Fall back to requests
            
            # Fallback to requests if gdown not available or failed
            if not use_gdown:
                # Create session to handle Google Drive redirects and cookies
                session = requests.Session()
                
                # First request to get the actual download URL (handle virus scan warning)
                response = session.get(MODEL_URL, stream=True, timeout=600, allow_redirects=True)
                response.raise_for_status()
                
                # Check if we got HTML instead of binary file (Google Drive virus scan warning)
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    # Try to extract download link from HTML
                    import re
                    html_content = response.text
                    # Look for the actual download link in the HTML
                    match = re.search(r'href="(/uc\?export=download[^"]+)"', html_content)
                    if match:
                        # Construct full URL
                        download_url = "https://drive.google.com" + match.group(1)
                        response = session.get(download_url, stream=True, timeout=600, allow_redirects=True)
                        response.raise_for_status()
                    else:
                        # Try alternative method: use direct download with confirm parameter
                        file_id = "180S_9i5886cn9l9qKCeAmft0otJpEN64"
                        download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                        response = session.get(download_url, stream=True, timeout=600, allow_redirects=True)
                        response.raise_for_status()
                        # Check again if still HTML
                        if 'text/html' in response.headers.get('content-type', '').lower():
                            raise ValueError("Google Drive returned HTML page instead of file. File may be too large for direct download.")
                
                # Validate we got binary content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    # Read first bytes to check
                    first_chunk = next(response.iter_content(chunk_size=1024), b'')
                    if b'<html' in first_chunk.lower() or b'<!doctype' in first_chunk.lower():
                        raise ValueError("Downloaded file is HTML, not binary. Google Drive may require manual download.")
                    # Reset response for reading
                    response = session.get(response.url, stream=True, timeout=600)
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Create models directory if not exists
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192 * 10  # 80KB chunks for faster download
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress every MB
                            if total_size > 0 and downloaded % (1024 * 1024) == 0:
                                progress = min(downloaded / total_size, 1.0)
                                progress_bar.progress(progress)
                                status_text.text(f"📥 Downloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                
                progress_bar.empty()
                status_text.empty()
                
                # Validate downloaded file
                file_size = model_path.stat().st_size
                if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
                    model_path.unlink()  # Delete invalid file
                    raise ValueError(f"Downloaded file is too small ({file_size / (1024*1024):.1f} MB). Expected > 100MB. File may be corrupted or HTML page.")
                
                # Check if it's a valid pickle file
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(10)
                    # Pickle files typically start with protocol bytes (0x80, 0x02, etc.) or 'PK' for zip
                    if not (first_bytes.startswith(b'\x80') or first_bytes.startswith(b'PK') or 
                            first_bytes[0] in [0x02, 0x03, 0x04, 0x05]):
                        # Check if it's HTML
                        f.seek(0)
                        first_100 = f.read(100)
                        if b'<html' in first_100.lower() or b'<!doctype' in first_100.lower():
                            model_path.unlink()
                            raise ValueError("Downloaded file is HTML page, not pickle file. Google Drive may require manual confirmation.")
                        # If not HTML but also not pickle, warn but continue
                        st.warning(f"⚠️ File doesn't start with pickle markers. Size: {file_size / (1024*1024):.1f} MB")
                
                st.success(f"✅ Model downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # Clean up invalid file
            st.error(f"❌ Error downloading model: {e}")
            st.warning("💡 **Possible solutions:**")
            st.info("""
            1. **Check internet connection** on Streamlit Cloud
            2. **Verify Google Drive link** is accessible and file is shared publicly
            3. **File size:** Google Drive may block large file downloads. Try:
               - Use alternative: Dropbox, GitHub Releases, or AWS S3
               - Split model into smaller parts
            4. **Manual download:** Download file manually and upload to Streamlit Cloud secrets
            """)
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())
            return None, None
    
    if not model_path.exists():
        return None, None
    
    # Check if file is actually a pickle file (not a Git LFS pointer)
    try:
        with open(model_path, 'rb') as f:
            first_bytes = f.read(100)
            # Check if it's a Git LFS pointer file
            if b'version https://git-lfs.github.com' in first_bytes:
                st.error("❌ Model file is a Git LFS pointer, not the actual file!")
                st.warning("💡 **Git LFS file chưa được download đúng cách.**")
                st.info("""
                **Giải pháp:**
                1. Đảm bảo Git LFS đã được cài đặt trên Streamlit Cloud
                2. File cần được download từ Git LFS, không phải pointer
                3. Kiểm tra xem file có kích thước đúng không (nên > 100MB)
                """)
                return None, None
            # Check if it's a valid pickle file (starts with pickle protocol markers)
            if not (first_bytes.startswith(b'\x80') or first_bytes.startswith(b'PK') or b'pickle' in first_bytes[:20].lower()):
                st.error("❌ File không phải là pickle file hợp lệ!")
                st.warning(f"💡 File có thể bị corrupt hoặc không đúng định dạng.")
                return None, None
    except Exception as e:
        st.error(f"❌ Không thể đọc file: {e}")
        return None, None
    
    try:
        # Try loading with error handling for version mismatches
        with open(model_path, 'rb') as f:
            try:
                model_data = pickle.load(f)
            except (ModuleNotFoundError, AttributeError, ImportError) as e:
                # If numpy version mismatch, try with encoding
                f.seek(0)
                try:
                    # Try with latin1 encoding (handles numpy 1.x -> 2.x issues)
                    model_data = pickle.load(f, encoding='latin1')
                except Exception as e2:
                    # Try with errors='ignore'
                    f.seek(0)
                    try:
                        model_data = pickle.load(f, encoding='latin1', errors='ignore')
                    except:
                        # Last attempt: try loading with pickle5 if available
                        f.seek(0)
                        try:
                            import pickle5 as p5
                            model_data = p5.load(f)
                        except ImportError:
                            raise e  # Re-raise original error
            except (pickle.UnpicklingError, ValueError, EOFError) as e:
                # Handle corrupt file or invalid pickle
                st.error(f"❌ Lỗi khi load pickle file: {e}")
                st.warning("💡 **File có thể bị corrupt hoặc không đúng định dạng.**")
                st.info("""
                **Nguyên nhân có thể:**
                1. File bị corrupt khi upload/download
                2. Git LFS chưa download file đúng cách
                3. File không phải là pickle file hợp lệ
                
                **Giải pháp:**
                1. Kiểm tra file trên GitHub có kích thước đúng không
                2. Đảm bảo Git LFS đã download file đúng cách
                3. Thử push lại model file lên Git LFS
                """)
                import traceback
                with st.expander("🔍 Full Error Details"):
                    st.code(traceback.format_exc())
                return None, None
        
        # Create system and load data
        try:
            system = HybridRecommendationSystem()
            system.load_from_pickle(model_data)
            
            # Get games dataframe
            games_df = model_data['games_df']
            
            return system, games_df
        except Exception as load_error:
            # Log error but don't show in UI (will be handled in main)
            import sys
            print(f"Error in load_from_pickle: {load_error}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise load_error
    except Exception as e:
        # Log error to stderr for debugging
        import sys
        print(f"Error loading model: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Don't show error here - let main() handle it
        raise e

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_cached_recommendations(_model, user_ratings_dict, top_n):
    """Cache recommendations to improve performance"""
    # Create a hash key from sorted ratings
    # Note: _model has leading underscore so Streamlit doesn't try to hash it
    return _model.recommend_by_user_realtime(user_ratings_dict, top_n=top_n)

@st.cache_data(show_spinner=False)
def get_all_games_list(games_df):
    """Get all games for selectbox"""
    return games_df[['game_id', 'title_clean', 'year', 'genres']].copy()

@st.cache_data(show_spinner=False)
def get_game_search_results(games_df, search_query, limit=100):
    """Get game search results with autocomplete"""
    if not search_query or len(search_query) < 1:
        return games_df.head(limit)
    
    search_query_lower = search_query.lower()
    mask = games_df['title_clean'].str.lower().str.contains(search_query_lower, na=False)
    results = games_df[mask].head(limit)
    return results

def get_autocomplete_suggestions(games_df, search_query, limit=10):
    """Get autocomplete suggestions for search (realtime)"""
    if not search_query or len(search_query) < 1:
        return []
    
    search_query_lower = search_query.lower()
    mask = games_df['title_clean'].str.lower().str.contains(search_query_lower, na=False)
    suggestions = games_df[mask].head(limit)
    
    # Format suggestions as list of tuples (game_id, display_text)
    result = []
    for _, row in suggestions.iterrows():
        year = int(row['year']) if pd.notna(row['year']) else 'N/A'
        display_text = f"{row['title_clean']} ({year})"
        result.append((row['game_id'], display_text))
    
    return result

def display_game_card(game, key_prefix=""):
    """Display beautiful game card with all available information"""
    with st.container():
        title = game.get('title_clean', game.get('title', 'Unknown Game'))
        year = int(game.get('year', 0)) if pd.notna(game.get('year')) else 'N/A'
        genres = game.get('genres', 'Unknown')
        price = game.get('price', 0)
        
        # Build info string with all available fields
        info_parts = []
        info_parts.append(f"<p><strong>📅 Year:</strong> {year}</p>")
        info_parts.append(f"<p><strong>🎭 Genres:</strong> {genres[:100]}{'...' if len(str(genres)) > 100 else ''}</p>")
        
        # Add description if available
        if 'description' in game and pd.notna(game.get('description')) and str(game.get('description')).strip():
            desc = str(game.get('description'))[:150]
            info_parts.append(f"<p><strong>📝 Description:</strong> {desc}{'...' if len(str(game.get('description'))) > 150 else ''}</p>")
        
        # Add price if available
        if price and price > 0:
            info_parts.append(f'<p><strong>💰 Price:</strong> ${price:.2f}</p>')
        
        info_html = ''.join(info_parts)
        
        st.markdown(f"""
        <div class="game-card">
            <div class="game-title">🎮 {title}</div>
            {info_html}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Beautiful header
    st.markdown('<h1 class="main-header">🎮 Steam Games Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Auto load model (with better error handling)
    if not st.session_state.model_loaded:
        try:
            with st.spinner("🔄 Loading model... This may take a moment."):
                system, games_df = load_model()
                if system is not None and games_df is not None:
                    st.session_state.model_data = system
                    st.session_state.games_df = games_df
                    st.session_state.model_loaded = True
                else:
                    st.error("❌ Model not found! Please place `models/recommendation_model.pkl`")
                    st.info("""
                    **Instructions:**
                    1. Make sure `recommendation_model.pkl` is in the `models/` folder
                    2. Refresh this page
                    """)
                    st.stop()
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.warning("💡 **Possible issues:**")
            st.info("""
            **Common causes:**
            1. **Memory limit:** Model is too large for Streamlit Cloud (1GB limit)
            2. **Timeout:** Loading model takes too long
            3. **Corrupted file:** Model file may be corrupted
            4. **Version mismatch:** Package versions don't match
            
            **Solutions:**
            1. Check Streamlit Cloud logs for detailed error
            2. Try reducing model size or using smaller model
            3. Verify model file integrity
            4. Check package versions in requirements.txt
            """)
            import traceback
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Sidebar - User Management
    with st.sidebar:
        st.header("👤 User Management")
        
        # User type selection
        user_type = st.radio(
            "User Type:",
            ["New User", "Existing User", "Test User (from Model)"],
            horizontal=False
        )
        
        if user_type == "New User":
            # Create new user
            if st.button("➕ Create New User", use_container_width=True):
                user_id = st.session_state.db.create_new_user()
                st.session_state.current_user_id = user_id
                st.rerun()
        
        elif user_type == "Existing User":
            # Select from database users
            all_ratings = st.session_state.db.get_all_user_ratings()
            if len(all_ratings) > 0:
                unique_users = sorted(all_ratings['user_id'].unique().tolist())
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_user = st.selectbox(
                        "Select User:",
                        options=unique_users,
                        index=0 if st.session_state.current_user_id is None else 
                              (unique_users.index(st.session_state.current_user_id) if st.session_state.current_user_id in unique_users else 0),
                        key="select_existing_user"
                    )
                with col2:
                    if st.button("🗑️", help="Delete user", key="delete_user_btn"):
                        if st.session_state.db.delete_user(selected_user):
                            st.success("✅ User deleted!")
                            if st.session_state.current_user_id == selected_user:
                                st.session_state.current_user_id = None
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete user")
                
                st.session_state.current_user_id = selected_user
            else:
                st.info("No users in database")
        
        else:  # Test User from Model
            if st.session_state.model_data:
                # Get sample users from model
                model_users = list(st.session_state.model_data.user_id_to_index.keys())[:1000]  # Limit to 1000 for performance
                test_user = st.selectbox(
                    "Select Test User:",
                    options=model_users,
                    format_func=lambda x: f"User {x}",
                    key="select_test_user"
                )
                test_user_id = f"test_{test_user}"  # Prefix to distinguish
                
                # Import ratings from model to database if not already imported
                existing_ratings = st.session_state.db.get_user_ratings(test_user_id)
                if len(existing_ratings) == 0:
                    # Get ratings from model
                    model_ratings = st.session_state.model_data.get_user_ratings_from_model(test_user)
                    if model_ratings:
                        # Create user if not exists
                        st.session_state.db.create_new_user(test_user_id)
                        # Import ratings
                        for game_id, rating in model_ratings.items():
                            st.session_state.db.add_rating(test_user_id, game_id, rating)
                        st.success(f"✅ Imported {len(model_ratings)} ratings from model!")
                        st.rerun()  # Rerun to refresh UI
                
                st.session_state.current_user_id = test_user_id
                st.info("💡 Test users use model's training data")
            else:
                st.warning("Model not loaded")
        
        if st.session_state.current_user_id:
            # Check if test user
            if str(st.session_state.current_user_id).startswith("test_"):
                actual_user_id = int(st.session_state.current_user_id.replace("test_", ""))
                st.info(f"👤 **Test User:** {actual_user_id}")
            else:
                st.info(f"👤 **User:** {st.session_state.current_user_id}")
                
                # Quick stats
                user_ratings = st.session_state.db.get_user_ratings(st.session_state.current_user_id)
                if len(user_ratings) > 0:
                    st.metric("📊 Games Rated", len(user_ratings))
                    st.metric("⭐ Avg Rating", f"{user_ratings['rating'].mean():.2f}")
    
    # Main content
    if st.session_state.current_user_id is None:
        st.warning("⚠️ Please create or select a user to start!")
        st.info("💡 Use the sidebar to create a new user or select an existing one.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Game Recommendations", "🔍 Search Games", "📊 History", "ℹ️ Model Info"])
        
        with tab1:
            st.header("🎯 Get Game Recommendations")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                rec_mode = st.radio(
                    "Recommendation Mode:",
                    ["Game-Based (Find Similar Games)", "User-Based (Rate Games First)"],
                    horizontal=True
                )
            
            with col2:
                num_recs = st.slider("Number of Recommendations:", 5, 20, 10)
            
            if rec_mode == "Game-Based (Find Similar Games)":
                st.subheader("🔍 Find a game to see similar recommendations")
                
                # Search with autocomplete
                col_search, col_btn = st.columns([4, 1])
                with col_search:
                    game_search = st.text_input(
                        "🔍 Search game (type to filter):",
                        placeholder="Start typing game name...",
                        key="game_search_input"
                    )
                    
                # Get search results
                if game_search:
                    search_results = get_game_search_results(st.session_state.games_df, game_search, limit=50)
                else:
                    search_results = st.session_state.games_df.head(50)
                
                if len(search_results) > 0:
                    # Display selectbox with search results
                    game_options = search_results['game_id'].tolist()
                    selected_game_id = st.selectbox(
                        f"📋 Select a game ({len(game_options)} found):",
                        options=game_options,
                        format_func=lambda x: f"{search_results[search_results['game_id'] == x]['title_clean'].iloc[0]} ({int(search_results[search_results['game_id'] == x]['year'].iloc[0]) if pd.notna(search_results[search_results['game_id'] == x]['year'].iloc[0]) else 'N/A'})",
                        key="select_game_for_rec"
                    )
                    
                    with col_btn:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                    
                    if st.button("🎮 Find Similar Games", use_container_width=True, type="primary", key="find_similar_btn"):
                        game_recs = st.session_state.model_data.recommend_by_game(
                            selected_game_id, top_n=num_recs
                        )
                        st.session_state[f'game_recs_{selected_game_id}'] = game_recs
                        st.session_state['selected_game_for_rec'] = selected_game_id
                        st.rerun()
                    
                    # Display recommendations if available
                    if f'game_recs_{st.session_state.get("selected_game_for_rec", None)}' in st.session_state:
                        recommendations = st.session_state[f'game_recs_{st.session_state.get("selected_game_for_rec", None)}']
                        selected_game_id = st.session_state.get('selected_game_for_rec')
                        
                        if len(recommendations) > 0:
                            # Get selected game info
                            selected_game = st.session_state.model_data.get_game_info(selected_game_id)
                            if selected_game:
                                st.subheader(f"🎮 Similar to: **{selected_game.get('title_clean', 'Unknown')}**")
                                st.markdown(f"**Genres:** {selected_game.get('genres', 'N/A')}")
                                st.markdown("---")
                            
                            st.subheader(f"🎯 Top {len(recommendations)} Similar Games:")
                            
                        # Display recommendations
                        cols = st.columns(3)
                        for idx, (game_id, score) in enumerate(recommendations):
                            game_info = st.session_state.model_data.get_game_info(game_id)
                            if game_info:
                                with cols[idx % 3]:
                                    display_game_card(game_info, key_prefix=f"rec_{idx}_")
                                    st.caption(f"Similarity Score: {score:.4f}")
                                    
                                    # Log recommendation viewed (only once per session)
                                    rec_key = f'game_rec_logged_{game_id}_{st.session_state.current_user_id}'
                                    if rec_key not in st.session_state and st.session_state.current_user_id:
                                        st.session_state.db.log_recommendation(
                                            st.session_state.current_user_id,
                                            game_id,
                                            recommendation_type='game-based',
                                            clicked=0
                                        )
                                        st.session_state[rec_key] = True
                                    
                                    # View Details button - track click
                                    if st.session_state.current_user_id:
                                        if st.button("👁️ View Details", key=f"game_view_{game_id}_{idx}", use_container_width=True):
                                            # Update clicked status
                                            st.session_state.db.update_recommendation_clicked(
                                                st.session_state.current_user_id,
                                                game_id,
                                                recommendation_type='game-based'
                                            )
                                            st.session_state[f'show_game_details_{game_id}'] = True
                                            st.rerun()
                                        
                                        # Show details if clicked
                                        if st.session_state.get(f'show_game_details_{game_id}', False):
                                            # Hide/Show toggle button
                                            col_toggle, col_close = st.columns([3, 1])
                                            with col_toggle:
                                                if st.button("👁️ Hide Details", key=f"hide_game_{game_id}", use_container_width=True):
                                                    st.session_state[f'show_game_details_{game_id}'] = False
                                                    st.rerun()
                                            
                                            with st.expander(f"📋 Details: {game_info.get('title_clean', 'Unknown')}", expanded=True):
                                                st.write(f"**🎮 Title:** {game_info.get('title_clean', 'Unknown')}")
                                                st.write(f"**📅 Year:** {int(game_info.get('year', 0)) if pd.notna(game_info.get('year')) else 'N/A'}")
                                                st.write(f"**🎭 Genres:** {game_info.get('genres', 'N/A')}")
                                                
                                                # Display all available game information
                                                if 'description' in game_info and pd.notna(game_info.get('description')) and str(game_info.get('description')).strip():
                                                    st.write(f"**📝 Description:** {game_info.get('description', 'N/A')[:200]}{'...' if len(str(game_info.get('description', ''))) > 200 else ''}")
                                                if 'price' in game_info and game_info.get('price', 0) > 0:
                                                    st.write(f"**💰 Price:** ${game_info.get('price', 0):.2f}")
                                                
                                                # Display any other available fields
                                                common_fields = ['title', 'title_clean', 'year', 'genres', 'price', 'description', 'all_tags', 'game_id']
                                                for key, value in game_info.items():
                                                    if key not in common_fields and pd.notna(value) and str(value).strip():
                                                        # Format field name nicely
                                                        field_name = key.replace('_', ' ').title()
                                                        st.write(f"**{field_name}:** {value}")
                                                
                                                st.write(f"**🆔 Game ID:** {game_id}")
                                                st.write(f"**⭐ Similarity Score:** {score:.4f}")
                                                
                                                # Rating option
                                                existing_rating = st.session_state.db.get_user_rating_for_game(
                                                    st.session_state.current_user_id, game_id
                                                )
                                                if existing_rating:
                                                    st.info(f"⭐ Your Rating: {existing_rating}/5")
                                                else:
                                                    user_rating = st.slider(
                                                        "Rate this game:", 1, 5, 3,
                                                        key=f"game_detail_rating_{game_id}"
                                                    )
                                                    if st.button("💾 Save Rating", key=f"save_game_detail_{game_id}"):
                                                        st.session_state.db.add_rating(
                                                            st.session_state.current_user_id,
                                                            game_id,
                                                            user_rating
                                                        )
                                                        st.success("✅ Rating saved!")
                                                        st.rerun()
                    else:
                        st.info("💡 Select a game above and click 'Find Similar Games' to see recommendations")
                else:
                    st.warning("❌ No games found! Try a different search term.")
        
            else:  # User-Based
                # Check if test user
                is_test_user = str(st.session_state.current_user_id).startswith("test_")
                recommendations = []  # Initialize
                
                if is_test_user:
                    # Test user - use model's user_id directly
                    actual_user_id = int(st.session_state.current_user_id.replace("test_", ""))
                    st.subheader("🎯 Get Recommendations for Test User")
                    st.info(f"Using test user {actual_user_id} from model data")
                    
                    # Filters
                    with st.expander("🔧 Filters", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            min_year = st.number_input("Min Year:", min_value=1980, max_value=2024, value=None, step=1)
                            max_price = st.number_input("Max Price ($):", min_value=0.0, max_value=1000.0, value=None, step=1.0)
                        with col2:
                            # Get unique genres for selectbox
                            all_genres = st.session_state.games_df['genres'].str.split(',').explode().str.strip()
                            unique_genres = sorted([g for g in all_genres.unique() if g and g != 'Unknown'])[:30]
                            
                            required_genres_list = st.multiselect(
                                "Required Genres:",
                                options=unique_genres,
                                key="test_required_genres"
                            )
                            exclude_genres_list = st.multiselect(
                                "Exclude Genres:",
                                options=unique_genres,
                                key="test_exclude_genres"
                            )
                    
                    if st.button("🎮 Get Recommendations", use_container_width=True, type="primary"):
                        recommendations = st.session_state.model_data.recommend_by_user_with_filters(
                            actual_user_id,
                            top_n=num_recs,
                            min_year=min_year,
                            max_price=max_price,
                            required_genres=required_genres_list if required_genres_list else None,
                            exclude_genres=exclude_genres_list if exclude_genres_list else None
                        )
                        
                        # Save recommendations to session state for test user
                        st.session_state[f'test_user_recs_{actual_user_id}'] = recommendations
                        
                        # Log recommendations for test user
                        for game_id, score in recommendations:
                            st.session_state.db.log_recommendation(
                                st.session_state.current_user_id,
                                game_id,
                                recommendation_type='test-user',
                                clicked=0
                            )
                        
                        st.rerun()
                
                # Display saved recommendations for test user
                if f'test_user_recs_{actual_user_id}' in st.session_state:
                    recommendations = st.session_state[f'test_user_recs_{actual_user_id}']
                else:
                    # Regular user - rate games first
                    st.subheader("💡 Rate some games to get personalized recommendations")
                    
                    # Game rating interface
                    st.markdown("### Rate Games")
                    
                    # Optimized search with filters
                    col_search, col_filter = st.columns([3, 1])
                    
                    with col_search:
                        rate_search = st.text_input(
                            "🔍 Search games to rate (type to filter):",
                            placeholder="Start typing game name...",
                            key="rate_search_input"
                        )
                        
                    with col_filter:
                        # Get unique genres
                        all_genres = st.session_state.games_df['genres'].str.split(',').explode().str.strip()
                        unique_genres = sorted([g for g in all_genres.unique() if g and g != 'Unknown'])[:50]
                        genre_filter = st.selectbox(
                            "🎭 Genre:",
                            options=["All"] + unique_genres,
                            key="genre_filter"
                        )
                    
                    # Get search results
                    if rate_search:
                        search_results = get_game_search_results(st.session_state.games_df, rate_search, limit=50)
                    else:
                        search_results = st.session_state.games_df.head(50)
                    
                    # Apply genre filter
                    if genre_filter != "All":
                        search_results = search_results[search_results['genres'].str.contains(genre_filter, case=False, na=False)]
                    
                    if len(search_results) > 0:
                        st.success(f"✅ Found **{len(search_results)}** games")
                        
                        # Display selectbox to choose game
                        selected_game_to_rate = st.selectbox(
                            f"📋 Select game to rate ({len(search_results)} found):",
                            options=search_results['game_id'].tolist(),
                            format_func=lambda x: f"{search_results[search_results['game_id'] == x]['title_clean'].iloc[0]} ({int(search_results[search_results['game_id'] == x]['year'].iloc[0]) if pd.notna(search_results[search_results['game_id'] == x]['year'].iloc[0]) else 'N/A'})",
                            key="select_game_to_rate"
                        )
                        
                        # Get game info
                        game_info = st.session_state.model_data.get_game_info(selected_game_to_rate)
                        if game_info:
                            col_info, col_rate = st.columns([2, 1])
                            with col_info:
                                st.write(f"**🎮 {game_info.get('title_clean', 'Unknown')}**")
                                st.write(f"**Genres:** {game_info.get('genres', 'N/A')}")
                                st.write(f"**Year:** {int(game_info.get('year', 0)) if pd.notna(game_info.get('year')) else 'N/A'}")
                            
                            with col_rate:
                                # Get existing rating
                                existing_rating = st.session_state.db.get_user_rating_for_game(
                                    st.session_state.current_user_id, selected_game_to_rate
                                )
                                
                                user_rating = st.slider(
                                    "Your Rating:",
                                    1, 5, int(existing_rating) if existing_rating else 3,
                                    key=f"rating_{selected_game_to_rate}"
                                )
                                
                                if st.button("💾 Save Rating", key=f"save_{selected_game_to_rate}", use_container_width=True):
                                    st.session_state.db.add_rating(
                                        st.session_state.current_user_id,
                                        selected_game_to_rate,
                                        user_rating
                                    )
                                    st.success("✅ Rating saved!")
                                    st.rerun()
                                
                                if existing_rating:
                                    st.info(f"⭐ Current: {existing_rating}/5")
                    else:
                        st.warning("❌ No games found! Try different search terms or filters.")
                    
                    # Get recommendations based on ratings
                    st.markdown("---")
                    st.subheader("🎯 Get Recommendations")
                    
                    # Filters for recommendations
                    with st.expander("🔧 Recommendation Filters", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            min_year = st.number_input("Min Year:", min_value=1980, max_value=2024, value=None, step=1, key="rec_min_year")
                            max_price = st.number_input("Max Price ($):", min_value=0.0, max_value=1000.0, value=None, step=1.0, key="rec_max_price")
                        with col2:
                            # Get unique genres for selectbox
                            all_genres = st.session_state.games_df['genres'].str.split(',').explode().str.strip()
                            unique_genres = sorted([g for g in all_genres.unique() if g and g != 'Unknown'])[:30]
                            
                            required_genres_list = st.multiselect(
                                "Required Genres:",
                                options=unique_genres,
                                key="rec_required_genres"
                            )
                            exclude_genres_list = st.multiselect(
                                "Exclude Genres:",
                                options=unique_genres,
                                key="rec_exclude_genres"
                            )
                    
                    if st.button("🎮 Get Recommendations Based on My Ratings", use_container_width=True, type="primary"):
                        # Get ratings from database
                        user_ratings_df = st.session_state.db.get_user_ratings(st.session_state.current_user_id)
                        
                        if len(user_ratings_df) == 0:
                            st.info("💡 Please rate at least one game above to get recommendations!")
                        else:
                            # Convert to dict format
                            user_ratings_dict = dict(zip(user_ratings_df['game_id'], user_ratings_df['rating']))
                            
                            # Use multiselect values directly
                            required_genres = required_genres_list if required_genres_list else None
                            exclude_genres = exclude_genres_list if exclude_genres_list else None
                            
                            # Use cached recommendations for better performance
                            recommendations = get_cached_recommendations(
                                st.session_state.model_data,
                                user_ratings_dict,
                                num_recs * 3  # Get more to account for filtering
                            )
                            
                            # Apply filters manually if needed
                            if min_year or max_price or required_genres or exclude_genres:
                                filtered_recs = []
                                for game_id, score in recommendations:
                                    game_info = st.session_state.model_data.get_game_info(game_id)
                                    if not game_info:
                                        continue
                                    
                                    if min_year and game_info.get('year', 0) < min_year:
                                        continue
                                    if max_price is not None and game_info.get('price', float('inf')) > max_price:
                                        continue
                                    
                                    genres = str(game_info.get('genres', '')).lower()
                                    if required_genres and not any(g.lower() in genres for g in required_genres):
                                        continue
                                    if exclude_genres and any(g.lower() in genres for g in exclude_genres):
                                        continue
                                    
                                    filtered_recs.append((game_id, score))
                                    if len(filtered_recs) >= num_recs:
                                        break
                                
                                recommendations = filtered_recs
                                st.session_state[f'user_recs_{st.session_state.current_user_id}'] = recommendations
                                st.rerun()
                
                # Get recommendations from session state if available
                if is_test_user:
                    if f'test_user_recs_{actual_user_id}' in st.session_state:
                        recommendations = st.session_state[f'test_user_recs_{actual_user_id}']
                    else:
                        recommendations = []
                else:
                    if f'user_recs_{st.session_state.current_user_id}' in st.session_state:
                        recommendations = st.session_state[f'user_recs_{st.session_state.current_user_id}']
                    else:
                        recommendations = []
                
                # Display recommendations (for both test user and regular user)
                if len(recommendations) > 0:
                    if is_test_user:
                        st.success(f"✅ Found {len(recommendations)} recommendations for test user!")
                    else:
                        user_ratings_df = st.session_state.db.get_user_ratings(st.session_state.current_user_id)
                        st.success(f"✅ Found {len(recommendations)} recommendations based on your {len(user_ratings_df)} ratings!")
                    
                    st.subheader("🎯 Recommended Games:")
                    
                    # Display recommendations
                    cols = st.columns(3)
                    for idx, (game_id, score) in enumerate(recommendations):
                        game_info = st.session_state.model_data.get_game_info(game_id)
                        if game_info:
                            with cols[idx % 3]:
                                display_game_card(game_info, key_prefix=f"user_rec_{idx}_")
                                st.caption(f"Score: {score:.4f}")
                                
                                # Log recommendation viewed (only once per session)
                                rec_key = f'rec_logged_{game_id}_{st.session_state.current_user_id}'
                                if rec_key not in st.session_state:
                                    if is_test_user:
                                        st.session_state.db.log_recommendation(
                                            st.session_state.current_user_id,
                                            game_id,
                                            recommendation_type='test-user',
                                            clicked=0
                                        )
                                    else:
                                        st.session_state.db.log_recommendation(
                                            st.session_state.current_user_id,
                                            game_id,
                                            recommendation_type='user-based',
                                            clicked=0
                                        )
                                    st.session_state[rec_key] = True
                                
                                # View Details button - track click
                                if st.button("👁️ View Details", key=f"view_{game_id}_{idx}", use_container_width=True):
                                    # Update clicked status
                                    if is_test_user:
                                        st.session_state.db.update_recommendation_clicked(
                                            st.session_state.current_user_id,
                                            game_id,
                                            recommendation_type='test-user'
                                        )
                                    else:
                                        st.session_state.db.update_recommendation_clicked(
                                            st.session_state.current_user_id,
                                            game_id,
                                            recommendation_type='user-based'
                                        )
                                    
                                    # Show game details in expander
                                    st.session_state[f'show_details_{game_id}'] = True
                                    st.rerun()
                                
                                # Show details if clicked
                                if st.session_state.get(f'show_details_{game_id}', False):
                                    # Hide/Show toggle button
                                    if st.button("👁️ Hide Details", key=f"hide_{game_id}", use_container_width=True):
                                        st.session_state[f'show_details_{game_id}'] = False
                                        st.rerun()
                                    
                                    with st.expander(f"📋 Details: {game_info.get('title_clean', 'Unknown')}", expanded=True):
                                        st.write(f"**🎮 Title:** {game_info.get('title_clean', 'Unknown')}")
                                        st.write(f"**📅 Year:** {int(game_info.get('year', 0)) if pd.notna(game_info.get('year')) else 'N/A'}")
                                        st.write(f"**🎭 Genres:** {game_info.get('genres', 'N/A')}")
                                        
                                        # Display all available game information
                                        if 'description' in game_info and pd.notna(game_info.get('description')) and str(game_info.get('description')).strip():
                                            st.write(f"**📝 Description:** {game_info.get('description', 'N/A')[:200]}{'...' if len(str(game_info.get('description', ''))) > 200 else ''}")
                                        if 'price' in game_info and game_info.get('price', 0) > 0:
                                            st.write(f"**💰 Price:** ${game_info.get('price', 0):.2f}")
                                        
                                        # Display any other available fields
                                        common_fields = ['title', 'title_clean', 'year', 'genres', 'price', 'description', 'all_tags', 'game_id']
                                        for key, value in game_info.items():
                                            if key not in common_fields and pd.notna(value) and str(value).strip():
                                                # Format field name nicely
                                                field_name = key.replace('_', ' ').title()
                                                st.write(f"**{field_name}:** {value}")
                                        
                                        st.write(f"**🆔 Game ID:** {game_id}")
                                        st.write(f"**⭐ Recommendation Score:** {score:.4f}")
                                        
                                        # Rating option for regular users
                                        if not is_test_user:
                                            existing_rating = st.session_state.db.get_user_rating_for_game(
                                                st.session_state.current_user_id, game_id
                                            )
                                            if existing_rating:
                                                st.info(f"⭐ Your Rating: {existing_rating}/5")
                                            else:
                                                user_rating = st.slider(
                                                    "Rate this game:", 1, 5, 3,
                                                    key=f"detail_rating_{game_id}"
                                                )
                                                if st.button("💾 Save Rating", key=f"save_detail_{game_id}"):
                                                    st.session_state.db.add_rating(
                                                        st.session_state.current_user_id,
                                                        game_id,
                                                        user_rating
                                                    )
                                                    st.success("✅ Rating saved!")
                                                    st.rerun()
                                
                                # Quick rating button (for regular users only)
                                if not is_test_user:
                                    existing_rating = st.session_state.db.get_user_rating_for_game(
                                        st.session_state.current_user_id, game_id
                                    )
                                    if not existing_rating:
                                        if st.button("⭐ Quick Rate", key=f"quick_rate_{game_id}", use_container_width=True):
                                            st.session_state.db.add_rating(
                                                st.session_state.current_user_id,
                                                game_id,
                                                4  # Default rating
                                            )
                                            st.session_state.db.update_recommendation_clicked(
                                                st.session_state.current_user_id,
                                                game_id,
                                                recommendation_type='user-based'
                                            )
                                            st.rerun()
                else:
                    if is_test_user:
                        st.info("💡 Click 'Get Recommendations' button above to see recommendations!")
                    else:
                        st.info("💡 Rate some games above, then click 'Get Recommendations Based on My Ratings'!")
        
        with tab2:
            st.header("🔍 Search Games Database")
            
            # Optimized search with filters
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "🔍 Search game (type to filter):",
                    placeholder="Start typing game name...",
                    key="search_tab_input"
                )
            
            with col2:
                # Get unique genres
                all_genres = st.session_state.games_df['genres'].str.split(',').explode().str.strip()
                unique_genres = sorted([g for g in all_genres.unique() if g and g != 'Unknown'])[:50]
                genre_filter = st.selectbox(
                    "🎭 Genre:",
                    options=["All"] + unique_genres,
                    key="search_genre_filter"
                )
            
            with col3:
                # Get unique years
                unique_years = sorted(st.session_state.games_df['year'].dropna().unique().astype(int).tolist(), reverse=True)[:30]
                year_filter = st.selectbox(
                    "📅 Year:",
                    options=["All"] + unique_years,
                    key="search_year_filter"
                )
            
            # Get search results
            if search_query:
                results = get_game_search_results(st.session_state.games_df, search_query, limit=100)
            else:
                results = st.session_state.games_df.head(100)
            
            # Apply filters
            if genre_filter != "All":
                results = results[results['genres'].str.contains(genre_filter, case=False, na=False)]
            
            if year_filter != "All":
                results = results[results['year'] == int(year_filter)]
            
            # Log search
            if st.session_state.current_user_id:
                search_text = search_query or f"Filter: {genre_filter}, {year_filter}"
                st.session_state.db.log_search(
                    st.session_state.current_user_id,
                    search_text,
                    len(results)
                )
            
            if len(results) > 0:
                st.success(f"✅ Found **{len(results)}** games")
                
                # Display selectbox to choose game
                selected_game_id = st.selectbox(
                    f"📋 Select game to view ({len(results)} found):",
                    options=results['game_id'].tolist(),
                    format_func=lambda x: f"{results[results['game_id'] == x]['title_clean'].iloc[0]} ({int(results[results['game_id'] == x]['year'].iloc[0]) if pd.notna(results[results['game_id'] == x]['year'].iloc[0]) else 'N/A'})",
                    key="select_game_search"
                )
                
                # Display selected game details
                selected_game = st.session_state.model_data.get_game_info(selected_game_id)
                if selected_game:
                    col_info, col_action = st.columns([2, 1])
                    with col_info:
                        st.subheader(f"🎮 {selected_game.get('title_clean', 'Unknown')}")
                        st.write(f"**📅 Year:** {int(selected_game.get('year', 0)) if pd.notna(selected_game.get('year')) else 'N/A'}")
                        if 'genres' in selected_game:
                            st.write(f"**🎭 Genres:** {selected_game['genres']}")
                        # Display all fields that have values
                        common_fields = ['title', 'title_clean', 'year', 'genres', 'game_id']
                        displayed_fields = set(common_fields)
                        
                        # Check and display description
                        if 'description' in selected_game:
                            desc_value = selected_game.get('description')
                            if pd.notna(desc_value) and str(desc_value).strip() and str(desc_value).strip().lower() != 'nan':
                                st.write(f"**📝 Description:** {desc_value[:300]}{'...' if len(str(desc_value)) > 300 else ''}")
                                displayed_fields.add('description')
                        
                        # Check and display price
                        if 'price' in selected_game:
                            price_value = selected_game.get('price', 0)
                            if price_value and price_value > 0:
                                st.write(f"**💰 Price:** ${price_value:.2f}")
                                displayed_fields.add('price')
                        
                        # Display any other available fields
                        for key, value in selected_game.items():
                            if key not in displayed_fields:
                                if pd.notna(value) and str(value).strip() and str(value).strip().lower() != 'nan':
                                    # Format field name nicely
                                    field_name = key.replace('_', ' ').title()
                                    st.write(f"**{field_name}:** {value}")
                        
                        st.write(f"**🆔 Game ID:** {selected_game_id}")
                    
                    with col_action:
                        display_game_card(selected_game)
                        
                        # Rating for users
                        if st.session_state.current_user_id:
                            existing_rating = st.session_state.db.get_user_rating_for_game(
                                st.session_state.current_user_id, selected_game_id
                            )
                            if existing_rating:
                                st.info(f"⭐ Rated: {existing_rating}/5")
                            else:
                                user_rating = st.slider(
                                    "Rate:", 1, 5, 3,
                                    key=f"search_rating_{selected_game_id}"
                                )
                                if st.button("💾 Save Rating", key=f"search_save_{selected_game_id}", use_container_width=True):
                                    st.session_state.db.add_rating(
                                        st.session_state.current_user_id,
                                        selected_game_id,
                                        user_rating
                                    )
                                    st.success("✅ Rating saved!")
                                    st.rerun()
            else:
                st.warning("❌ No games found! Try different search terms or filters.")
        
        with tab3:
            st.header("📊 History & Statistics")
            
            user_ratings = st.session_state.db.get_user_ratings(st.session_state.current_user_id)
            
            if len(user_ratings) > 0:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Games", len(user_ratings))
                with col2:
                    st.metric("⭐ Avg Rating", f"{user_ratings['rating'].mean():.2f}")
                with col3:
                    st.metric("👍 High Ratings", len(user_ratings[user_ratings['rating'] >= 4]))
                with col4:
                    st.metric("👎 Low Ratings", len(user_ratings[user_ratings['rating'] <= 2]))
                
                # History table
                st.subheader("📋 Rating History")
                merge_cols = ['game_id', 'title_clean', 'genres']
                if 'description' in st.session_state.games_df.columns:
                    merge_cols.append('description')
                history_df = user_ratings.merge(
                    st.session_state.games_df[merge_cols],
                    on='game_id',
                    how='left'
                )[merge_cols[1:] + ['rating', 'timestamp']].sort_values('timestamp', ascending=False)
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Chart
                st.subheader("📈 Rating Distribution")
                rating_counts = user_ratings['rating'].value_counts().sort_index()
                st.bar_chart(rating_counts)
                
                # Recommendation stats (realtime)
                st.subheader("🎯 Recommendation Statistics")
                
                # Get stats (will be updated on each rerun for realtime feel)
                stats = st.session_state.db.get_recommendation_stats(st.session_state.current_user_id)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📤 Unique Games Recommended", stats['total_recommendations'])
                with col2:
                    st.metric("👆 Games Clicked", stats['total_clicks'])
                with col3:
                    st.metric("📊 Click Rate", f"{stats['click_rate']:.2f}%")
                with col4:
                    st.metric("🔄 Recommendation Sessions", stats.get('total_sessions', 0))
                
                # Info about what the stats mean
                with st.expander("ℹ️ About these statistics", expanded=False):
                    st.write("""
                    **📤 Unique Games Recommended:** Total number of different games that have been recommended to you.
                    
                    **👆 Games Clicked:** Number of unique games you've clicked to view details.
                    
                    **📊 Click Rate:** Percentage of recommended games that you've clicked on.
                    
                    **🔄 Recommendation Sessions:** Number of times you've requested recommendations.
                    
                    *Note: Statistics update automatically when you interact with recommendations.*
                    """)
            else:
                st.info("💡 You don't have any rating history yet. Rate some games to get started!")
        
        with tab4:
            st.header("ℹ️ Model Information")
            
            if st.session_state.model_data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎮 Total Games", len(st.session_state.games_df))
                with col2:
                    st.metric("👥 Total Users", len(st.session_state.model_data.user_id_to_index))
                with col3:
                    st.metric("⚖️ Alpha (Weight)", st.session_state.model_data.alpha)
                with col4:
                    st.metric("📊 Features", "TF-IDF + KNN")
                
                st.markdown("---")
                
                st.subheader("📋 Model Details")
                st.write("""
                **Hybrid Recommendation System:**
                - **Content-Based Filtering**: Uses TF-IDF vectorization of game features (title, genres, developers, etc.)
                - **Collaborative Filtering**: Uses user-item interaction matrix with KNN
                - **Hybrid Approach**: Combines both methods with weighted alpha parameter
                
                **Features per Game:**
                - Title
                - Genres
                - Description
                - Release Year
                - Price
                """)
                
                # Sample games
                st.subheader("📋 Sample Games in Database")
                # Show columns that are available
                display_cols = ['game_id', 'title_clean', 'genres', 'year']
                # Add other columns if they exist
                for col in ['description', 'all_tags', 'price']:
                    if col in st.session_state.games_df.columns:
                        display_cols.append(col)
                
                sample_games = st.session_state.games_df.head(10)[display_cols]
                st.dataframe(sample_games, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

