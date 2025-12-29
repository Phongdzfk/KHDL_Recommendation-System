"""
Generate and save all visualizations from the notebook
Run this once to pre-compute visualizations, then load them in the Streamlit app
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.database import UserHistoryDB

# Create visualizations folder
VIZ_DIR = Path('models/visualizations')
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Load model data
try:
    with open('models/recommendation_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    games_df = model_data['games_df']
    print(f"✅ Model loaded: {len(games_df)} games")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Load database ratings
db = UserHistoryDB()
all_ratings = db.get_all_user_ratings()

if len(all_ratings) == 0:
    print("⚠️  No ratings in database. Skipping visualization generation.")
    sys.exit(0)

print(f"✅ Database loaded: {len(all_ratings)} ratings")

# ============================================================================
# 1. RATING DISTRIBUTION
# ============================================================================
print("\n1. Generating Rating Distribution...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Rating Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Histogram
    rating_values = all_ratings['rating'].values
    axes[0].hist(rating_values, bins=20, edgecolor='black', alpha=0.75, color='skyblue')
    axes[0].set_xlabel('Rating', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Histogram - Rating Distribution')
    axes[0].grid(True, alpha=0.3)
    
    mean_rating = rating_values.mean()
    median_rating = np.median(rating_values)
    axes[0].axvline(mean_rating, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rating:.2f}')
    axes[0].axvline(median_rating, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rating:.2f}')
    axes[0].legend()
    
    # Box Plot
    axes[1].boxplot([rating_values], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', alpha=0.7),
                    meanprops=dict(color='red', linewidth=2),
                    medianprops=dict(color='blue', linewidth=2))
    axes[1].set_ylabel('Rating', fontweight='bold')
    axes[1].set_title('Box Plot - Rating Distribution')
    axes[1].set_xticklabels(['All Ratings'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '1_rating_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 1_rating_distribution.png")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# 2. GENRE FREQUENCY
# ============================================================================
print("\n2. Generating Genre Analysis...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Genre Frequency Analysis', fontsize=14, fontweight='bold')
    
    # Merge ratings with genres
    games_df_small = games_df[['game_id', 'genres']]
    ratings_with_genres = all_ratings.merge(games_df_small, on='game_id', how='left')
    
    genre_ratings = []
    for idx, row in ratings_with_genres.iterrows():
        if pd.notna(row['genres']):
            genres_list = str(row['genres']).split(',')
            for genre in genres_list:
                genre = genre.strip()
                if genre and genre.lower() not in ['unknown', 'nan']:
                    genre_ratings.append({'genre': genre, 'rating': row['rating']})
    
    if genre_ratings:
        genre_df = pd.DataFrame(genre_ratings)
        
        # Genre frequency
        genre_counts = genre_df['genre'].value_counts().head(15)
        axes[0].barh(range(len(genre_counts)), genre_counts.values, alpha=0.8, edgecolor='black')
        axes[0].set_yticks(range(len(genre_counts)))
        axes[0].set_yticklabels(genre_counts.index, fontsize=9)
        axes[0].set_xlabel('Number of Ratings', fontweight='bold')
        axes[0].set_title('Top 15 Genres by Frequency')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Genre average rating
        genre_avg = genre_df.groupby('genre')['rating'].agg(['mean', 'count'])
        genre_avg = genre_avg[genre_avg['count'] >= 5].sort_values('mean', ascending=False).head(15)
        
        axes[1].barh(range(len(genre_avg)), genre_avg['mean'], alpha=0.8, edgecolor='black', color='coral')
        axes[1].set_yticks(range(len(genre_avg)))
        axes[1].set_yticklabels(genre_avg.index, fontsize=9)
        axes[1].set_xlabel('Average Rating', fontweight='bold')
        axes[1].set_title('Top 15 Genres by Average Rating')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / '2_genre_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 2_genre_analysis.png")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# 3. TOP GAMES
# ============================================================================
print("\n3. Generating Top Games Analysis...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Top Games Analysis', fontsize=14, fontweight='bold')
    
    # Top games by rating count
    top_games_ids = all_ratings['game_id'].value_counts().head(20).index
    top_games_stats = []
    for gid in top_games_ids:
        game_info = games_df[games_df['game_id'] == gid]
        if len(game_info) > 0:
            title = game_info.iloc[0]['title_clean']
            count = len(all_ratings[all_ratings['game_id'] == gid])
            avg = all_ratings[all_ratings['game_id'] == gid]['rating'].mean()
            top_games_stats.append({'game_id': gid, 'title': title[:30], 'count': count, 'avg': avg})
    
    if top_games_stats:
        top_df = pd.DataFrame(top_games_stats)
        
        # Bar chart - Top games by count
        axes[0].barh(range(len(top_df)), top_df['count'], alpha=0.85, edgecolor='black', color='steelblue')
        axes[0].set_yticks(range(len(top_df)))
        axes[0].set_yticklabels(top_df['title'], fontsize=8)
        axes[0].set_xlabel('Number of Ratings', fontweight='bold')
        axes[0].set_title('Top 20 Most Rated Games')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Scatter - Popularity vs Quality
        scatter = axes[1].scatter(top_df['count'], top_df['avg'],
                                 s=top_df['count']/10, c=top_df['avg'],
                                 cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2)
        axes[1].set_xlabel('Number of Ratings (Popularity)', fontweight='bold')
        axes[1].set_ylabel('Average Rating (Quality)', fontweight='bold')
        axes[1].set_title('Popularity vs Quality')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Avg Rating')
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / '3_top_games.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✅ Saved: 3_top_games.png")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# 4. USER STATISTICS
# ============================================================================
print("\n4. Generating User Statistics...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('User Activity Analysis', fontsize=14, fontweight='bold')
    
    user_stats = all_ratings.groupby('user_id').agg({'rating': ['count', 'mean']}).reset_index()
    user_stats.columns = ['user_id', 'num_ratings', 'avg_rating']
    
    # Histogram - User activity
    axes[0].hist(user_stats['num_ratings'], bins=30, edgecolor='black', alpha=0.75, color='coral')
    axes[0].set_xlabel('Number of Ratings per User', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('User Activity Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Scatter - Activity vs avg rating
    axes[1].scatter(user_stats['num_ratings'], user_stats['avg_rating'],
                   s=80, alpha=0.6, c=user_stats['avg_rating'],
                   cmap='RdYlGn', edgecolors='black', linewidth=1)
    axes[1].set_xlabel('Number of Ratings', fontweight='bold')
    axes[1].set_ylabel('Average Rating Given', fontweight='bold')
    axes[1].set_title('User Activity vs Rating Behavior')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / '4_user_statistics.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 4_user_statistics.png")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ ALL VISUALIZATIONS GENERATED!")
print(f"📁 Saved to: {VIZ_DIR}")
print("=" * 60)
