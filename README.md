# Steam Games Recommendation System

## Description
A Steam Games recommendation system using Hybrid Recommendation (Content-Based + Collaborative Filtering) with Streamlit interface, supporting user history storage and real-time recommendations.

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
KHDL/
├── data/
│   ├── raw/          # Raw data (not needed if model already exists)
│   └── processed/    # Processed data (games_clean.csv)
├── models/           # Trained models (recommendation_model.pkl)
├── src/              # Source code
│   ├── recommendation.py  # Recommendation system
│   └── database.py        # Database for user history
├── app/
│   └── streamlit_app.py   # Main interface
├── kaggle_notebook_template.py  # Template for training on Kaggle
└── requirements.txt
```

## Features

- ✅ **Beautiful Interface**: Streamlit with custom CSS
- ✅ **Hybrid Recommendation**: Content-Based + Collaborative Filtering
- ✅ **User History Storage**: SQLite database
- ✅ **Real-time Recommendations**: Updates immediately when rating games
- ✅ **Pre-trained Model Loading**: No need to train locally
- ✅ **≥5 Features per Game**: Title, Genres, Developers, Publishers, Release Year, Price

## Quick Setup

1. **Train model on Kaggle** (see `KAGGLE_GUIDE.md`)
2. **Download model**:
   - `recommendation_model.pkl` → `models/`
   - `games_clean.csv` → `data/processed/` (or rename from `movies_clean.csv`)
3. **Run app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Deploy

The application can be deployed to Streamlit Cloud or other platforms.
