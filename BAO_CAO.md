# BÁO CÁO DỰ ÁN
## HỆ THỐNG GỢI Ý GAME STEAM SỬ DỤNG HYBRID RECOMMENDATION SYSTEM

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Giới thiệu
Dự án xây dựng một hệ thống gợi ý game Steam sử dụng phương pháp Hybrid Recommendation, kết hợp Content-Based Filtering và Collaborative Filtering. Hệ thống được triển khai dưới dạng ứng dụng web với giao diện Streamlit, hỗ trợ lưu trữ lịch sử người dùng và gợi ý theo thời gian thực.

### 1.2. Mục tiêu
- Xây dựng hệ thống gợi ý game chính xác và đa dạng
- Tích hợp giao diện người dùng thân thiện và dễ sử dụng
- Lưu trữ và quản lý lịch sử tương tác của người dùng
- Cung cấp gợi ý theo thời gian thực dựa trên đánh giá của người dùng
- Hỗ trợ triển khai lên cloud để sử dụng rộng rãi

### 1.3. Phạm vi
- **Dataset**: Steam Games Dataset với hơn 50,000 games và 41 triệu+ ratings
- **Thuật toán**: Hybrid Recommendation (Content-Based + Collaborative Filtering)
- **Giao diện**: Streamlit web application
- **Lưu trữ**: SQLite database cho user history
- **Triển khai**: Streamlit Cloud hoặc local deployment

---

## 2. PHƯƠNG PHÁP VÀ THUẬT TOÁN

### 2.1. Hybrid Recommendation System

Hệ thống kết hợp hai phương pháp chính:

#### 2.1.1. Content-Based Filtering
- **Mục đích**: Tìm các game tương tự dựa trên đặc điểm của game (nội dung)
- **Kỹ thuật**:
  - **TF-IDF Vectorization**: Chuyển đổi thông tin game (title, genres, developers, publishers, description, tags) thành vector số
  - **K-Nearest Neighbors (KNN)**: Tìm k game gần nhất dựa trên cosine similarity
- **Ưu điểm**: 
  - Không cần dữ liệu đánh giá từ người dùng khác
  - Gợi ý dựa trên đặc điểm thực tế của game
  - Phù hợp cho game mới hoặc ít người chơi

#### 2.1.2. Collaborative Filtering
- **Mục đích**: Tìm các game được người dùng tương tự đánh giá cao
- **Kỹ thuật**:
  - **Item-Item Collaborative Filtering**: Xây dựng ma trận tương tác game-user (sparse matrix)
  - **KNN trên Item-User Matrix**: Tìm các game có pattern đánh giá tương tự
- **Ưu điểm**:
  - Tận dụng hành vi của cộng đồng người dùng
  - Phát hiện các mối quan hệ ẩn giữa games
  - Gợi ý dựa trên sở thích thực tế của người dùng

#### 2.1.3. Hybrid Approach
- **Công thức kết hợp**:
  ```
  Final Score = α × Collaborative Score + (1-α) × Content-Based Score
  ```
  
  **Chi tiết**:
  ```
  score(game) = Σ_{rated_game ∈ user_ratings} [
      α × sim_collab(game, rated_game) × weight(rated_game) +
      (1-α) × sim_content(game, rated_game) × weight(rated_game)
  ]
  ```
  
  Trong đó:
  - `α` (alpha): Tham số trọng số, mặc định = 0.5
    - `α = 0.5`: Cân bằng giữa Content-Based và Collaborative
    - `α < 0.5`: Ưu tiên Content-Based (dựa trên nội dung game)
    - `α > 0.5`: Ưu tiên Collaborative (dựa trên pattern từ cộng đồng)
  - `sim_collab`: Similarity từ Item-Item Collaborative Filtering
  - `sim_content`: Similarity từ Content-Based (TF-IDF)
  - `weight(rated_game) = (rating / 5.0)²`: Trọng số dựa trên rating (exponential)
  
- **Lợi ích**:
  - Kết hợp ưu điểm của cả hai phương pháp
  - Giảm thiểu nhược điểm của từng phương pháp riêng lẻ
  - Tăng độ chính xác và đa dạng của gợi ý
  - Ratings cao có ảnh hưởng lớn hơn (exponential weight)

### 2.2. Xử lý Dữ liệu

#### 2.2.1. Dữ liệu Game
- **Nguồn**: `games.csv` với các cột:
  - `app_id`, `title`, `date_release`
  - `genres`, `developers`, `publishers`
  - `price_final`, `price_original`, `discount`
  - `rating`, `positive_ratio`, `user_reviews`
  - `win`, `mac`, `linux`, `steam_deck`
- **Xử lý**:
  - Làm sạch title (loại bỏ ký tự đặc biệt, chuẩn hóa)
  - Trích xuất năm phát hành từ `date_release`
  - Xử lý price: ưu tiên `price_final` (giá sau discount), fallback `price_original`
  - Gộp genres, developers, publishers thành chuỗi text cho TF-IDF
  - Xử lý missing values

#### 2.2.2. Dữ liệu Ratings
- **Nguồn**: `recommendations.csv` với 41+ triệu ratings
- **Xử lý**:
  - Chuyển đổi `is_recommended` (True/False) thành rating (5/1)
  - Tính toán rating dựa trên `positive_ratio` và `user_reviews` nếu cần
  - Xây dựng ma trận sparse (CSR Matrix) để tối ưu bộ nhớ
  - Tạo mapping `user_id ↔ index` và `game_id ↔ index`

### 2.3. Feature Engineering

#### 2.3.1. TF-IDF Features
- **Input**: Kết hợp các trường:
  - Title (cleaned)
  - Genres
  - Developers
  - Publishers
  - Description (nếu có)
  - Tags (nếu có)
- **Output**: Vector TF-IDF với số chiều tùy chọn (thường 1000-5000)
- **Mục đích**: Mã hóa thông tin nội dung game thành vector số để tính similarity

#### 2.3.2. User-Item Matrix
- **Cấu trúc**: Sparse matrix (CSR format)
  - Rows: Games (items)
  - Columns: Users
  - Values: Ratings (1-5) hoặc binary (0/1)
- **Tối ưu**: Sử dụng sparse matrix để tiết kiệm bộ nhớ (chỉ lưu giá trị khác 0)

### 2.4. Công thức Toán học

#### 2.4.1. TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF được sử dụng để vector hóa thông tin text của game (title, genres, developers, publishers, description, tags).

**Term Frequency (TF)**:
```
TF(t, d) = (Số lần từ t xuất hiện trong document d) / (Tổng số từ trong document d)
```

Hoặc dạng log:
```
TF(t, d) = log(1 + số_lần_từ_t_xuất_hiện)
```

**Inverse Document Frequency (IDF)**:
```
IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)
```

Trong đó:
- `N`: Tổng số documents (games) trong corpus
- `|{d ∈ D : t ∈ d}|`: Số documents chứa từ `t`

**TF-IDF Score**:
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Vector TF-IDF cho game**:
```
v_game = [TF-IDF(t₁, game, D), TF-IDF(t₂, game, D), ..., TF-IDF(tₙ, game, D)]
```

#### 2.4.2. Distance Metrics trong KNN

KNN sử dụng khoảng cách (distance) để tìm các game gần nhất. Có thể dùng nhiều metric khác nhau:

**Euclidean Distance** (mặc định trong scikit-learn):
```
d(A, B) = √(Σ(Aᵢ - Bᵢ)²)
```

**Cosine Distance** (nếu set `metric='cosine'`):
```
d_cosine(A, B) = 1 - cos(θ)
```

Trong đó cosine similarity:
```
cos(θ) = (A · B) / (||A|| × ||B||) = Σ(Aᵢ × Bᵢ) / (√(ΣAᵢ²) × √(ΣBᵢ²))
```

**Lưu ý**: 
- Trong implementation hiện tại, KNN mặc định dùng **Euclidean distance** (metric='minkowski')
- Để dùng cosine similarity, cần khởi tạo: `NearestNeighbors(metric='cosine')`
- Code chuyển đổi distance thành similarity: `sim = 1 - dist`
- Với Euclidean: `sim = 1 - dist` không phải cosine similarity thực sự, chỉ là cách normalize
- Với Cosine: `sim = 1 - dist` chính là cosine similarity (vì `dist = 1 - cos(θ)`)

#### 2.4.3. K-Nearest Neighbors (KNN)

KNN tìm k game gần nhất dựa trên khoảng cách (distance).

**KNN Algorithm**:
1. Tính khoảng cách từ game query đến tất cả games khác (dùng metric đã chọn)
2. Sắp xếp theo khoảng cách tăng dần
3. Chọn k game gần nhất (k = 15 trong implementation)
4. Chuyển đổi distance thành similarity: `sim = 1 - dist`

**Trong implementation**:
- **Content-Based KNN**: Tìm neighbors dựa trên TF-IDF vectors
- **Collaborative KNN**: Tìm neighbors dựa trên item-user matrix
- Mặc định dùng **Euclidean distance** (metric='minkowski')
- Có thể thay đổi sang `metric='cosine'` để dùng cosine similarity

#### 2.4.4. Hybrid Recommendation Score

**Công thức tính điểm gợi ý cho một game**:

Với mỗi game `g` mà user đã rate:

**Content-Based Score**:
```
score_content(g, g_rated) = similarity_content(g, g_rated) × (1 - α) × weight(g_rated)
```

**Collaborative Score**:
```
score_collab(g, g_rated) = similarity_collab(g, g_rated) × α × weight(g_rated)
```

**Tổng điểm cho game `g`**:
```
total_score(g) = Σ[score_content(g, g_rated) + score_collab(g, g_rated)]
```

Trong đó:
- `g_rated`: Các game user đã đánh giá
- `similarity_content(g, g_rated)`: Similarity từ TF-IDF vectors (từ KNN distance: `sim = 1 - dist`)
- `similarity_collab(g, g_rated)`: Similarity từ item-user matrix (từ KNN distance: `sim = 1 - dist`)
- `weight(g_rated)`: Trọng số dựa trên rating
- **Lưu ý**: Similarity được tính từ distance của KNN, không phải cosine similarity trực tiếp (trừ khi set `metric='cosine'`)

**Weight Calculation**:
```
weight(rating) = (rating / 5.0)²
```

Ví dụ:
- Rating = 5 → weight = 1.0
- Rating = 4 → weight = 0.64
- Rating = 3 → weight = 0.36
- Rating = 2 → weight = 0.16
- Rating = 1 → weight = 0.04

**Công thức tổng quát**:
```
score(g) = Σ_{g_rated ∈ R_user} [
    (1 - α) × sim_content(g, g_rated) × (r_g_rated / 5)² +
    α × sim_collab(g, g_rated) × (r_g_rated / 5)²
]
```

Trong đó:
- `R_user`: Tập các game user đã rate
- `r_g_rated`: Rating của user cho game `g_rated`
- `α`: Tham số trọng số (mặc định = 0.5)
- `sim_content`: Similarity từ Content-Based (TF-IDF) - được tính từ KNN distance
- `sim_collab`: Similarity từ Collaborative (Item-Item CF) - được tính từ KNN distance

#### 2.4.5. Evaluation Metrics

**Root Mean Squared Error (RMSE)**:
```
RMSE = √(1/n × Σ(predicted_i - actual_i)²)
```

**Mean Absolute Error (MAE)**:
```
MAE = 1/n × Σ|predicted_i - actual_i|
```

**Precision@K**:
```
Precision@K = (Số items relevant trong top-K) / K
```

**Recall@K**:
```
Recall@K = (Số items relevant trong top-K) / (Tổng số items relevant)
```

**Hit Rate**:
```
Hit Rate = (Số users có ít nhất 1 hit trong top-K) / (Tổng số users)
```

Trong đó:
- `hit`: Một item được gợi ý nằm trong test set của user
- `relevant`: Item mà user thực sự đánh giá cao trong test set

#### 2.4.6. Item-Item Collaborative Filtering

**Item Similarity** (từ KNN trên item-user matrix):
```
sim(i, j) = 1 - distance(i, j)
```

Trong đó:
- `distance(i, j)`: Khoảng cách giữa item `i` và `j` (từ KNN)
- Nếu dùng `metric='cosine'`: `distance = 1 - cos(θ)` → `sim = cos(θ) = (R_i · R_j) / (||R_i|| × ||R_j||)`
- Nếu dùng `metric='minkowski'` (mặc định): `distance = √(Σ(R_i - R_j)²)` → `sim = 1 - distance`
- `R_i`: Vector ratings của item `i` (ratings từ tất cả users)
- `R_j`: Vector ratings của item `j`

**Prediction cho user `u` và item `i`**:
```
pred(u, i) = Σ_{j ∈ N(i)} sim(i, j) × r(u, j) / Σ_{j ∈ N(i)} |sim(i, j)|
```

Trong đó:
- `N(i)`: Tập các items tương tự với item `i` (k nearest neighbors)
- `r(u, j)`: Rating của user `u` cho item `j`

---

## 3. KIẾN TRÚC HỆ THỐNG

### 3.1. Cấu trúc Dự án

```
KHDL/
├── data/
│   ├── raw/              # Dữ liệu thô (không cần nếu đã có model)
│   └── processed/        # Dữ liệu đã xử lý (games_clean.csv)
├── models/               # Model đã train (recommendation_model.pkl)
├── src/
│   ├── database.py       # Module quản lý database SQLite
│   └── recommendation.py # Module recommendation system (optional)
├── app/
│   └── streamlit_app.py  # Ứng dụng Streamlit chính
├── khdl-game.ipynb       # Notebook training model trên Kaggle
├── requirements.txt      # Dependencies
├── .streamlit/
│   └── config.toml       # Cấu hình Streamlit
└── README.md, DEPLOY.md  # Tài liệu
```

### 3.2. Kiến trúc Ứng dụng

#### 3.2.1. Backend Components
- **HybridRecommendationSystem**: Class chính xử lý recommendation
  - `load_from_pickle()`: Load model đã train
  - `recommend_by_game()`: Gợi ý game tương tự
  - `recommend_by_user_realtime()`: Gợi ý dựa trên ratings của user
  - `recommend_by_user_with_filters()`: Gợi ý với filters (year, price, genres)
- **UserHistoryDB**: Class quản lý database SQLite
  - Lưu trữ ratings, recommendations, search history
  - Tracking clicks và views
  - Thống kê user behavior

#### 3.2.2. Frontend Components (Streamlit)
- **Sidebar**: Quản lý user (create, select, delete, test users)
- **Tab 1 - Game Recommendations**:
  - Game-Based: Tìm game tương tự
  - User-Based: Gợi ý cá nhân hóa với filters
- **Tab 2 - Search Games**: Tìm kiếm và lọc game
- **Tab 3 - History**: Lịch sử ratings và thống kê
- **Tab 4 - Model Info**: Thông tin về model

### 3.3. Data Flow

```
User Input (Ratings/Search)
    ↓
Streamlit App (UI)
    ↓
UserHistoryDB (SQLite) ← Lưu trữ lịch sử
    ↓
HybridRecommendationSystem
    ↓
    ├─→ Content-Based (TF-IDF + KNN)
    └─→ Collaborative (Item-Item CF + KNN)
    ↓
Hybrid Score Calculation
    ↓
Filtering (year, price, genres)
    ↓
Top-N Recommendations
    ↓
Display to User
```

---

## 4. TÍNH NĂNG CHÍNH

### 4.1. Gợi Ý Game

#### 4.1.1. Game-Based Recommendations
- Người dùng chọn một game
- Hệ thống tìm các game tương tự dựa trên:
  - Nội dung (genres, developers, publishers, description)
  - Pattern đánh giá từ người dùng khác
- Hiển thị top-N game tương tự với similarity score

#### 4.1.2. User-Based Recommendations
- Người dùng đánh giá các game (1-5 sao) → Lưu vào database SQLite
- Hệ thống gợi ý game dựa trên:
  - **Lịch sử đánh giá của chính người dùng**: 
    - Lấy tất cả ratings từ database: `get_user_ratings(user_id)`
    - Với mỗi game đã rate, tìm game tương tự
  - **Pattern đánh giá từ cộng đồng (Item-Item Collaborative Filtering)**:
    - Sử dụng `item_user_matrix` - ma trận cho biết ai đã đánh giá game nào
    - Tìm các game có pattern đánh giá tương tự từ TẤT CẢ người dùng khác
    - Ví dụ: Nếu bạn rate "Game A" cao, và trong training data, những người rate "Game A" cao cũng thường rate "Game B" cao → "Game B" được gợi ý
  - **Kết hợp Content-Based**: Tìm game tương tự về nội dung (genres, developers, etc.)
- **Cách sử dụng**:
  1. Vào tab "🎯 Game Recommendations"
  2. Chọn "User-Based (Rate Games First)"
  3. Tìm và rate các game bạn đã chơi (1-5 sao)
  4. Click "🎮 Get Recommendations Based on My Ratings"
  5. Hệ thống sẽ gợi ý dựa trên ratings của bạn + pattern từ cộng đồng
- **Filters**:
  - `min_year`: Năm phát hành tối thiểu
  - `max_price`: Giá tối đa
  - `required_genres`: Genres bắt buộc
  - `exclude_genres`: Genres loại trừ

### 4.2. Tìm kiếm và Lọc Game

- **Tìm kiếm theo tên**: Autocomplete với gợi ý real-time
- **Lọc theo**:
  - Genre (dropdown với autocomplete)
  - Năm phát hành (slider)
  - Giá (slider)
- **Hiển thị thông tin chi tiết**:
  - Title, Genres, Year, Price
  - Developers, Publishers, Producers
  - Description, Tags
  - Game ID

### 4.3. Quản lý User

- **Tạo user mới**: Tự động generate UUID
- **Chọn user hiện có**: Dropdown danh sách users
- **Xóa user**: Xóa user và toàn bộ lịch sử
- **Test users**: Chọn user từ model training data, tự động import ratings

### 4.4. Lịch sử và Thống kê

- **Rating History**:
  - Danh sách tất cả game đã đánh giá
  - Rating distribution chart
  - Số lượng ratings theo mức độ
- **Recommendation Statistics**:
  - Tổng số game được gợi ý (unique)
  - Tổng số game đã click
  - Tổng số sessions
  - Real-time updates

### 4.5. Giao diện Người dùng

- **Custom CSS**: Gradient colors, card design
- **Responsive Layout**: Columns, expanders
- **Interactive Elements**:
  - Autocomplete search bars
  - Quick selection buttons
  - Hide/Show details buttons
  - Star rating widgets
- **Real-time Updates**: Recommendations cập nhật ngay khi rate game

---

## 5. DỮ LIỆU VÀ XỬ LÝ

### 5.1. Dataset

- **Games**: 50,872 games
- **Ratings**: 41,154,794 recommendations
- **Users**: 13,781,059+ users (từ training data)
- **Features per Game**:
  - Title (cleaned)
  - Genres
  - Developers
  - Publishers
  - Release Year
  - Price (price_final hoặc price_original)
  - Description (optional)
  - Tags (optional)

### 5.2. Data Preprocessing

#### 5.2.1. Games Data
- Clean title: Loại bỏ ký tự đặc biệt, chuẩn hóa
- Extract year từ date_release
- Handle price: Ưu tiên price_final, fallback price_original
- Process missing values: Fill với giá trị mặc định
- Combine text features: Gộp genres, developers, publishers cho TF-IDF

#### 5.2.2. Ratings Data
- Convert is_recommended → rating (5/1)
- Build sparse matrix (CSR format)
- Create ID mappings (user_id ↔ index, game_id ↔ index)
- Memory optimization: Chunk processing cho file lớn

### 5.3. Model Training

- **Environment**: Kaggle Notebook (GPU Tesla T4, 15.83 GB)
- **Process**:
  1. Load và preprocess data
  2. Build TF-IDF vectors
  3. Train Content-Based KNN model
  4. Build Item-User matrix
  5. Train Collaborative KNN model
  6. Evaluate model (RMSE, MAE, Precision@10, Recall@10)
  7. Save model (full và lightweight version)
- **Output**: `recommendation_model.pkl` (lightweight cho deployment)

---

## 6. ĐÁNH GIÁ VÀ KẾT QUẢ

### 6.1. Metrics

#### 6.1.1. Accuracy Metrics
- **RMSE (Root Mean Squared Error)**: ~0.79
- **MAE (Mean Absolute Error)**: ~0.55
- **Đánh giá**: RMSE và MAE ở mức chấp nhận được cho hệ thống recommendation

#### 6.1.2. Ranking Metrics
- **Precision@10**: ~0.13% (0.0013)
- **Recall@10**: ~1.15% (0.0115)
- **Hit Rate**: 13/1000 users có ít nhất 1 hit trong top-10
- **Đánh giá**: Precision/Recall thấp do:
  - Dataset lớn (50k+ games)
  - Cold start problem cho users mới
  - Có thể cải thiện bằng tuning hyperparameters

### 6.2. Performance

- **Model Loading**: ~2-5 giây (với caching)
- **Recommendation Generation**: <1 giây (với caching)
- **Real-time Updates**: Tức thì khi user rate game
- **Memory Usage**: Tối ưu với sparse matrices

### 6.3. User Experience

- **Giao diện**: Đẹp, hiện đại với custom CSS
- **Tốc độ**: Nhanh nhờ caching
- **Tính năng**: Đầy đủ (search, filter, history, statistics)
- **Usability**: Dễ sử dụng, autocomplete, quick selection

---

## 7. TRIỂN KHAI VÀ SỬ DỤNG

### 7.1. Local Deployment

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Đảm bảo có model file
# models/recommendation_model.pkl

# 3. Chạy ứng dụng
streamlit run app/streamlit_app.py
```

### 7.2. Cloud Deployment (Streamlit Cloud)

#### 7.2.1. Chuẩn bị
- Push code lên GitHub
- Sử dụng Git LFS cho model file lớn (>100MB)
- Đảm bảo có `requirements.txt`, `.streamlit/config.toml`

#### 7.2.2. Deploy
1. Truy cập https://share.streamlit.io
2. Đăng nhập bằng GitHub
3. Chọn repository và branch
4. Set main file: `app/streamlit_app.py`
5. Deploy

#### 7.2.3. Lưu ý
- Model file: Dùng Git LFS hoặc tải từ URL
- Database: SQLite sẽ reset khi restart (cần persistent storage cho production)
- Memory: Streamlit Cloud free tier có 1GB RAM
- Performance: Sử dụng caching để tối ưu

### 7.3. Sử dụng Ứng dụng

1. **Khởi tạo**: App tự động load model
2. **Tạo/Chọn User**: Sidebar → Create new user hoặc Select user
3. **Đánh giá Game**: 
   - Tab "Game Recommendations" → User-Based
   - Tìm game và rate (1-5 sao)
4. **Xem Gợi ý**: Recommendations tự động cập nhật
5. **Tìm kiếm**: Tab "Search Games" → Tìm và xem chi tiết
6. **Xem Lịch sử**: Tab "History" → Ratings và statistics

---

## 8. CÔNG NGHỆ SỬ DỤNG

### 8.1. Backend
- **Python 3.10+**
- **scikit-learn**: TF-IDF, KNN, metrics
- **pandas, numpy**: Data processing
- **scipy**: Sparse matrices
- **pickle**: Model serialization

### 8.2. Frontend
- **Streamlit**: Web framework
- **Custom CSS**: Styling
- **Plotly**: Charts (rating distribution)

### 8.3. Database
- **SQLite**: User history storage
- **Tables**:
  - `users`: User information
  - `ratings`: User ratings
  - `recommendations_log`: Recommendation tracking
  - `search_log`: Search history

### 8.4. Deployment
- **Streamlit Cloud**: Cloud hosting
- **Git LFS**: Large file storage
- **GitHub**: Version control

---

## 9. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 9.1. Kết luận

Dự án đã xây dựng thành công một hệ thống gợi ý game Steam với các đặc điểm:

✅ **Thuật toán**: Hybrid approach kết hợp Content-Based và Collaborative Filtering  
✅ **Giao diện**: Đẹp, hiện đại, dễ sử dụng  
✅ **Tính năng**: Đầy đủ (search, filter, history, real-time recommendations)  
✅ **Performance**: Tối ưu với caching và sparse matrices  
✅ **Triển khai**: Hỗ trợ local và cloud deployment  

### 9.2. Điểm mạnh

- **Hybrid Approach**: Kết hợp ưu điểm của cả hai phương pháp
- **Real-time**: Gợi ý cập nhật ngay khi user rate game
- **User History**: Lưu trữ và quản lý lịch sử người dùng
- **Filters**: Hỗ trợ lọc theo nhiều tiêu chí
- **Scalable**: Có thể mở rộng với dataset lớn hơn

### 9.3. Hạn chế

- **Precision/Recall**: Còn thấp, cần tuning hyperparameters
- **Cold Start**: Vấn đề với users/games mới
- **Database**: SQLite không persistent trên Streamlit Cloud free tier
- **Model Size**: Model file lớn, cần Git LFS

### 9.4. Hướng phát triển

#### 9.4.1. Cải thiện Thuật toán
- **Deep Learning**: Thử nghiệm Neural Collaborative Filtering, Wide & Deep
- **Matrix Factorization**: SVD, NMF để giảm chiều dữ liệu
- **Ensemble Methods**: Kết hợp nhiều models
- **Hyperparameter Tuning**: Grid search, Bayesian optimization

#### 9.4.2. Tính năng mới
- **Context-Aware Recommendations**: Dựa trên thời gian, thiết bị, vị trí
- **Advanced Embeddings**: Word2Vec, BERT cho text features
- **Explainability**: Giải thích tại sao gợi ý game này
- **A/B Testing**: So sánh hiệu quả các thuật toán

#### 9.4.3. Infrastructure
- **Database**: Chuyển sang PostgreSQL/MySQL cho production
- **Caching**: Redis cho caching recommendations
- **API**: REST API để tích hợp với ứng dụng khác
- **Monitoring**: Logging, metrics, alerting

#### 9.4.4. User Experience
- **Personalization**: Profile page, preferences
- **Social Features**: Share recommendations, follow users
- **Notifications**: Thông báo game mới phù hợp
- **Mobile App**: Native app cho iOS/Android

---

## 10. TÀI LIỆU THAM KHẢO

- **Dataset**: Steam Games Dataset trên Kaggle
- **Libraries**: scikit-learn, pandas, numpy, streamlit
- **Papers**: 
  - "Item-based Collaborative Filtering Recommendation Algorithms" (Sarwar et al., 2001)
  - "Hybrid Recommender Systems: Survey and Experiments" (Burke, 2002)
