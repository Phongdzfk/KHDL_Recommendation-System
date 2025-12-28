# 🚀 HƯỚNG DẪN DEPLOY LÊN STREAMLIT CLOUD

## 📋 Tổng quan

Model file của bạn rất lớn (~1.18GB), cần dùng **Git LFS** để upload lên GitHub và deploy lên Streamlit Cloud.

---

## BƯỚC 1: Chuẩn bị Repository GitHub

### 1.1. Tạo Repository trên GitHub

1. Truy cập: https://github.com/new
2. Tạo repository mới:
   - **Repository name**: `steam-games-recommendation` (hoặc tên bạn muốn)
   - **Visibility**: Public (để Streamlit Cloud free tier có thể truy cập)
   - **Không** tích "Initialize with README" (vì bạn đã có code)
3. Click "Create repository"

### 1.2. Khởi tạo Git Local

Mở terminal/command prompt trong thư mục dự án:

```bash
# Kiểm tra xem đã có git chưa
git --version

# Nếu chưa có git, cài đặt từ: https://git-scm.com/downloads

# Khởi tạo git repository
git init

# Kiểm tra trạng thái
git status
```

---

## BƯỚC 2: Cài đặt và Cấu hình Git LFS

### 2.1. Cài đặt Git LFS

**Windows:**
```bash
# Download và cài đặt từ: https://git-lfs.github.com/
# Hoặc dùng Chocolatey:
choco install git-lfs

# Hoặc dùng winget:
winget install GitHub.GitLFS
```

**Kiểm tra cài đặt:**
```bash
git lfs version
# Nên hiển thị: git-lfs/3.x.x
```

### 2.2. Cấu hình Git LFS cho Model File

```bash
# Khởi tạo Git LFS trong repository
git lfs install

# Track file model (file lớn > 100MB)
git lfs track "models/*.pkl"
git lfs track "*.pkl"

# Tạo file .gitattributes (nếu chưa có)
# File này sẽ tự động được tạo khi chạy lệnh trên
```

### 2.3. Kiểm tra .gitattributes

Đảm bảo file `.gitattributes` có nội dung:
```
*.pkl filter=lfs diff=lfs merge=lfs -text
models/*.pkl filter=lfs diff=lfs merge=lfs -text
```

---

## BƯỚC 3: Commit và Push Code

### 3.1. Tạo .gitignore (nếu chưa có)

Đảm bảo `.gitignore` có:
```
# Model files sẽ được track bởi Git LFS, không ignore
# Nhưng các file khác cần ignore:
__pycache__/
*.py[cod]
*.db
*.sqlite
*.sqlite3
data/user_history.db
.env
*.log
```

### 3.2. Add và Commit Files

```bash
# Add tất cả files (bao gồm .gitattributes)
git add .

# Kiểm tra xem model file có được track bởi LFS không
git lfs ls-files
# Nên thấy: recommendation_model.pkl

# Commit
git commit -m "Initial commit: Steam Games Recommendation System with Git LFS"

# Kiểm tra kích thước commit (model file không nên làm commit lớn)
git log --stat
```

### 3.3. Push lên GitHub

```bash
# Thêm remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# Thay YOUR_USERNAME và YOUR_REPO_NAME bằng thông tin của bạn

# Push lên GitHub
git push -u origin main

# Nếu branch của bạn là master thay vì main:
# git branch -M main  # Đổi tên branch thành main
# git push -u origin main
```

**Lưu ý:**
- Lần push đầu tiên sẽ mất thời gian vì phải upload model file 1.18GB
- Đảm bảo có kết nối internet ổn định
- Có thể mất 10-30 phút tùy tốc độ upload

### 3.4. Kiểm tra trên GitHub

1. Vào repository trên GitHub
2. Kiểm tra file `models/recommendation_model.pkl`:
   - Nếu thấy "Stored with Git LFS" → ✅ Thành công
   - Nếu thấy file lớn bình thường → ❌ Chưa dùng LFS đúng

---

## BƯỚC 4: Deploy lên Streamlit Cloud

### 4.1. Đăng ký/Đăng nhập Streamlit Cloud

1. Truy cập: https://share.streamlit.io
2. Click "Sign in" → Chọn "Continue with GitHub"
3. Authorize Streamlit Cloud truy cập GitHub repositories

### 4.2. Tạo App mới

1. Click "New app"
2. Điền thông tin:
   - **Repository**: Chọn repository vừa tạo
   - **Branch**: `main` (hoặc `master`)
   - **Main file path**: `app/streamlit_app.py`
3. Click "Deploy"

### 4.3. Chờ Deploy

- Streamlit Cloud sẽ:
  1. Clone repository
  2. Cài đặt dependencies từ `requirements.txt`
  3. Tải model file từ Git LFS
  4. Chạy app

- Thời gian: 5-15 phút (tùy kích thước model)

### 4.4. Kiểm tra Logs

1. Vào app settings → "Logs"
2. Kiểm tra:
   - ✅ "Model loaded successfully" → Thành công
   - ❌ "Model not found" → Kiểm tra lại Git LFS
   - ❌ "Out of memory" → Model quá lớn, cần giải pháp khác

---

## BƯỚC 5: Xử lý Lỗi Thường Gặp

### Lỗi 1: "Model not found"

**Nguyên nhân:** Git LFS chưa được cấu hình đúng

**Giải pháp:**
```bash
# Kiểm tra lại Git LFS
git lfs ls-files

# Nếu không thấy model file, track lại:
git lfs track "models/*.pkl"
git add .gitattributes
git add models/recommendation_model.pkl
git commit -m "Fix: Track model with Git LFS"
git push
```

### Lỗi 2: "Out of memory" hoặc App crash

**Nguyên nhân:** Model quá lớn cho Streamlit Cloud free tier (1GB RAM)

**Giải pháp A: Tải Model từ URL (Khuyến nghị)**

1. Upload model lên cloud storage (Google Drive, Dropbox, AWS S3, etc.)
2. Sửa `app/streamlit_app.py`:

```python
@st.cache_resource(show_spinner=False)
def load_model():
    """Load pre-trained model from pickle file or URL"""
    import requests
    import os
    
    model_path = "models/recommendation_model.pkl"
    
    # Nếu model không có local, tải từ URL
    if not os.path.exists(model_path):
        model_url = "https://your-model-url.com/recommendation_model.pkl"
        st.info("📥 Downloading model... This may take a few minutes.")
        
        # Download model
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Progress bar (optional)
                    if total_size > 0:
                        progress = downloaded / total_size
                        # st.progress(progress)
        
        st.success("✅ Model downloaded!")
    
    # Load model như bình thường
    try:
        with open(model_path, 'rb') as f:
            # ... rest of load_model code
```

**Giải pháp B: Giảm kích thước Model**

Trong notebook, khi save model, chỉ lưu lightweight version:
```python
# Chỉ lưu essential data, không lưu full data
model_payload_light = {
    'alpha': recommendation_system.alpha,
    'games_df': recommendation_system.games_df[essential_cols],  # Chỉ essential columns
    # ... other essential fields
}
```

### Lỗi 3: "Module not found"

**Giải pháp:** Kiểm tra `requirements.txt` có đầy đủ:
```
pandas==2.1.4
numpy>=1.24.3,<2.0.0
scikit-learn>=1.3.2,<2.0.0
matplotlib==3.8.2
seaborn==0.13.0
streamlit==1.29.0
requests==2.31.0
scipy==1.11.4
plotly==5.18.0
Pillow==10.1.0
```

### Lỗi 4: Git LFS không hoạt động trên Streamlit Cloud

**Giải pháp:** Streamlit Cloud tự động hỗ trợ Git LFS, nhưng nếu có vấn đề:

1. Kiểm tra `.gitattributes` có trong repo
2. Đảm bảo model file được track: `git lfs ls-files`
3. Thử push lại: `git push --force`

---

## BƯỚC 6: Tối ưu Performance

### 6.1. Caching

App đã có caching:
- `@st.cache_resource` cho model loading
- `@st.cache_data` cho recommendations

### 6.2. Database

SQLite sẽ reset khi app restart. Để lưu trữ lâu dài:

1. Dùng Streamlit Secrets để lưu database URL
2. Kết nối PostgreSQL/MySQL
3. Sửa `src/database.py` để dùng external database

### 6.3. Monitoring

- Kiểm tra logs thường xuyên
- Monitor memory usage
- Kiểm tra response time

---

## 📝 Checklist Trước Khi Deploy

- [ ] Git LFS đã được cài đặt
- [ ] Model file được track: `git lfs ls-files`
- [ ] `.gitattributes` có trong repo
- [ ] `requirements.txt` đầy đủ dependencies
- [ ] `app/streamlit_app.py` là main file
- [ ] Code đã được push lên GitHub
- [ ] Model file hiển thị "Stored with Git LFS" trên GitHub
- [ ] Repository là Public (hoặc đã authorize Streamlit Cloud)

---

## 🎉 Sau Khi Deploy Thành Công

1. **URL công khai**: Streamlit Cloud sẽ cung cấp URL dạng:
   `https://your-app-name.streamlit.app`

2. **Share với người dùng**: URL này có thể share cho bất kỳ ai

3. **Update code**: Mỗi khi push code mới lên GitHub, app sẽ tự động update

---

## 💡 Tips

1. **Test local trước**: Đảm bảo app chạy tốt local trước khi deploy
2. **Kiểm tra logs**: Luôn kiểm tra logs khi có lỗi
3. **Backup model**: Giữ backup model file ở nơi an toàn
4. **Monitor usage**: Streamlit Cloud free tier có giới hạn, monitor usage

---

## 🔗 Links Hữu Ích

- **Git LFS**: https://git-lfs.github.com/
- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io/
- **GitHub**: https://github.com

---

**Chúc bạn deploy thành công! 🚀**

