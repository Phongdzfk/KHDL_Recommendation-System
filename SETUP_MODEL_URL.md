# 🔧 Cấu hình Model URL cho Streamlit Cloud

## Link Model của bạn:

```
https://drive.google.com/uc?export=download&id=180S_9i5886cn9l9qKCeAmft0otJpEN64
```

---

## Cách 1: Dùng Streamlit Secrets (Khuyến nghị)

### Bước 1: Vào Streamlit Cloud Secrets

1. Vào https://share.streamlit.io
2. Chọn app của bạn
3. Click "Settings" (⚙️) ở góc trên bên phải
4. Click "Secrets" trong menu bên trái

### Bước 2: Thêm Model URL

Trong ô "Secrets", thêm:

```toml
[model]
url = "https://drive.google.com/uc?export=download&id=180S_9i5886cn9l9qKCeAmft0otJpEN64"
```

### Bước 3: Save và Restart

1. Click "Save"
2. App sẽ tự động restart
3. Model sẽ được download từ Google Drive

---

## Cách 2: Hardcode trong Code (Nhanh, nhưng không bảo mật)

Nếu không muốn dùng Secrets, có thể hardcode:

Sửa `app/streamlit_app.py`, thêm ở đầu function `load_model()`:

```python
@st.cache_resource(show_spinner=False)
def load_model():
    """Load pre-trained model from pickle file or URL"""
    model_path = Path('models/recommendation_model.pkl')
    
    # Model URL (fallback if not in secrets)
    MODEL_URL = "https://drive.google.com/uc?export=download&id=180S_9i5886cn9l9qKCeAmft0otJpEN64"
    
    # Check if model URL is provided in secrets
    model_url = MODEL_URL  # Default
    try:
        if hasattr(st, 'secrets') and 'model' in st.secrets and 'url' in st.secrets.model:
            model_url = st.secrets.model.url
    except:
        pass
```

---

## ✅ Kiểm tra:

1. **Test link download:**
   - Mở link trong browser
   - File nên bắt đầu download (1.1GB)
   - Nếu chỉ thấy warning về virus scan → Link vẫn OK, chỉ cần click "Download anyway"

2. **Kiểm tra app:**
   - Restart app trên Streamlit Cloud
   - App sẽ tự động download model từ Google Drive
   - Progress bar sẽ hiển thị tiến trình download

---

## ⚠️ Lưu ý:

1. **Google Drive Rate Limiting:**
   - Nếu quá nhiều download, Google có thể chặn
   - Nên dùng Dropbox hoặc GitHub Releases nếu có vấn đề

2. **Download Time:**
   - File 1.1GB sẽ mất 5-10 phút để download
   - Lần đầu sẽ chậm, sau đó được cache

3. **Virus Warning:**
   - Google Drive sẽ hiển thị warning vì file lớn
   - Code sẽ tự động xử lý, không cần lo

---

## 🔄 Nếu link không hoạt động:

### Thử link khác:

1. **Link với confirm:**
   ```
   https://drive.google.com/uc?export=download&id=180S_9i5886cn9l9qKCeAmft0otJpEN64&confirm=t
   ```

2. **Hoặc dùng gdown (Python library):**
   ```python
   import gdown
   gdown.download("https://drive.google.com/uc?id=180S_9i5886cn9l9qKCeAmft0otJpEN64", "models/recommendation_model.pkl")
   ```

---

## 📝 Checklist:

- [ ] Link Google Drive đã được test (download được)
- [ ] Đã thêm URL vào Streamlit Secrets
- [ ] Code đã được push lên GitHub
- [ ] App đã được restart trên Streamlit Cloud
- [ ] Model đã được download thành công

