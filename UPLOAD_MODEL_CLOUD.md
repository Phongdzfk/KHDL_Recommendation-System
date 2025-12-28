# ☁️ Hướng dẫn Upload Model lên Cloud Storage

## Tại sao cần upload lên cloud?

- Git LFS có thể không hoạt động tốt trên Streamlit Cloud
- File 1.18GB quá lớn cho Git LFS
- Upload lên cloud storage và tải về khi cần sẽ ổn định hơn

---

## Các phương án Upload:

### Option 1: Google Drive (Dễ nhất, Miễn phí)

#### Bước 1: Upload lên Google Drive

1. Vào https://drive.google.com
2. Tạo folder mới (ví dụ: "Steam-Game-Model")
3. Upload file `recommendation_model.pkl` vào folder
4. Right-click file → "Get link" → Chọn "Anyone with the link"
5. Copy link (sẽ có dạng: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)

#### Bước 2: Lấy Direct Download Link

Link Google Drive cần convert sang direct download link:

**Cách 1: Dùng công thức**
```
https://drive.google.com/uc?export=download&id=FILE_ID
```

Trong đó `FILE_ID` là phần ID trong link (giữa `/d/` và `/view`)

**Cách 2: Dùng tool online**
- https://sites.google.com/site/gdocs2direct/
- Paste link Google Drive → Get direct link

#### Bước 3: Sửa code để tải từ URL

Xem file `app/streamlit_app.py` đã được cập nhật với function `load_model_from_url()`

---

### Option 2: Dropbox (Dễ, Miễn phí)

#### Bước 1: Upload lên Dropbox

1. Vào https://www.dropbox.com
2. Upload file `recommendation_model.pkl`
3. Right-click file → "Copy link"
4. Link sẽ có dạng: `https://www.dropbox.com/s/xxxxx/recommendation_model.pkl?dl=0`

#### Bước 2: Convert sang Direct Link

Thay `?dl=0` bằng `?dl=1`:
```
https://www.dropbox.com/s/xxxxx/recommendation_model.pkl?dl=1
```

---

### Option 3: AWS S3 (Chuyên nghiệp, Có thể tốn phí)

#### Bước 1: Tạo S3 Bucket

1. Vào AWS Console → S3
2. Create bucket
3. Upload file
4. Set public access (hoặc dùng signed URL)

#### Bước 2: Lấy URL

```
https://BUCKET_NAME.s3.REGION.amazonaws.com/recommendation_model.pkl
```

---

### Option 4: GitHub Releases (Miễn phí, Dễ)

#### Bước 1: Tạo Release

1. Vào repository trên GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0`
4. Upload `recommendation_model.pkl` như asset
5. Publish release

#### Bước 2: Lấy Download URL

```
https://github.com/USERNAME/REPO/releases/download/v1.0.0/recommendation_model.pkl
```

---

## 🔧 Cách sử dụng trong Code:

### Cách 1: Dùng Streamlit Secrets (Khuyến nghị)

1. Vào Streamlit Cloud → App Settings → Secrets
2. Thêm:
```toml
[model]
url = "https://your-direct-download-url.com/recommendation_model.pkl"
```

3. Code sẽ tự động đọc từ secrets

### Cách 2: Hardcode URL (Nhanh, nhưng không bảo mật)

Sửa trong `app/streamlit_app.py`:
```python
MODEL_URL = "https://your-direct-download-url.com/recommendation_model.pkl"
```

---

## 📝 Checklist:

- [ ] Model file đã được upload lên cloud storage
- [ ] Đã có direct download link
- [ ] Đã test link download được
- [ ] Đã cập nhật code với URL
- [ ] Đã push code lên GitHub
- [ ] Đã restart app trên Streamlit Cloud

---

## 💡 Tips:

1. **Google Drive**: Dễ nhất, nhưng có thể bị rate limit
2. **Dropbox**: Ổn định, dễ dùng
3. **AWS S3**: Chuyên nghiệp nhất, nhưng cần setup
4. **GitHub Releases**: Miễn phí, nhưng file lớn có thể chậm

---

## ⚠️ Lưu ý:

- **File lớn**: 1.18GB có thể mất 5-10 phút để download
- **Rate limiting**: Một số service có giới hạn download
- **Cost**: AWS S3 có thể tốn phí nếu traffic lớn
- **Security**: Không hardcode sensitive URLs trong code

