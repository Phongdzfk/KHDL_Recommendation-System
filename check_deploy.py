"""
Script kiểm tra sẵn sàng deploy
"""
import os
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_file_exists(filepath, description):
    """Kiểm tra file có tồn tại không"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_file_size(filepath, max_size_mb=100):
    """Kiểm tra kích thước file"""
    if not os.path.exists(filepath):
        return False
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    status = "⚠️" if size_mb > max_size_mb else "✅"
    print(f"{status} File size: {size_mb:.2f} MB")
    return size_mb

def main():
    print("=" * 60)
    print("🚀 KIỂM TRA SẴN SÀNG DEPLOY")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 1. Kiểm tra các file cần thiết
    print("📁 Kiểm tra files:")
    print("-" * 60)
    
    files_to_check = [
        ("app/streamlit_app.py", "Main app file"),
        ("src/database.py", "Database module"),
        ("requirements.txt", "Dependencies"),
        (".gitattributes", "Git LFS config"),
        ("models/recommendation_model.pkl", "Model file"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_ok = False
    
    print()
    
    # 2. Kiểm tra kích thước model
    print("📊 Kiểm tra kích thước model:")
    print("-" * 60)
    model_path = "models/recommendation_model.pkl"
    if os.path.exists(model_path):
        size_mb = check_file_size(model_path, max_size_mb=100)
        if size_mb > 100:
            print("⚠️  Model file lớn hơn 100MB - CẦN DÙNG GIT LFS!")
            print("   Chạy: git lfs track 'models/*.pkl'")
    print()
    
    # 3. Kiểm tra Git LFS
    print("🔧 Kiểm tra Git LFS:")
    print("-" * 60)
    try:
        import subprocess
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git LFS installed: {result.stdout.strip()}")
        else:
            print("❌ Git LFS chưa được cài đặt")
            print("   Download từ: https://git-lfs.github.com/")
            all_ok = False
    except FileNotFoundError:
        print("❌ Git chưa được cài đặt")
        all_ok = False
    except Exception as e:
        print(f"⚠️  Không thể kiểm tra Git LFS: {e}")
    
    # Kiểm tra .gitattributes
    if os.path.exists(".gitattributes"):
        with open(".gitattributes", "r") as f:
            content = f.read()
            if "filter=lfs" in content:
                print("✅ .gitattributes đã cấu hình Git LFS")
            else:
                print("⚠️  .gitattributes chưa có cấu hình Git LFS")
    print()
    
    # 4. Kiểm tra requirements.txt
    print("📦 Kiểm tra requirements.txt:")
    print("-" * 60)
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            required_packages = [
                "streamlit",
                "pandas",
                "numpy",
                "scikit-learn",
                "scipy",
            ]
            for pkg in required_packages:
                if pkg.lower() in requirements.lower():
                    print(f"✅ {pkg}")
                else:
                    print(f"❌ {pkg} - THIẾU!")
                    all_ok = False
    print()
    
    # 5. Kiểm tra Git repository
    print("🔍 Kiểm tra Git repository:")
    print("-" * 60)
    if os.path.exists(".git"):
        print("✅ Git repository đã được khởi tạo")
        
        # Kiểm tra remote
        try:
            result = subprocess.run(['git', 'remote', '-v'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                print("✅ Remote repository đã được cấu hình")
                print(f"   {result.stdout.strip()}")
            else:
                print("⚠️  Chưa có remote repository")
                print("   Chạy: git remote add origin <your-repo-url>")
        except:
            pass
    else:
        print("⚠️  Git repository chưa được khởi tạo")
        print("   Chạy: git init")
        all_ok = False
    print()
    
    # 6. Tóm tắt
    print("=" * 60)
    if all_ok:
        print("✅ TẤT CẢ ĐÃ SẴN SÀNG!")
        print()
        print("📝 Các bước tiếp theo:")
        print("1. git add .")
        print("2. git commit -m 'Initial commit'")
        print("3. git push -u origin main")
        print("4. Deploy trên https://share.streamlit.io")
    else:
        print("⚠️  CẦN SỬA MỘT SỐ VẤN ĐỀ TRƯỚC KHI DEPLOY")
        print()
        print("📖 Xem hướng dẫn chi tiết trong: DEPLOY_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

