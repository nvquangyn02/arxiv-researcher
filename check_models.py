import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Lỗi: Chưa tìm thấy GOOGLE_API_KEY trong file .env")
else:
    genai.configure(api_key=api_key)
    print(f"🔑 Đang kiểm tra với Key: {api_key[:5]}...{api_key[-5:]}")
    print("\n--- DANH SÁCH MODEL ĐƯỢC PHÉP DÙNG ---")
    try:
        found_any = False
        for m in genai.list_models():
            found_any = True
            print(f"- {m.name} | Methods: {m.supported_generation_methods}")
        
        if not found_any:
            print("⚠️ Key này hợp lệ nhưng không tìm thấy model nào. Có thể do Region hoặc cấu hình Project.")
    except Exception as e:
        print(f"❌ Lỗi khi gọi API: {e}")
