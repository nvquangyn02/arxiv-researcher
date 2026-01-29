from dotenv import load_dotenv
import os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Load biến môi trường
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Cấu hình Embed Model (Dùng để chuyển văn bản thành Vector)
# Updated: Dùng text-embedding-004 để đảm bảo output là 768 dimensions
embed_model = GeminiEmbedding(
    api_key=GOOGLE_API_KEY, 
    model_name="models/text-embedding-004"
)

# 2. Cấu hình LLM (Model ngôn ngữ chính)
llm_model = Gemini(
    api_key=GOOGLE_API_KEY, 
    model_name="models/gemini-2.5-flash",
    max_tokens=8192, # Tăng giới hạn token đầu ra để tránh lỗi MAX_TOKENS khi tóm tắt văn bản dài
    # transport="rest" # Tạm thời tắt REST để fix lỗi await, quay về gRPC + nest_asyncio
)