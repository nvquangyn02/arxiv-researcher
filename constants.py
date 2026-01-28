from dotenv import load_dotenv
import os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Load biến môi trường
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Cấu hình Embed Model (Dùng để chuyển văn bản thành Vector)
# Model phổ biến: "models/text-embedding-004" hoặc "models/embedding-001"
embed_model = GeminiEmbedding(
    api_key=GOOGLE_API_KEY, 
    model_name="models/gemini-embedding-001"
)

# 2. Cấu hình LLM (Model ngôn ngữ chính)
llm_model = Gemini(
    api_key=GOOGLE_API_KEY, 
    model_name="models/gemini-2.5-flash"
)