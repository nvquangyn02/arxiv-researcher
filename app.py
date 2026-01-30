import streamlit as st
import asyncio
import nest_asyncio
from agent_class import Agent
from index_manager import IndexManager
from constants import GOOGLE_API_KEY, embed_model
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from index_manager_pinecone import IndexManagerPinecone
from llama_index.core.memory import ChatMemoryBuffer

from index_manager_pinecone import IndexManagerPinecone

# Patch asyncio để tránh lỗi khi chạy lồng trong môi trường có sẵn event loop
nest_asyncio.apply()

st.set_page_config(page_title="Arxiv Research Agent", page_icon="📚")
st.title("📚 Arxiv Research Agent")

# 1. Caching Resource cho Index (Dữ liệu nặng)
@st.cache_resource
def load_index_data():
    """Chỉ load dữ liệu Index 1 lần để tiết kiệm RAM"""
    try:
        # Sử dụng embed_model từ constants (đã fix dimension 768)
        index_manager = IndexManagerPinecone(embed_model, "arxiv-research")
        index = index_manager.retrieve_index()
        return index
    except Exception as e:
        print(f"Index load error: {e}")
        return None

def create_agent(index, memory):
    """Tạo Agent mới mỗi lần run để gắn đúng Event Loop hiện tại"""
    # Fix lỗi Loop: Tạo mới LLM instance ngay tại đây thay vì import từ constants
    llm_model = Gemini(
        api_key=GOOGLE_API_KEY, 
        model_name="models/gemini-2.5-flash", 
        max_tokens=8192
    )
    # Truyền memory từ session state vào
    return Agent(index, llm_model, memory=memory)

# 2. Khởi tạo State ban đầu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lưu trữ Chat Memory riêng biệt (không phụ thuộc Agent object)
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=20000)

# --- SIDEBAR: QUẢN LÝ DỮ LIỆU ---
with st.sidebar:
    st.header("📂 Nạp Tài Liệu (PDF)")
    st.write("Tải file PDF lên để dạy cho AI:")
    
    uploaded_files = st.file_uploader(
        "Chọn file PDF", 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Nạp vào Trí Tuệ"):
        with st.spinner("Đang đọc và học tài liệu... (Cứ bình tĩnh nhé)"):
            import os
            
            # 1. Lưu file tạm vào ổ cứng để thư viện đọc được
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            saved_paths = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)
            
            # 2. Gọi IndexManagerPinecone để xử lý
            try:
                # Tạo manager mới để xử lý upload
                # Lưu ý: Lúc này hàm khởi tạo sẽ tạo connection tới Pinecone
                idx_manager = IndexManagerPinecone(embed_model, "arxiv-research")
                success, msg = idx_manager.ingest_uploaded_files(saved_paths)
                
                if success:
                    st.success(f"✅ {msg}")
                    # Clear index cache để lần load sau nó cập nhật dữ liệu mới nếu cần
                    # Tuy nhiên Pinecone là vector store rời, nên query engine sẽ tự tìm thấy data mới.
                else:
                    st.error(f"❌ {msg}")
                    
            except Exception as e:
                st.error(f"Lỗi khi xử lý: {e}")
            
            # 3. Dọn dẹp file tạm
            for p in saved_paths:
                if os.path.exists(p):
                    os.remove(p)
    
    st.divider()
    
    # Nút Xóa Lịch Sử Chat
    if st.button("🗑️ Xóa Lịch Sử Chat", type="primary"):
        st.session_state.messages = []
        st.session_state.chat_memory.reset()
        st.rerun()

# 3. Load Index và Tạo Agent cho Run hiện tại
index = load_index_data()

if index:
    # QUAN TRỌNG: Luôn tạo Agent mới cho mỗi lần chạy script
    # Nhưng dùng lại bộ nhớ cũ (st.session_state.chat_memory)
    agent = create_agent(index, st.session_state.chat_memory)
else:
    st.error("⚠️ Không tìm thấy Index! Hãy chạy file 'build_index.ipynb' để tạo dữ liệu trước.")
    st.stop()

# 4. Hiển thị lịch sử chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Xử lý Input
# Cú pháp walrus (:=) giúp gán giá trị và kiểm tra điều kiện cùng lúc
if prompt := st.chat_input("Ask me anything about research papers!"):
    # Hiển thị câu hỏi User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        # Xử lý câu trả lời Assistant
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # GỌI ASYNC TRỰC TIẾP QUA asyncio.run() ĐÃ PATCH
                # Vì nest_asyncio.apply() đã được gọi ở đầu, ta có thể dùng loop.run_until_complete an toàn
                # Hoặc đơn giản hơn: gọi thẳng hàm chat (bên trong agent class đã có cơ chế gọi)
                
                # Cách 1: Gọi qua event loop hiện tại (An toàn nhất với Streamlit)
                loop = asyncio.get_event_loop()
                answer_text = loop.run_until_complete(agent.chat(prompt))
                
                st.markdown(answer_text)
                
                # Lưu lịch sử UI
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except RuntimeError as e:
                # Nếu loop đã đóng hoặc lỗi loop
                st.error(f"Async Loop Error: {e}")
                # Fallback: Tạo loop mới (ít khi cần nhờ nest_asyncio)
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                answer_text = new_loop.run_until_complete(agent.chat(prompt))
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                import traceback
                st.error(f"Error details: {traceback.format_exc()}")
