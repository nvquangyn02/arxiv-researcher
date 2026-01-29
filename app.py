import streamlit as st
import asyncio
import nest_asyncio
from agent_class import Agent
from index_manager import IndexManager
from constants import GOOGLE_API_KEY
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core.memory import ChatMemoryBuffer

# Patch asyncio để tránh lỗi khi chạy lồng trong môi trường có sẵn event loop
nest_asyncio.apply()

st.set_page_config(page_title="Arxiv Research Agent", page_icon="📚")
st.title("📚 Arxiv Research Agent")

# 1. Caching Resource cho Index (Dữ liệu nặng)
@st.cache_resource
def load_index_data():
    """Chỉ load dữ liệu Index 1 lần để tiết kiệm RAM"""
    try:
        # Dùng gRPC mặc định
        embed_model = GeminiEmbedding(
            api_key=GOOGLE_API_KEY, 
            model_name="models/gemini-embedding-001"
        )
        index_manager = IndexManager(embed_model)
        index = index_manager.retrieve_index()
        return index
    except Exception as e:
        print(f"Index load error: {e}")
        return None

def create_agent(index, memory):
    """Tạo Agent mới mỗi lần run để gắn đúng Event Loop hiện tại"""
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
                # Lấy event loop hiện tại (đã được patch bởi nest_asyncio)
                loop = asyncio.get_event_loop()
                # Chạy task trên loop hiện tại
                answer_text = loop.run_until_complete(agent.chat(prompt))
                
                st.markdown(answer_text)
                
                # Lưu lịch sử UI
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error(f"Error: {str(e)}")
