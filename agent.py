import os
import arxiv
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# 1. Load biến môi trường
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  Cảnh báo: Chưa thấy GOOGLE_API_KEY trong file .env")

# 2. Định nghĩa công cụ tìm kiếm Arxiv
def search_arxiv(query: str, max_results: int = 3):
    """
    Tìm kiếm các bài báo khoa học trên Arxiv.
    Args:
        query: Chủ đề hoặc từ khóa tìm kiếm.
        max_results: Số lượng bài báo tối đa trả về.
    """
    print(f"\n... Đang tìm kiếm Arxiv với từ khóa: '{query}' ...")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for result in search.results():
        results.append(f"Title: {result.title}\nSummary: {result.summary}\nURL: {result.entry_id}\n---")
    
    return "\n".join(results)

# Chuyển đổi hàm python thành Tool để Agent hiểu
arxiv_tool = FunctionTool.from_defaults(fn=search_arxiv)

# 3. Khởi tạo LLM (Gemini)
# Lưu ý: model_name có thể là "models/gemini-1.5-flash" hoặc "models/gemini-pro" tùy key của bạn
llm = Gemini(model_name="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)

# 4. Khởi tạo Agent
agent = ReActAgent.from_tools([arxiv_tool], llm=llm, verbose=True)

# 5. Chạy thử
if __name__ == "__main__":
    print("🤖 Agent Arxiv Researcher sẵn sàng! (Gõ 'exit' để thoát)")
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            response = agent.chat(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"Lỗi: {e}")
