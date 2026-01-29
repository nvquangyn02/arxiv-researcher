from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from tools import download_pdf, fetch_arxiv_papers

class Agent:
    def __init__(self, index, llm_model):
        """
        Khởi tạo Agent quản lý quy trình RAG và Tool use.
        
        Args:
            index (VectorStoreIndex): Bộ chỉ mục chứa dữ liệu bài báo (đã được load).
            llm_model (Gemini): Mô hình LLM để suy luận.
        """
        self.index = index
        self.llm_model = llm_model
        
        # Tạo bộ nhớ để lưu lịch sử hội thoại (do Workflow Agent là stateless)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
        
        # Xây dựng các thành phần của Agent
        self.build_query_engine()
        self.build_rag_tool()
        self.build_pdf_download_tool()
        self.build_fetch_arxiv_tool()
        self.build_agent()

    def build_query_engine(self):
        """Tạo 'Động cơ tìm kiếm' từ Index để tra cứu thông tin."""
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_model,
            similarity_top_k=5,  # Lấy 5 tài liệu liên quan nhất mỗi lần tìm
        )

    def build_rag_tool(self):
        """Đóng gói Query Engine thành một Tool để Agent sử dụng."""
        # Lưu vào biến rag_tool (khác với tên hàm để tránh lỗi)
        self.rag_tool = QueryEngineTool.from_defaults(
            self.query_engine,
            name="research_paper_query_tool",
            description="A RAG engine with recent research papers.",
        )

    def build_pdf_download_tool(self):
        """Tạo Tool cho phép Agent tải file PDF từ link."""
        self.pdf_download_tool = FunctionTool.from_defaults(
            download_pdf,
            name="download_pdf_file_tool",
            description="python function that downloads a PDF file by link.",
        )

    def build_fetch_arxiv_tool(self):  
        """Tạo Tool cho phép Agent tìm kiếm bài báo trên Arxiv."""
        self.fetch_arxiv_tool = FunctionTool.from_defaults(
            fn=fetch_arxiv_papers, # Dùng hàm import từ tools.py
            name="fetch_from_arxiv",
            description="download the {max_results} recent papers regarding the {topic} from arxiv",
        )

    def build_agent(self):
        """Lắp ráp và khởi tạo ReAct Agent."""
        # Sử dụng class constructor trực tiếp (thay vì .from_tools cũ)
        self.agent = ReActAgent(
            tools=[self.pdf_download_tool, self.rag_tool, self.fetch_arxiv_tool],
            llm=self.llm_model,
            verbose=True,
            streaming=False # Tắt streaming để tránh lỗi với Gemini
        )

    async def chat(self, message: str):
        """
        Gửi tin nhắn đến Agent và nhận câu trả lời (Bất đồng bộ).
        """
        # Sử dụng .run() và truyền memory vào để Agent nhớ ngữ cảnh
        return await self.agent.run(
            user_msg=message,
            memory=self.memory
        )