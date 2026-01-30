from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from tools import download_pdf, fetch_arxiv_papers

class Agent:
    def __init__(self, index, llm_model, memory=None):
        """
        Khởi tạo Agent quản lý quy trình RAG và Tool use.
        
        Args:
            index (VectorStoreIndex): Bộ chỉ mục chứa dữ liệu bài báo (đã được load).
            llm_model (Gemini): Mô hình LLM để suy luận.
            memory (ChatMemoryBuffer): Bộ nhớ hội thoại (optional).
        """
        self.index = index
        self.llm_model = llm_model
        
        # Tạo bộ nhớ để lưu lịch sử hội thoại (do Workflow Agent là stateless)
        # Nếu được truyền vào thì dùng, không thì tạo mới
        self.memory = memory if memory else ChatMemoryBuffer.from_defaults(token_limit=20000)
        
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
        
        # Định nghĩa System Prompt (Chỉ dẫn hành vi cốt lõi cho Agent)
        # Giúp Agent biết ưu tiên tìm trong DB trước, rồi mới lên Arxiv, và không tự tiện tải file.
        system_prompt = """You are an AI Researcher Agent composed of a RAG engine and several tools.
        
        RULES:
        1. When asked about a topic, FIRST query the 'research_paper_query_tool' to check if you already have information in your local database.
        2. IF relevant papers are found locally, use them to answer.
        3. IF NOT found locally (or if the user specifically asks for *new* papers), use 'fetch_from_arxiv' to get new papers.
        4. IMPORTANT: Do NOT use 'download_pdf_file_tool' unless the user strictly commands you to "download" or "save" the papers.
        5. Always provide the Title, Summary, and Authors when introducing a paper.
        """
        
        # Sử dụng class constructor trực tiếp
        self.agent = ReActAgent(
            tools=[self.pdf_download_tool, self.rag_tool, self.fetch_arxiv_tool],
            llm=self.llm_model,
            verbose=True,
            streaming=False,     # Tắt streaming để tránh lỗi với Gemini
            system_prompt=system_prompt  # Truyền chỉ dẫn vào Agent
        )

    async def chat(self, message: str):
        """
        Gửi tin nhắn đến Agent và nhận câu trả lời (Bất đồng bộ).
        """
        # Sử dụng .run() và truyền memory vào để Agent nhớ ngữ cảnh (Logic cũ của bạn)
        response = await self.agent.run(
            user_msg=message,
            memory=self.memory,
            max_iterations=10    # Giới hạn số bước suy luận để tránh loop vô hạn
        )
        return str(response)