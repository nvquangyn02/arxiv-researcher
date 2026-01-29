from tools import fetch_arxiv_papers
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core import StorageContext,load_index_from_storage
class IndexManager:
    def __init__(self, embed_model):
        """
        Khởi tạo IndexManager.
        
        Args:
            embed_model: Mô hình Embeddings (ví dụ: GeminiEmbedding) dùng để mã hóa văn bản.
        """
        self.embed_model = embed_model
        self.papers = []      # Lưu trữ danh sách bài báo dạng raw dictionary
        self.documents = []   # Lưu trữ danh sách Document object của LlamaIndex
        self.index = None     # Lưu trữ VectorStoreIndex

    def fetch_papers(self, topic, paper_count=10):
        """
        Tìm kiếm và tải thông tin bài báo từ Arxiv về bộ nhớ đệm.
        
        Args:
            topic (str): Chủ đề cần tìm kiếm.
            paper_count (int): Số lượng bài báo muốn lấy.
        """
        self.papers = fetch_arxiv_papers(topic, paper_count)
    
    def create_documents_from_papers(self):
        """
        Chuyển đổi dữ liệu bài báo thô (dicts) thành các đối tượng Document.
        Mỗi bài báo sẽ được gom các trường (Tiêu đề, Tác giả, Tóm tắt...) thành một chuỗi văn bản.
        """
        self.documents = [] # Reset danh sách cũ nếu có
        
        for paper in self.papers:
            # Gom các thông tin quan trọng vào một chuỗi văn bản duy nhất để model dễ đọc
            content = (
                f"Title: {paper['title']}\n"
                f"Authors: {', '.join(paper['authors'])}\n"
                f"Abstract: {paper['summary']}\n"
                f"URL: {paper.get('pdf_url', 'N/A')}\n"
                f"DOI: {paper.get('doi', 'N/A')}\n"
                f"Primary Category: {paper['primary_category']}\n"
                f"arXiv URL: {paper['arxiv_url']}\n"
            )
            # Tạo Document và thêm vào danh sách (Sửa lỗi indentation cũ)
            self.documents.append(Document(text=content))
    
    def create_index(self):
        """
        Tạo mới VectorStoreIndex từ danh sách bài báo hiện có.
        Quá trình này sẽ chia nhỏ văn bản (chunking) và vector hóa chúng.
        """
        # Bước 1: Chuẩn bị documents
        self.create_documents_from_papers()
        
        # Bước 2: Cấu hình chunking (chia nhỏ văn bản)
        Settings.chunk_size = 1024  # Kích thước mỗi đoạn văn bản (chunk) lớn hơn mặc định để giữ trọn vẹn ngữ cảnh
        Settings.chunk_overlap = 50 # Độ chồng lặp giữa các chunk để không mất thông tin ở biên
        
        # Bước 3: Tạo Index
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            embed_model=self.embed_model
        )

    def retrieve_index(self, persist_dir="index/"):
        """
        Khôi phục (Load) Index đã lưu từ ổ cứng lên RAM.
        
        Args:
            persist_dir (str): Đường dẫn thư mục chứa index đã lưu.
        Returns:
            VectorStoreIndex: Index đã được khôi phục.
        """
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context, embed_model=self.embed_model)
    
    def list_papers(self):
        """
        In ra danh sách tiêu đề các bài báo đang được quản lý.
        """
        print([paper["title"] for paper in self.papers])