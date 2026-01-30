from index_manager import IndexManager
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
load_dotenv()


class IndexManagerPinecone(IndexManager):
    def __init__(self, embed_model, index_name):
        super().__init__(embed_model)
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.pinecone_index = pc.Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def create_index(self):
        self.documents = [] # Reset danh sách cũ nếu có
        self.create_documents_from_papers()
        
        # Thêm phần đọc file local từ thư mục papers/
        if os.path.exists("papers"):
            try:
                print("Dang doc file tu folder papers/...")
                # SimpleDirectoryReader tự động đọc pdf, txt, docx...
                reader = SimpleDirectoryReader("papers")
                local_docs = reader.load_data()
                print(f"Tim thay {len(local_docs)} trang tai lieu.")
                self.documents.extend(local_docs)
            except Exception as e:
                print(f"Co loi khi doc file local: {e}")

        if not self.documents:
            print("Khong co tai lieu nao de nap vao Index.")
            return

        Settings.chunk_size = 1024
        Settings.chunk_overlap = 50
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )

    def ingest_uploaded_files(self, file_paths):
        """
        Doc va nap truc tiep danh sach file (tu upload) vao Pinecone
        """
        try:
            print(f"Dang xu ly {len(file_paths)} file upload...")
            # Doc noi dung file
            reader = SimpleDirectoryReader(input_files=file_paths)
            new_documents = reader.load_data()
            
            if not new_documents:
                return False, "Khong doc duoc noi dung file na."

            print(f"Tim thay {len(new_documents)} trang tai lieu. Dang day len Pinecone...")

            # Settings chunk text
            Settings.chunk_size = 1024
            Settings.chunk_overlap = 50
            
            # Nap vao Index hien co (Append mode)
            VectorStoreIndex.from_documents(
                new_documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            return True, f"Da nap thanh cong {len(file_paths)} file voi {len(new_documents)} trang tai lieu!"
        except Exception as e:
            return False, f"Gap loi khi nap file: {str(e)}"
    def retrieve_index(self):
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )