from constants import embed_model
from index_manager_pinecone import IndexManagerPinecone
import os

def main():
    # 1. Init Manager
    # Đảm bảo tên index khớp với tên bạn đã tạo trên Pinecone (mặc định là arxiv-research)
    index_name = "arxiv-research" 
    print(f"Khoi tao ket noi Pinecone voi index: {index_name}...")
    
    try:
        index_manager = IndexManagerPinecone(embed_model, index_name=index_name)
    except Exception as e:
        print(f"Loi ket noi: {e}")
        return

    # 2. Kiem tra thu muc papers
    if not os.path.exists("papers"):
        os.makedirs("papers")
        print("Da tao thu muc 'papers'. Hay copy file PDF vao day va chay lai script.")
        return

    # Dem so file
    pdf_files = [f for f in os.listdir("papers") if f.lower().endswith('.pdf')]
    file_count = len(pdf_files)
    
    if file_count == 0:
        print("Khong tim thay file PDF nao trong thu muc 'papers/'.")
        print("Vui long copy file vao roi thu lai.")
        return

    print(f"Tim thay {file_count} file PDF: {pdf_files}")
    print("Bat dau nap du lieu vao Pinecone (viec nay co the mat vai phut)...")

    # 3. Tao Index (Ham nay da duoc nang cap de doc folder papers/)
    index_manager.create_index()
    
    print("XONG! Toan bo hang da duoc chuyen len Pinecone.")

if __name__ == "__main__":
    main()
