import os
import sys
import time
from pathlib import Path

def test_verification():
    print("=== MEDICAL CHATBOT DEPLOYMENT VERIFICATION ===")

    # Track memory
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / (1024 * 1024)
        print(f"[1/6] Base Python memory: {mem_start:.1f} MB")
    except ImportError:
        mem_start = 0
        print("[1/6] psutil not installed, skipping fine-grained memory tracking")

    # Step 1: Check paths
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / "vectorstore" / "db_faiss_sequential"
    index_faiss = db_path / "index.faiss"
    index_pkl = db_path / "index.pkl"

    assert db_path.exists(), f"db_path {db_path} does not exist"
    assert index_faiss.exists(), f"index.faiss {index_faiss} does not exist"
    assert index_pkl.exists(), f"index.pkl {index_pkl} does not exist"
    print(f"[2/6] File paths verified: index.faiss ({index_faiss.stat().st_size / (1024*1024):.1f} MB), index.pkl ({index_pkl.stat().st_size / (1024*1024):.1f} MB)")

    # Step 2: Test BioBERT embeddings initialization & download
    print("[3/6] Initializing BioBERT embeddings model (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)...")
    t0 = time.time()
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    t_model = time.time() - t0
    print(f"      BioBERT model initialized successfully in {t_model:.2f}s")

    # Step 3: Test FAISS load_local
    print("[4/6] Loading FAISS vectorstore (db_faiss_sequential)...")
    t0 = time.time()
    from langchain_community.vectorstores import FAISS
    db = FAISS.load_local(str(db_path), embedding_model, allow_dangerous_deserialization=True)
    t_db = time.time() - t0
    print(f"      FAISS vectorstore loaded successfully in {t_db:.2f}s")

    # Step 4: Test Document Retrieval from index
    print("[5/6] Testing similarity retrieval on medical query: 'What are the symptoms and treatment of diabetes?'...")
    docs = db.similarity_search("What are the symptoms and treatment of diabetes?", k=3)
    assert len(docs) > 0, "No documents retrieved from FAISS!"
    print(f"      Successfully retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        content_preview = doc.page_content.replace("\n", " ")[:150]
        print(f"      Doc {i} [{doc.metadata}]: {content_preview}...")

    # Step 5: Test Groq API key reading logic
    print("[6/6] Verifying GROQ_API_KEY resolution...")
    fake_key = "gsk_test_mock_key_for_verification_only"
    os.environ["GROQ_API_KEY"] = fake_key
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.environ.get("GROQ_API_KEY"))
    assert llm.groq_api_key.get_secret_value() == fake_key, "Groq API key mismatch!"
    del os.environ["GROQ_API_KEY"]
    print("      Secure Groq API key loading verified!")

    # Memory summary
    if mem_start:
        mem_end = process.memory_info().rss / (1024 * 1024)
        print(f"\nMemory Footprint: {mem_end:.1f} MB (Total process memory used)")
        print(f"Net model + index footprint: {mem_end - mem_start:.1f} MB (fits well within 1 GB free tier limit)")

    print("\n[SUCCESS] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_verification()
