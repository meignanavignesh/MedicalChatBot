import os, re, string
import nltk, spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# -----------------------------
# INITIAL SETUP
# -----------------------------
load_dotenv()
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nlp = spacy.load("en_core_web_sm")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss_sequential"
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# -----------------------------
# SEQUENTIAL NLP PREPROCESSING PIPELINE
# -----------------------------
def preprocess_full_pipeline(text):
    """
    Sequentially applies multiple NLP cleaning and normalization steps
    to produce an optimized text representation.
    """
    # 1️⃣ Lowercase and basic cleaning
    text = re.sub(r"[^a-z\s]", " ", text.lower())

    # 2️⃣ Tokenization
    tokens = nltk.word_tokenize(text)

    # 3️⃣ Stopword Removal
    tokens = [w for w in tokens if w not in stop_words]

    # 4️⃣ Lemmatization (WordNet)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # 5️⃣ Stemming (Porter)
    tokens = [stemmer.stem(w) for w in tokens]

    # 6️⃣ POS-based Lemmatization (spaCy)
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]

    # Final clean output
    processed_text = " ".join(tokens)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    return processed_text


# -----------------------------
# LOAD PDF AND SPLIT INTO CHUNKS
# -----------------------------
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from your book(s).")
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks for embedding.")
    return chunks


# -----------------------------
# EMBEDDING MODEL (Medical-Tuned)
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)


# -----------------------------
# MAIN WORKFLOW
# -----------------------------
def main():
    print("\n🧠 Starting Sequential Preprocessing FAISS Builder...")
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)

    # Sequentially preprocess all chunks
    print("\n🔄 Running sequential preprocessing on text chunks...")
    processed_docs = []
    for doc in tqdm(text_chunks, desc="Processing"):
        cleaned_text = preprocess_full_pipeline(doc.page_content)
        processed_docs.append(Document(page_content=cleaned_text))

    # Create FAISS vector store
    print("\n⚙️ Building FAISS vector database using sequentially processed text...")
    db = FAISS.from_documents(processed_docs, embedding_model)

    # Save optimized FAISS
    db.save_local(DB_FAISS_PATH)
    print(f"\n✅ Successfully saved optimized FAISS store at: {DB_FAISS_PATH}")
    print("✨ This store uses fully sequential preprocessing for maximum semantic accuracy.")

if __name__ == "__main__":
    main()
