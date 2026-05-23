import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_LIBRARY = "default"


def sanitize_library(name: str | None) -> str:
    """Pinecone namespaces accept arbitrary strings, but we restrict to a-z0-9_-
    and lowercase for predictable user-facing labels."""
    if not name:
        return DEFAULT_LIBRARY
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip()).lower()
    return cleaned[:48] or DEFAULT_LIBRARY


pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="dotproduct",
        spec=spec,
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index = pc.Index(PINECONE_INDEX_NAME)


def list_libraries() -> list[str]:
    """Return the list of namespaces (libraries) currently in the index."""
    try:
        stats = index.describe_index_stats()
        ns = (stats or {}).get("namespaces") or {}
        libs = sorted([k for k in ns.keys() if k])
        if DEFAULT_LIBRARY not in libs:
            libs = [DEFAULT_LIBRARY] + libs
        return libs
    except Exception:
        return [DEFAULT_LIBRARY]


def load_vectorstore(uploaded_files, library: str | None = None):
    namespace = sanitize_library(library)
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", output_dimensionality=768
    )
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            # Store the chunk text in metadata so retrieval surfaces it back.
            meta["text"] = chunk.page_content
            meta["library"] = namespace
            meta["source"] = meta.get("source") or Path(file_path).name
            metadatas.append(meta)
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks for library '{namespace}'...")
        embeddings = embed_model.embed_documents(texts)

        print(f"📤 Uploading to Pinecone namespace '{namespace}'...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(
                vectors=list(zip(ids, embeddings, metadatas)),
                namespace=namespace,
            )
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path} → library '{namespace}'")

    return namespace
