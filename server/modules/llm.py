from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are **MediBot**, an AI-powered medical assistant helping doctors and healthcare professionals.

---

📄 **Uploaded Document Context** (may be empty if no PDFs uploaded):
{context}

🙋 **User Question**:
{question}

---

💬 **Instructions**:
- If the context contains relevant information, use it as your primary source and cite it.
- If the context is empty or not relevant to the question, answer using your general medical knowledge — you are trained on medical literature and can answer standard clinical questions.
- For well-established medical facts (disease symptoms, drug classes, anatomy, physiology), always provide a helpful answer even without document context.
- Only say you cannot answer if the question is about a specific private document or a highly specialized topic you genuinely don't have knowledge about.
- Respond in a calm, factual, professional tone.
- Do NOT fabricate specific statistics, dosages, or trial results without a source.
- Do NOT provide final diagnoses for individual patients.
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )