import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag_utils_ocr import process_pdf_to_vectorstore

load_dotenv()

st.set_page_config(page_title="My Free RAG Bot", page_icon="🤖")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY!")
    st.stop()

st.title("🤖 Chat with your PDF (Free & Fast)")
st.sidebar.header("Settings & Support")

# Show PDF upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")


# Initialize vectorstore and qa_chain
vectorstore = None
qa_chain = None

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# تصميم الـ Prompt يدوياً (أسرع وأضمن من تحميله من الإنترنت كل مرة)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing PDF..."):
        try:
            vectorstore = process_pdf_to_vectorstore("temp.pdf")
            retriever = vectorstore.as_retriever()
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # بناء الـ Chain بالطريقة الحديثة (LCEL)
            qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.sidebar.success("✅ PDF Indexed Successfully!")
        except ValueError as e:
            error_msg = str(e)
            st.sidebar.error(error_msg)
            
            if "EXTRACTION FAILED" in error_msg:
                st.sidebar.warning("""
🛠️ **Troubleshooting:**

**Current Setup:**
- ✅ Tesseract OCR installed 
- ✅ PyMuPDF & pdf2image ready
- ⚠️ Arabic language data NOT installed (only English)

**If Your PDF is in Arabic:**
To install Arabic support:
1. Download: `ara.traineddata` from:
   https://github.com/UB-Mannheim/tesseract/tree/master/tessdata
2. Save to: `C:\\Program Files\\Tesseract-OCR\\tessdata\\`
3. Restart the app

**Quick Fix:**
- Use **Google Docs** → Download as PDF (works great!)
- Or try an **online converter** like ILovePDF or SmallPDF
                """)
            else:
                st.sidebar.info("💡 Please try another PDF")
            vectorstore = None
            qa_chain = None
        except Exception as e:
            st.sidebar.error(f"❌ Unexpected error: {str(e)}")
            vectorstore = None
            qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if uploaded_file and vectorstore is not None:
            # هنا بننادي الـ qa_chain اللي عملناه فوق
            try:
                response = qa_chain.invoke(user_input)
            except Exception as e:
                response = f"❌ Error generating response: {str(e)}"
        else:
            response = "❌ Please upload a valid PDF first!"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})