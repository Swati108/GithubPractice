import streamlit as st
from app.rag_pipeline import initialize_rag_pipeline

st.title(" RAG-based Document Q&A System")
qa_chain = initialize_rag_pipeline()

query = st.text_input("Ask a question based on the internal documents:")

if query:
    with st.spinner("Retrieving answer..."):
        result = qa_chain(query)
        st.subheader("Answer:")
        st.write(result['result'])

        st.subheader("Sources:")
        for doc in result['source documents']:
            st.markdown(f"• *{doc.metadata.get('source', 'Unknown')}*")
