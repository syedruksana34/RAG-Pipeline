import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

# Core Code
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """ 
Answer the question based only on the following context: 
{context}
---
Answer the question based on the above context: {question}
"""



def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Prepare the context and prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Get response from the model
    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    return response_text, sources

# Streamlit App
def main():
    # Add logo at the top left corner
    logo_path = "aisensum.png"  # Replace with your logo file name or path
    st.image(logo_path, width=50)  # Adjust width as needed

    st.title("RAGVance AI Chatbot")
    st.write("Ask questions and get AI-generated responses based on your data!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your query?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the query
        with st.spinner("AI is thinking..."):
            try:
                response, sources = query_rag(prompt)
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Display sources
                if sources:
                    sources_text = f"Sources: {', '.join(sources)}"
                else:
                    sources_text = "No sources found."
                st.chat_message("assistant").markdown(sources_text)
                st.session_state.messages.append({"role": "assistant", "content": sources_text})

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()