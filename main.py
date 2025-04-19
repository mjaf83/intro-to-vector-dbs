from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    print("Ingesting data...")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    query = "What is Pinecone in machine learning?"
    #chain = PromptTemplate.from_template(template=query) | llm
 

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke({"input": query})
    print(result)

    template = """Use the following piece of context to answer the question at the end.
    If you don't know the answer, just say "I don't know", don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Allways "say thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Heppful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)