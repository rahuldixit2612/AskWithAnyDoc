from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_prompt_template():
    """
    Creates a custom prompt template for QA retrieval.

    Returns:
    PromptTemplate: An instance of PromptTemplate with the custom template and input variables.
    """
    custom_prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def create_qa_chain(llm, prompt, db):
    """
    Creates a RetrievalQA chain for question answering.

    Args:
    llm (CTransformers): The language model for question answering.
    prompt (PromptTemplate): The prompt template for constructing queries.
    db (FAISS): The vector store for document retrieval.

    Returns:
    RetrievalQA: An instance of RetrievalQA configured for the given parameters.
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_language_model():
    """
    Loads the language model for question answering.

    Returns:
    CTransformers: An instance of CTransformers representing the loaded language model.
    """
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def initialize_qa_bot():
    """
    Initializes and returns a question answering bot.

    Returns:
    RetrievalQA: An instance of RetrievalQA configured for question answering.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_language_model()
    qa_prompt = create_prompt_template()
    qa = create_qa_chain(llm, qa_prompt, db)
    return qa

def get_final_answer(query):
    """
    Gets the final result from the question answering bot.

    Args:
    query (str): The user's query.

    Returns:
    dict: The result of the question answering process.
    """
    qa_result = initialize_qa_bot()
    response = qa_result({'query': query})
    return response

@cl.on_chat_start
async def start():
    """
    Initializes the chatbot and sends a welcome message.
    """
    chain = initialize_qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the chat bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def handle_user_message(message: cl.Message):
    """
    Handles incoming user messages and processes them using the question answering chain.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: {sources}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
