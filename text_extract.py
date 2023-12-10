import os
import pytesseract
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz

DB_FAISS_PATH = 'D:\\Project_2023\\chatbot\\vectorstore\\db_faiss'

class SimpleDocument:
    """A simple document class."""
    def __init__(self, page_content, metadata=None):
        """
        Initialize a SimpleDocument.

        Parameters:
        - page_content (str): The content of the document.
        - metadata (dict): Additional metadata for the document.
        """
        self.page_content = page_content
        self.metadata = metadata

def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - str: The extracted text.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.

    Parameters:
    - file_path (str): The path to the PDF file.

    Returns:
    - str: The extracted text.
    """
    text = ""

    if not file_path or not os.path.exists(file_path):
        print("Invalid file path. Exiting.")
        return text

    try:
        with fitz.open(file_path) as pdf_document:
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                text += page.get_text()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")

    return text

def extract_text_from_txt(txt_path):
    """
    Extract text from a text file.

    Parameters:
    - txt_path (str): The path to the text file.

    Returns:
    - str: The extracted text.
    """
    with open(txt_path, 'r') as file:
        file_content = file.read()
    return file_content

def extract_text_from_user_input():
    """
    Extract text from user input.

    Returns:
    - str: The user-inputted text.
    """
    user_text = input("Enter the text: ")
    return user_text

def create_vector_store(text, embeddings, db_path):
    """
    Create and save a vector store from text using embeddings.

    Parameters:
    - text (str): The input text.
    - embeddings: An instance of the embeddings class.
    - db_path (str): The path to save the vector store.
    """
    documents = [SimpleDocument(text, metadata={'file_type': 'text'})]
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)

def determine_file_type(file_path):
    """
    Determine the file type based on the file extension.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - str: The file type ('image', 'pdf', 'text', 'unsupported').
    """
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'image'
    elif file_path.lower().endswith('.pdf'):
        return 'pdf'
    elif file_path.lower().endswith('.txt'):
        return 'text'
    else:
        return 'unsupported'

def main():
    """
    Main function to interact with the user and create a vector store.
    """
    user_input = input("Enter 'file' to input a file path or 'text' to input text directly: ")

    if user_input.lower() == 'file':
        file_path = input("Enter the path to the file: ")

        if not os.path.exists(file_path):
            print("Invalid file path. Exiting.")
            return

        file_type = determine_file_type(file_path)
        if file_type == 'image':
            text = extract_text_from_image(file_path)
        elif file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_type == 'text':
            text = extract_text_from_txt(file_path)
        else:
            print("Unsupported file type. Exiting.")
            return
    elif user_input.lower() == 'text':
        text = extract_text_from_user_input()
    else:
        print("Invalid input. Exiting.")
        return

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    create_vector_store(text, embeddings, DB_FAISS_PATH)

    print("Vector store created successfully.")

if __name__ == "__main__":
    main()
