import os
import pytesseract
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz

DB_FAISS_PATH = 'D:\\Project_2023\\medical_chatbot\\vectorstore\\db_faiss'

class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

def extract_text_from_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(image)

    return text

def extract_text_from_pdf(file_path):
    text = ""

    # Check if the file exists
    if not file_path or not os.path.exists(file_path):
        print("Invalid file path. Exiting.")
        return text

    try:
        # Open the PDF file
        with fitz.open(file_path) as pdf_document:
            # Iterate over pages
            for page_number in range(pdf_document.page_count):
                # Get the page
                page = pdf_document[page_number]

                # Extract text from the page
                text += page.get_text()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")

    return text



def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        file_content = file.read()
    return file_content

def extract_text_from_user_input():
    # Get user input for text
    user_text = input("Enter the text: ")
    return user_text

def create_vector_store(text, embeddings, db_path):
    documents = [SimpleDocument(text, metadata={'file_type': 'text'})]
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_path)


def determine_file_type(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'image'
    elif file_path.lower().endswith('.pdf'):
        return 'pdf'
    elif file_path.lower().endswith('.txt'):
        return 'text'
    else:
        return 'unsupported'

def main():
    # Ask the user for a file path or input text
    user_input = input("Enter 'file' to input a file path or 'text' to input text directly: ")

    if user_input.lower() == 'file':
        # Ask the user for a file path
        file_path = input("Enter the path to the file: ")

        # Ensure the file path is valid
        if not os.path.exists(file_path):
            print("Invalid file path. Exiting.")
            return

        # Determine the file type and extract text accordingly
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
        # Extract text from user input
        text = extract_text_from_user_input()
    else:
        print("Invalid input. Exiting.")
        return

    # Create a vector store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    create_vector_store(text, embeddings, DB_FAISS_PATH)

    print("Vector store created successfully.")

if __name__ == "__main__":
    main()
