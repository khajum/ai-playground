from pathlib import Path
from typing import List, Any
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader, CSVLoader

# Read all the PDFs inside directory './data/pdf/'
def load_all_document(data_dir: str) -> List[Any]:
  """
  Load all supported files from the data directory and convert them to LangChain document structure
  Supported format: PDF, TXT, CSV, Excel, Doc, JSON
  """
  data_path = Path(data_dir).resolve()
  all_documents = []

  #------------------------------------------------------------------------------------
  # Load PDF Files
  #------------------------------------------------------------------------------------
  # Find all the pdf files recursively
  pdf_files = list(data_path.glob('**/*.pdf'))

  # print all the pdf files found in the directory
  print(f"Found {len(list(pdf_files))} pdf files in {data_dir}")

  for pdf_file in pdf_files:
    print(f"\nProcessing: {pdf_file.name}")
    try:
      loader = PyPDFLoader(str(pdf_file))
      documents = loader.load()

      # Add source information to metadata
      for doc in documents:
        doc.metadata['source_file'] = str(pdf_file.name)
        doc.metadata['file_type'] = "pdf"

      all_documents.extend(documents)
      print(f" Processed {pdf_file.name} with {len(documents)} pages")
    except Exception as e:
      print(f" Error processing {pdf_file.name}: {e}")
  
#------------------------------------------------------------------------------------
# Load TXT Files
#------------------------------------------------------------------------------------ 
    text_documents = load_txt_files(data_dir)
    all_documents.extend(text_documents)
#------------------------------------------------------------------------------------

  print(f" Total documents processed: {len(all_documents)}")
  return all_documents

# Process all the PDFs in the data directory
# Call the corrected function
all_pdf_document = load_all_document("./data")
print(f"Successfully processed and collected {len(all_pdf_document)} documents.")


def load_txt_files(data_dir: str):
    # Find all the txt files recursively
    txt_files = list(data_path.glob('**/*.txt'))

    # print all the txt files found in the directory
    print(f"Found {len(list(txt_files))} txt files in {data_dir}")

    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file.name}")
        try:
        loader = PyPDFLoader(str(txt_file))
        documents = loader.load()

        # Add source information to metadata
        for doc in documents:
            doc.metadata['source_file'] = str(txt_file.name)
            doc.metadata['file_type'] = "pdf"

        all_documents.extend(documents)
        print(f" Processed {txt_file.name} with {len(documents)} pages")
        except Exception as e:
        print(f" Error processing {txt_file.name}: {e}")