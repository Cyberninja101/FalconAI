from langchain.document_loaders import PyPDFLoader #pip install langchain pypdf

def read_pdf(filePath):
    """
    This loads the the desired pdf and returns its content as a string.

    Parameters:
        filePath (String):
            This is the file path of the desired pdf

    Returns:
        pages (String):
            This is a large string that has the entire content of the desired pdf
    """

    loader = PyPDFLoader(filePath)
    pages = loader.load_and_split()


    pages = "\n".join([i.page_content for i in pages]) # changes from list of Document objects to single string (remove if Document object is better)

    return pages
