from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader #pip install langchain pypdf

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


def pdfToTxt(filePath, txtOutDir):
    """
    This takes all the content from desired pdf and converts it into a plaintext file.

    Parameters:
        filePath (String):
            path of the pdf file which will be converted to plaintext
        
        txtOutDir (String):
            path of directory which output plain text file will be placed

    Output:
        One plain text file with the same name as the pdf will be placed in txtOutDir
    """
    loader = PyPDFLoader(filePath)
    docs = loader.load()

    for document in docs:
        docPath = document.metadata['source']
        with open(txtOutDir+"/"+docPath[docPath.rfind('/')+1:-4]+".txt", "a") as f: # cursed but it works so I don't really care (see line 43)
            f.write(document.page_content+" ")

def pdfDirToTxt(pdfFilesDir, txtOutDir):
    """
    Similar to pdfToTxt but converts a whole directory of pdfs to txt files

    Parameters:
        pdfFilesDir (String):
            Path of pdf files which will all be converted to plaintext file
        
        txtOutDir (String):
            path of directory which output plain text files will be placed
    
    Outputs:
        Plaintext files will be output in txtOutDir. Each file has the same name of the pdf from which it was derived
    """
    loader = PyPDFDirectoryLoader(pdfFilesDir)
    docs = loader.load()

    for document in docs:
        docPath = document.metadata['source']
        with open(txtOutDir+"/"+docPath[docPath.rfind('/')+1:-4]+".txt", "a") as f: # names new txt file the same as name of pdf
            f.write(document.page_content+" ")
