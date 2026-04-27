from pypdf import PdfReader

def load_pdf(path):
    reader = PdfReader(path) #Create a PdfReader object to read the PDF file at the specified path
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n" #Extract the text from each page of the PDF and concatenate it into a single string, adding a newline character after each page's text

    return text