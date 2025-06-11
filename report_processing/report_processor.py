import pymupdf
import requests
import os
import io

class ReportProcessor:
    def __init__(self, path=None, url=None):
        self.path = path
        self.url = url
        assert (path is not None) != (url is not None), "Either path or url must be provided"
        self.text_list = []
        self.all_text = ''
        # self.load_pdf() # if I want to automatically load the pdf when creating the object
        # self.extract_text() # if I want to automatically extract text when creating the object

    def load_pdf(self):
        if self.path:
            self.pdf = pymupdf.open(self.path)
        else:
            response = requests.get(self.url)
            pdf_bytes = io.BytesIO(response.content)
            self.pdf = pymupdf.open(stream=pdf_bytes, filetype='pdf')

    def extract_text(self):
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ''.join(self.text_list)


    
# Notes: could include title and section recognition, but did not work well so far