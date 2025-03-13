# pdf_extractor.py
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader
from typing import Optional
import pytesseract
from PIL import Image
import io

class PDFExtractor:
    def __init__(self):
        self.extraction_methods = {
            'primary': self._extract_with_pymupdf,
            'complex': self._extract_with_pdfplumber,
            'fallback': self._extract_with_pypdf
        }

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using multiple methods if necessary.
        Falls back to alternative methods if the primary method fails.
        """
        text = None
        errors = []

        # Try methods in order until one succeeds
        for method_name in ['primary', 'complex', 'fallback']:
            try:
                text = self.extraction_methods[method_name](pdf_path)
                if text and text.strip():
                    return text
            except Exception as e:
                errors.append(f"{method_name} extraction failed: {str(e)}")
                continue

        # If all methods fail, raise an exception with details
        if not text:
            raise Exception(f"Failed to extract text using all methods: {'; '.join(errors)}")

        return text

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (primary method)"""
        text = ""
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

            # Extract images from the page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text += pytesseract.image_to_string(image)

        return text

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        with pdfplumber.open(pdf_path) as pdf:
            return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())

    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """Extract text using pypdf (fallback method)"""
        reader = PdfReader(pdf_path)
        return '\n'.join(page.extract_text() for page in reader.pages if page.extract_text())

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and normalizing line endings"""
        if not text:
            return ""
        # Remove extra whitespace and normalize line endings
        text = ' '.join(text.split())
        # Additional cleaning if needed
        return text