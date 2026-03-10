import fitz  # PyMuPDF
from typing import Dict, Any, List
import re

class PdfParsingService:
    async def parse_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract text, page count, and metadata from PDF using PyMuPDF.
        Also performs basic heuristic extraction of project info.
        """
        doc = fitz.open(stream=file_content, filetype="pdf")
        page_count = doc.page_count
        
        full_text = ""
        pages_text: List[str] = []
        
        # Extract text from each page
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages_text.append(text)
            full_text += text + "\n"
            
        # Basic metadata
        metadata = doc.metadata
        
        # Heuristic extraction (lightweight regex/keyword matching)
        registration_number = self._extract_registration_number(full_text)
        call_title = self._extract_call_title(full_text)
        project_title = self._extract_project_title(full_text)
        
        doc.close()
        
        return {
            "page_count": page_count,
            "full_text": full_text,
            "pages_text": pages_text,
            "metadata": metadata,
            "preview": full_text[:500] if full_text else "",
            "extracted_text_length": len(full_text),
            "detected_project_registration_number": registration_number,
            "detected_call_title": call_title,
            "detected_project_title": project_title,
            "detected_language_note": "Parsed as PDF"
        }

    def _extract_registration_number(self, text: str) -> str:
        # Match patterns like 2023.03/0014 or 2024.01-0055
        pattern = r"(\d{4}\.\d{2}[/\-]\d{4})"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _extract_call_title(self, text: str) -> str:
        # Heuristic: search for "Назва конкурсу" or "Competition title" and take the following text
        # Limited to first 2000 characters for performance
        head = text[:2000]
        uk_pattern = r"(?:Назва конкурсу|НАЗВА КОНКУРСУ)[:\s]*([^\n\r]+)"
        en_pattern = r"(?:Competition title|COMPETITION TITLE)[:\s]*([^\n\r]+)"
        
        # Try exact matches first
        match = re.search(uk_pattern, head)
        if not match:
            match = re.search(en_pattern, head)
            
        if not match:
             # Try case-insensitive
             match = re.search(uk_pattern, head, re.IGNORECASE)
             if not match:
                 match = re.search(en_pattern, head, re.IGNORECASE)
            
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
            
        # Fallback for some PDF text extractions where labels might be separate lines
        if "Назва конкурсу" in head or "НАЗВА КОНКУРСУ" in head:
             return "Detected (Ukrainian)"
        if "Competition title" in head or "COMPETITION TITLE" in head:
             return "Detected (English)"
             
        return None

    def _extract_project_title(self, text: str) -> str:
        # Heuristic: search for "Назва проєкту" or "Project title"
        head = text[:2000]
        uk_pattern = r"(?:Назва проєкту|НАЗВА ПРОЄКТУ)[:\s]*([^\n\r]+)"
        en_pattern = r"(?:Project title|PROJECT TITLE)[:\s]*([^\n\r]+)"
        
        match = re.search(uk_pattern, head)
        if not match:
            match = re.search(en_pattern, head)

        if not match:
            match = re.search(uk_pattern, head, re.IGNORECASE)
            if not match:
                match = re.search(en_pattern, head, re.IGNORECASE)
            
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
            
        if "Назва проєкту" in head or "НАЗВА ПРОЄКТУ" in head:
            return "Detected (Ukrainian)"
        if "Project title" in head or "PROJECT TITLE" in head:
            return "Detected (English)"
            
        return None

pdf_parsing_service = PdfParsingService()
