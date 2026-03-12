import fitz  # PyMuPDF
from typing import Dict, Any, List
import re

class PdfParsingService:
    async def parse_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract text, page count, and metadata from PDF using PyMuPDF.
        Also performs basic heuristic extraction of project info and document structure.
        """
        doc = fitz.open(stream=file_content, filetype="pdf")
        page_count = doc.page_count
        
        full_text = ""
        pages_text: List[str] = []
        sections: List[Dict[str, Any]] = []
        page_map: Dict[int, List[str]] = {}
        
        # Extract text and structure from each page
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages_text.append(text)
            full_text += text + "\n"
            
            # Heuristic section extraction for this page
            page_sections = self._extract_page_sections(page)
            for section in page_sections:
                section["page"] = page_num + 1
                sections.append(section)
            
            page_map[page_num + 1] = [s["title"] for s in page_sections]
            
        # Basic metadata
        metadata = doc.metadata
        
        # Heuristic extraction (lightweight regex/keyword matching)
        registration_number = self._extract_registration_number(full_text)
        call_title = self._extract_call_title(full_text)
        project_title = self._extract_project_title(full_text)
        
        # Detect markers
        markers = self._detect_markers(full_text, pages_text)
        
        doc.close()
        
        parsing_result = {
            "sections": sections,
            "page_map": page_map,
            "markers": markers,
            "metadata_fields": {
                "registration_number": registration_number,
                "call_title": call_title,
                "project_title": project_title
            }
        }
        
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
            "detected_language_note": "Parsed as PDF with structure",
            "parsing_result": parsing_result
        }

    def _extract_page_sections(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Extract headings based on font size and common patterns.
        """
        sections = []
        blocks = page.get_text("dict")["blocks"]
        
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        # Heuristic for headings: font size > 11 or all caps and not too long
                        # This is a simple heuristic and might need tuning for specific NRFU docs
                        is_heading = False
                        
                        # Rule 1: Larger font
                        if s["size"] > 11.5:
                            is_heading = True
                        
                        # Rule 2: All caps (at least 5 chars)
                        if len(text) > 5 and text.isupper() and not text.isdigit():
                            is_heading = True
                            
                        # Rule 3: Common section patterns (e.g., "1. Project info", "Section 2")
                        if re.match(r"^(?:Section|Розділ|Пункт|[\d\.]{1,5})\s+.+", text, re.IGNORECASE):
                            is_heading = True
                            
                        if is_heading and 3 < len(text) < 200:
                            sections.append({
                                "title": text,
                                "font_size": s["size"],
                                "is_bold": "bold" in s["font"].lower()
                            })
        
        # Merge consecutive spans that likely form one heading
        merged = []
        if sections:
            curr = sections[0]
            for i in range(1, len(sections)):
                # If they are very close in font size and the next one is just a continuation
                if abs(sections[i]["font_size"] - curr["font_size"]) < 0.5:
                     curr["title"] += " " + sections[i]["title"]
                else:
                    merged.append(curr)
                    curr = sections[i]
            merged.append(curr)
            
        return merged

    def _detect_markers(self, full_text: str, pages_text: List[str]) -> Dict[str, Any]:
        """
        Detect markers for budget, annexes, signatures.
        """
        markers = {
            "budget": [],
            "annex": [],
            "signature": []
        }
        
        budget_patterns = [r"бюджет", r"budget", r"кошторис", r"finances"]
        annex_patterns = [r"додаток", r"annex", r"appendix"]
        signature_patterns = [r"підпис", r"signature", r"signed", r"мп"]
        
        for i, page_text in enumerate(pages_text):
            page_num = i + 1
            low_text = page_text.lower()
            
            if any(re.search(p, low_text) for p in budget_patterns):
                markers["budget"].append(page_num)
            if any(re.search(p, low_text) for p in annex_patterns):
                markers["annex"].append(page_num)
            if any(re.search(p, low_text) for p in signature_patterns):
                markers["signature"].append(page_num)
                
        # Deduplicate
        for key in markers:
            markers[key] = sorted(list(set(markers[key])))
            
        return markers

    def _extract_registration_number(self, text: str) -> str:
        # Match patterns like 2023.03/0014 or 2024.01-0055
        pattern = r"(\d{4}\.\d{2}[/\-]\d{4})"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _extract_call_title(self, text: str) -> str:
        # Heuristic: search for "Назва конкурсу" or "Competition title" and take the following text
        # Limited to first 5000 characters for performance
        head = text[:5000]
        uk_pattern = r"(?:Назва конкурсу|НАЗВА КОНКУРСУ)[:\s]*([^\n\r]+)"
        en_pattern = r"(?:Competition title|COMPETITION TITLE)[:\s]*([^\n\r]+)"
        
        # Try exact matches first
        match = re.search(uk_pattern, head)
        if not match:
            match = re.search(en_pattern, head)
            
        if not match:
             # Try multi-line match
             uk_block = r"(?:Назва конкурсу|НАЗВА КОНКУРСУ)[:\s]*(.*?)(?=Назва проєкту|Project title|Registration|$)"
             en_block = r"(?:Competition title|COMPETITION TITLE)[:\s]*(.*?)(?=Project title|Registration|$)"
             match = re.search(uk_block, head, re.IGNORECASE | re.DOTALL)
             if not match or not match.group(1).strip():
                 match = re.search(en_block, head, re.IGNORECASE | re.DOTALL)

        if not match:
             # Try case-insensitive
             match = re.search(uk_pattern, head, re.IGNORECASE)
             if not match:
                 match = re.search(en_pattern, head, re.IGNORECASE)
            
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return re.sub(r"\s+", " ", extracted)
            
        # Fallback for some PDF text extractions where labels might be separate lines
        if "Назва конкурсу" in head or "НАЗВА КОНКУРСУ" in head:
             return "Detected (Ukrainian)"
        if "Competition title" in head or "COMPETITION TITLE" in head:
             return "Detected (English)"
             
        return None

    def _extract_project_title(self, text: str) -> str:
        # Heuristic: search for "Назва проєкту" or "Project title"
        # NRFU PDFs often have project titles after a label on the same line or next line
        head = text[:5000] # Increased head for better coverage
        # Common NRFU labels
        labels = [
            r"Назва проєкту", r"НАЗВА ПРОЄКТУ", 
            r"Project title", r"PROJECT TITLE",
            r"Назва теми проєкту", r"НАЗВА ТЕМИ ПРОЄКТУ",
            r"Тема проєкту", r"ТЕМА ПРОЄКТУ"
        ]
        
        for label in labels:
            # Try to find the label and the text following it on the same line
            # Using a more restrictive capture for the same line first
            pattern = rf"{label}[:\s]*([^\n\r]+)"
            match = re.search(pattern, head, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 5 and not extracted.startswith("("):
                    return extracted

            # If not found on same line, try multi-line with negative lookahead for other fields
            pattern_multi = rf"{label}[:\s]*((?:(?!(?:Project acronym|Acronym|Short name|Registration|Competition|Короткий|Абревіатура|Реєстраційний|Назва конкурсу)).)+)"
            match = re.search(pattern_multi, head, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 5 and not extracted.startswith("("):
                    # Clean up multiple newlines/spaces
                    return re.sub(r"\s+", " ", extracted)

        # If still not found, look for text between "Project title" and "Project acronym" or similar
        # Ukrainian
        uk_block_match = re.search(r"Назва проєкту[:\s]*(.*?)(?=Короткий|Project|Абревіатура|$)", head, re.IGNORECASE | re.DOTALL)
        if uk_block_match:
            extracted = uk_block_match.group(1).strip()
            if len(extracted) > 5:
                return extracted
        
        # English
        en_block_match = re.search(r"Project title[:\s]*(.*?)(?=Acronym|Short name|Project acronym|$)", head, re.IGNORECASE | re.DOTALL)
        if en_block_match:
            extracted = en_block_match.group(1).strip()
            if len(extracted) > 5:
                return extracted

        # Fallback keyword presence
        if any(l in head for l in ["Назва проєкту", "НАЗВА ПРОЄКТУ"]):
            return "Detected (Ukrainian)"
        if any(l in head for l in ["Project title", "PROJECT TITLE"]):
            return "Detected (English)"
            
        return None

pdf_parsing_service = PdfParsingService()
