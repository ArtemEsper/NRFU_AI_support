import fitz  # PyMuPDF
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import os
import tempfile
import json
import logging
from app.core.logger import logger
from app.core.config import settings
from app.services.layout_enrichment import LayoutEnrichmentService
from app.schemas.schemas import (
    DocumentZone,
    DocumentSection,
    ParsedDocumentStructure,
    PageClassification,
    DocumentBlock,
    TablePageClassification,
    CanonicalTable,
    CanonicalTableColumn,
    CanonicalTableRow,
    CanonicalTableCell,
)

class SemanticValidator:
    """Helper class for semantic validation of extracted fields."""
    
    @staticmethod
    def is_orcid(text: Optional[str]) -> bool:
        if not text: return False
        # ORCID pattern: https://orcid.org/0000-0002-1825-0097 or 0000-0002-1825-0097
        pattern = r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])"
        return bool(re.search(pattern, text))

    @staticmethod
    def normalize_orcid(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not text: return None, None
        pattern = r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])"
        match = re.search(pattern, text)
        if match:
            return match.group(1), text.strip()
        return None, text.strip()

    @staticmethod
    def is_email(text: Optional[str]) -> bool:
        if not text: return False
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return bool(re.search(pattern, text))

    @staticmethod
    def is_phone(text: Optional[str]) -> bool:
        if not text: return False
        # Ukrainian phone numbers usually +380... or similar
        pattern = r"(\+?\d{2}\s?\(?\d{3}\)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2})"
        # Or more generic but numeric-heavy
        if len(re.findall(r"\d", text)) < 7: return False
        return bool(re.search(pattern, text)) or bool(re.search(r"[\+\d\s\-\(\)]{7,}", text))

    @staticmethod
    def is_citizenship(text: Optional[str]) -> bool:
        if not text: return False
        # Should look like a country or nationality
        nationalities = ["Україна", "України", "Українець", "Українка", "Ukraine", "Ukrainian"]
        if any(nat.lower() in text.lower() for nat in nationalities):
            return True
        # If it's short and contains only letters, it might be a country
        if len(text) < 30 and text.isalpha():
            return True
        return False

    @staticmethod
    def is_address(text: Optional[str], labels: List[str]) -> bool:
        if not text: return False
        # Should not be equal to any of the field labels
        for label in labels:
            if label.lower() == text.lower().strip():
                return False
        # Addresses usually contain numbers, streets, cities
        # Simple heuristic: at least some letters and maybe a number
        return len(text.strip()) > 5

    @staticmethod
    def clean_name_and_get_titles(name: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Remove titles and honorifics from name and return them separately."""
        # Order matters: longer strings first to avoid partial matches
        titles = [r"Доктор\s+наук", r"Професор", r"Доктор", r"Пан", r"Пані", r"К\.т\.н\.", r"Д\.т\.н\.", r"Ph\.D\.", r"PhD", r"Professor", r"Dr\."]
        cleaned = name
        detected_titles = []
        
        # Apply cleaning iteratively to handle multiple titles
        changed = True
        while changed:
            original = cleaned
            for title in titles:
                match = re.search(r"^\s*(" + title + r")\b\s*", cleaned, flags=re.IGNORECASE)
                if match:
                    detected_titles.append(match.group(1).strip())
                    cleaned = re.sub(r"^\s*" + title + r"\b\s*", "", cleaned, flags=re.IGNORECASE).strip()
            changed = (original != cleaned)
            
        degree = None
        # Heuristic for degree: often contains "наук" or "к.т.н." etc.
        for t in detected_titles:
            if any(kw in t.lower() for kw in ["наук", "к.", "д.", "ph"]):
                degree = t
                break
        
        title = None
        for t in detected_titles:
            if t != degree:
                title = t
                break
                
        # Additional cleanup for name (ensure it doesn't contain institution text)
        # Names are usually 2-3 words. If it's much longer, it might be noisy.
        # But for now we just return the cleaned name.
        return cleaned.strip(), title, degree

    @staticmethod
    def clean_name(name: str) -> str:
        cleaned, _, _ = SemanticValidator.clean_name_and_get_titles(name)
        return cleaned

class PdfParsingService:
    def __init__(self):
        self.enrichment_service = LayoutEnrichmentService()

    async def parse_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Extract text, page count, and metadata from PDF using PyMuPDF.
        Also performs structured zone/section reconstruction.
        """
        doc = fitz.open(stream=file_content, filetype="pdf")
        page_count = doc.page_count
        
        full_text = ""
        pages_text: List[str] = []
        
        # Extract text from each page
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if not text.strip():
                text = page.get_text("text")
            text = text.replace("\u0000", "")
            pages_text.append(text)
            full_text += text + "\n"
            
        metadata = doc.metadata
        markers = self._detect_markers(full_text, pages_text)
        # Downgrade noisy lexical markers - ensure they don't influence zone transitions
        # by only using them for debug signals, which is already the case as they are 
        # returned in 'markers' but not used in _reconstruct_structure anchor_defs.

        page_table_classifications = self._classify_pages_for_tables(doc, pages_text)
        # Deterministic Zone/Section Reconstruction
        structure = self._reconstruct_structure(doc, pages_text, full_text, page_table_classifications)
        if settings.USE_LITEPARSE_SCANNED_TABLE_ROUTING:
            self._augment_scanned_tables_with_liteparse(file_content, structure)
        
        page_map: Dict[int, List[str]] = {}
        for zone in structure.zones:
            for section in zone.sections:
                p = section.page_number
                if p not in page_map:
                    page_map[p] = []
                page_map[p].append(section.title)

        # Build normalized structure result (Lightweight)
        # We exclude full page text from 'structure' in the main result to keep it lightweight
        # and remove raw text from metadata/profile blocks for normalized output.
        normalized_structure_obj = structure.model_copy(deep=True)
        for zone in normalized_structure_obj.zones:
            for block in zone.blocks:
                if block.block_type in ["metadata_block", "profile_block"] and block.metadata:
                    block.text = "[STRUCTURED DATA]" # Remove raw text from normalized
            for section in zone.sections:
                self._recursive_clean_blocks(section)
        
        normalized_structure = normalized_structure_obj.model_dump()
        normalized_structure.pop("pages", None)

        parsing_result = {
            "structure": normalized_structure,
            "tables": [t.model_dump() for t in structure.tables],
            "page_table_classifications": [c.model_dump() for c in structure.page_table_classifications],
            # Legacy compatibility projection. Canonical consumer path is parsing_result["structure"].
            "sections": [s.model_dump() for zone in structure.zones for s in zone.sections],
            "page_map": page_map,
            "markers": markers,
            "metadata_fields": {
                "registration_number": structure.application_id or self._extract_registration_number(full_text),
                "call_title": structure.call_title or self._extract_call_title(full_text),
                "project_title": structure.project_title or self._extract_project_title(full_text),
                "pi_name": structure.pi_name
            }
        }

        # Optional LiteParse enrichment
        enrichment_result = None
        if self.enrichment_service.enabled:
            pages_to_enrich = []
            for i, page_text in enumerate(pages_text):
                context = {"section_type": None}
                if self.enrichment_service.should_apply_layout_enrichment(page_text, i, context):
                    pages_to_enrich.append(i)
            
            if pages_to_enrich:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    enrichment_result = self.enrichment_service.enrich_document(tmp_path, page_indices=pages_to_enrich)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        
        registration_number = parsing_result["metadata_fields"]["registration_number"]
        call_title = parsing_result["metadata_fields"]["call_title"]
        project_title = parsing_result["metadata_fields"]["project_title"]

        doc.close()

        if enrichment_result:
            parsing_result["enrichment"] = enrichment_result.model_dump()
        
        # Save full debug JSON if needed (includes full text)
        if os.getenv("DEBUG_PARSING") == "true":
            full_result = parsing_result.copy()
            full_result["full_pages_text"] = pages_text
            self._save_debug_json(full_result, filename="debug_parse_output.json")
            
            # Save normalized structure directly
            self._save_debug_json(normalized_structure, filename="normalized_structure.json")

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
            "detected_language_note": "Parsed with structured reconstruction",
            "parsing_result": parsing_result
        }

    def _classify_pages_for_tables(self, doc: fitz.Document, pages_text: List[str]) -> List[TablePageClassification]:
        classifications: List[TablePageClassification] = []

        keywords = [
            "обсяг фінансування",
            "етапи фінансування",
            "таблиця",
            "бюджет",
            "кошторис",
        ]

        for idx, page_text in enumerate(pages_text):
            page = doc.load_page(idx)
            page_rect = page.rect
            page_area = float(max(page_rect.width * page_rect.height, 1.0))
            text = page_text or ""
            text_len = len(text.strip())
            word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
            digits = sum(ch.isdigit() for ch in text)
            digit_ratio = digits / max(len(text), 1)

            image_area = 0.0
            try:
                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type") == 1 and block.get("bbox"):
                        x0, y0, x1, y1 = block["bbox"]
                        image_area += max((x1 - x0), 0) * max((y1 - y0), 0)
            except Exception:
                image_area = 0.0

            image_area_ratio = image_area / page_area
            low_text = text.lower()
            has_financial_keyword = any(kw in low_text for kw in keywords)

            if word_count < 20 and image_area_ratio > 0.4:
                page_class = "scanned_image_only"
                confidence = 0.95
            elif word_count < 20 and text_len == 0:
                # Keep explicit scan suspicion for pages with no extractable text.
                page_class = "scanned_image_only"
                confidence = 0.75
            elif has_financial_keyword or (digit_ratio > 0.12 and word_count > 20):
                page_class = "native_text_complex_table"
                confidence = 0.9 if has_financial_keyword else 0.75
            else:
                page_class = "native_text"
                confidence = 0.9

            classifications.append(TablePageClassification(
                page_number=idx + 1,
                page_class=page_class,
                confidence=round(confidence, 3),
                signals={
                    "word_count": word_count,
                    "digit_ratio": round(digit_ratio, 4),
                    "image_area_ratio": round(image_area_ratio, 4),
                    "has_financial_keyword": has_financial_keyword,
                },
            ))

        return classifications

    def _augment_scanned_tables_with_liteparse(self, file_content: bytes, structure: ParsedDocumentStructure):
        def _zone_for_page(page_number: int) -> Optional[str]:
            zone = next(
                (z for z in structure.zones if z.page_start <= page_number <= (z.page_end or z.page_start)),
                None,
            )
            return zone.zone_type if zone else None

        # Narrow pilot: scanned annex/certificate pages only.
        scanned_pages = [
            c.page_number
            for c in structure.page_table_classifications
            if c.page_class == "scanned_image_only" and _zone_for_page(c.page_number) == "annex"
        ]
        if not scanned_pages:
            return

        if not self.enrichment_service.cli_available:
            if not structure.metadata:
                structure.metadata = {}
            structure.metadata["liteparse_scanned_routing_warning"] = "LiteParse CLI not available; scanned routing skipped."
            return

        pages_arg = ",".join(str(p) for p in scanned_pages)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            liteparse_result = self.enrichment_service.run_liteparse_cli(
                file_path=tmp_path,
                pages=pages_arg,
                dpi=settings.LITEPARSE_HIGH_QUALITY_DPI,
                ocr_enabled=True,
            )
        except Exception as exc:
            if not structure.metadata:
                structure.metadata = {}
            structure.metadata["liteparse_scanned_routing_warning"] = f"LiteParse scanned routing failed: {exc}"
            return
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        for page_data in liteparse_result.get("pages", []):
            page_num = page_data.get("page_number") or page_data.get("page")
            if not isinstance(page_num, int):
                continue

            cls = next((c for c in structure.page_table_classifications if c.page_number == page_num), None)
            if not cls or cls.page_class != "scanned_image_only":
                continue

            raw_text = page_data.get("text") or ""
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines() if ln and ln.strip()]
            warnings: List[str] = []
            if not lines:
                warnings.append("no_text_from_liteparse")

            rows: List[CanonicalTableRow] = []
            for idx, line in enumerate(lines):
                rows.append(
                    CanonicalTableRow(
                        row_id=f"r{idx}",
                        cells=[
                            CanonicalTableCell(col_id="c0", text=str(idx + 1), normalized=idx + 1),
                            CanonicalTableCell(col_id="c1", text=line, normalized=line),
                        ],
                    )
                )

            page_zone = next(
                (z for z in structure.zones if z.page_start <= page_num <= (z.page_end or z.page_start)),
                None,
            )
            zone_type = page_zone.zone_type if page_zone else "unknown"

            scanned_table = CanonicalTable(
                table_id=f"tbl_scanned_liteparse_p{page_num:03d}_01",
                table_family="scanned_page_ocr_layout",
                zone_type=zone_type,
                title="LiteParse OCR Layout Lines",
                page_start=page_num,
                page_end=page_num,
                source={
                    "parser": "liteparse_cli",
                    "page_class": cls.page_class,
                    "confidence": cls.confidence,
                    "extraction_mode": "scanned_page_only_route",
                    "warnings": warnings,
                },
                columns=[
                    CanonicalTableColumn(col_id="c0", name="line_number", semantic_type="int"),
                    CanonicalTableColumn(col_id="c1", name="ocr_text", semantic_type="string"),
                ],
                rows=rows,
                spans=[],
                validation={
                    "row_count": len(rows),
                    "column_count": 2,
                    "normalization_warnings": warnings,
                },
            )

            structure.tables = [
                t for t in structure.tables
                if not (
                    t.table_family == "scanned_page_ocr_layout"
                    and t.page_start == page_num
                )
            ]
            structure.tables.append(scanned_table)

    def _reconstruct_structure(
        self,
        doc: fitz.Document,
        pages_text: List[str],
        full_text: str = "",
        page_table_classifications: Optional[List[TablePageClassification]] = None
    ) -> ParsedDocumentStructure:
        """
        Implements deterministic zone/section reconstruction based on NRFU Spec (Master Spec).
        Strictly follows Zone Boundary Rules and page-based transition semantics.
        """
        structure = ParsedDocumentStructure()
        structure.pages = pages_text
        structure.page_table_classifications = page_table_classifications or []
        debug_log = []
        
        # Define anchors and their classification behaviors
        # NULL_CONTENT_PAGE triggers: next zone starts on NEXT page
        # CONTENT_PAGE triggers (like Finance): next zone starts on SAME page
        anchor_defs = [
            {"name": "Опис проєкту", "type": "description", "behavior": "content_page", "patterns": [r"^\s*Опис\s+проєкту\s*$"]},
            {"name": "Фінансування проєкту", "type": "financial", "behavior": "content_page", "patterns": [r"^\s*Фінансування\s+проєкту\s*$"]},
            {"name": "Учасник конкурсу або партнер", "type": "institution", "behavior": "marker_only", "patterns": [r"^\s*Учасник\s+конкурсу\s+або\s+партнер\s*$", r"^\s*Учасник\s+конкурсу/субвиконавці\s*$"]},
            {"name": "Керівник проєкту", "type": "pi_profile", "behavior": "marker_only", "patterns": [r"^\s*Керівник\s+проєкту\s*$"]},
            {"name": "Виконавці", "type": "team", "behavior": "marker_only", "patterns": [r"^\s*Виконавці\s*$"]},
            {"name": "Додатки", "type": "annex", "behavior": "marker_only", "patterns": [r"^\s*Додатки\s*$"]},
            {"name": "Довідки", "type": "annex", "behavior": "marker_only", "patterns": [r"^\s*Довідки\s*$"]},
        ]
        
        header_noise_patterns = [
            r"Національний\s+фонд\s+досліджень\s+України",
            r"Конкурс\s+проєктів\s+із\s+виконання\s+наукових\s+досліджень\s+і\s+розробок",
            r"Передова\s+наука\s+в\s+Україні",
            r"Реєстраційний\s+номер\s+проєкту:?\s*[\d\.\/]+",
        ]

        # 1. Page-by-page Anchor Detection and Classification
        page_classifications = []
        detected_anchors = []
        
        # Deterministic Header Detection: look for lines appearing on > 10% of pages (min 3)
        # or exactly matching the mandated noise pattern.
        line_counts = {}
        for p_text in pages_text:
            unique_lines = set(line.strip() for line in p_text.split("\n") if len(line.strip()) > 10)
            for line in unique_lines:
                line_counts[line] = line_counts.get(line, 0) + 1
        
        page_count = len(pages_text)
        detected_noise_lines = {line for line, count in line_counts.items() 
                               if count >= max(3, page_count * 0.1)}
        
        # Add mandated patterns to detected noise
        header_noise_patterns = [
            r"Національний\s+фонд\s+досліджень\s+України",
            r"Конкурс\s+проєктів\s+із\s+виконання\s+наукових\s+досліджень\s+і\s+розробок",
            r"Передова\s+наука\s+в\s+Україні",
            r"Реєстраційний\s+номер\s+проєкту:?\s*[\d\.\/]+",
            r"\"Передова\s+наука\s+в\s+Україні\"",
            r"Сторінка\s+\d+\s+із\s+\d+",
        ]

        # Pre-process pages to remove headers and create blocks
        cleaned_pages_lines = []
        for i, page_text in enumerate(pages_text):
            lines = [line.strip() for line in page_text.split("\n") if line.strip()]
            cleaned_lines = []
            for line in lines:
                is_noise = line in detected_noise_lines
                if not is_noise:
                    for noise_pat in header_noise_patterns:
                        if re.search(noise_pat, line, re.IGNORECASE):
                            is_noise = True
                            break
                if not is_noise:
                    cleaned_lines.append(line)
            cleaned_pages_lines.append(cleaned_lines)

        for i, cleaned_lines in enumerate(cleaned_pages_lines):
            page_anchors = []
            for a_def in anchor_defs:
                for pattern in a_def["patterns"]:
                    # Match pattern against each cleaned line
                    for line in cleaned_lines:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            anchor_info = {
                                "text": a_def["name"],
                                "page": i + 1,
                                "match": match.group(0),
                                "behavior": a_def["behavior"],
                                "type": a_def["type"]
                            }
                            page_anchors.append(anchor_info)
                            break
                    if page_anchors and page_anchors[-1]["text"] == a_def["name"]:
                        break

            # Classification
            if not page_anchors:
                p_type = "content"
            else:
                # Spec: "contains no meaningful content besides the marker"
                # A marker page usually has very few lines after removing noise.
                if len(cleaned_lines) <= 3: # Marker + maybe 1-2 small artifacts
                    p_type = "marker_only"
                else:
                    p_type = "marker_plus_content"
            
            page_classifications.append(PageClassification(
                page_number=i + 1,
                type=p_type,
                anchors=[pa["text"] for pa in page_anchors]
            ))
            detected_anchors.extend(page_anchors)

        structure.detected_anchors = detected_anchors
        structure.page_classifications = page_classifications
        
        # 2. Zone Transition Logic - State Machine based on updated spec
        zones = []
        page_count = len(pages_text)
        
        # States: WAITING_FOR_ZONE, IN_ZONE, ON_MARKER_PAGE
        # We start by initializing Zone 1 (Registration Data)
        current_zone = DocumentZone(
            name="Zone 1 — Header and Registration Data",
            zone_type="header",
            page_start=1,
            trigger_reason="start of document"
        )
        
        active_zone_types = {"header"}
        marker_pages = [] # track pages that are markers to exclude them from zones
        
        for i in range(page_count):
            p_num = i + 1
            p_class = page_classifications[i]
            
            # Check for triggers on this page
            trigger_anchor = None
            if p_class.anchors:
                for pa in detected_anchors:
                    if pa["page"] == p_num:
                        # Skip if this zone type is already active (prevents duplicates/restarts)
                        if pa["type"] in active_zone_types and pa["type"] != "annex":
                            continue
                        trigger_anchor = pa
                        break
            
            if trigger_anchor:
                # MARKER DETECTED
                active_zone_types.add(trigger_anchor["type"])
                
                if trigger_anchor["behavior"] == "marker_only" and p_class.type == "marker_only":
                    # ON_MARKER_PAGE (Null Content)
                    marker_pages.append(p_num)
                    
                    # Terminate current zone at PREVIOUS page
                    # Unless the previous page was also a marker-only page (consecutive markers)
                    if current_zone:
                        if p_num - 1 not in marker_pages:
                            current_zone.page_end = p_num - 1
                            if current_zone.page_start <= current_zone.page_end:
                                zones.append(current_zone)
                            else:
                                debug_log.append(f"Skipping empty zone {current_zone.name} ending at {p_num-1}")
                        else:
                            # Consecutive marker page - current_zone is already waiting to start on next page
                            debug_log.append(f"Consecutive marker page {p_num} detected. Extending null zone.")
                    
                    # Next zone starts from NEXT page
                    current_zone = DocumentZone(
                        name=f"Zone — {trigger_anchor['text']}",
                        zone_type=trigger_anchor["type"],
                        page_start=p_num + 1,
                        trigger_reason=f"Marker-only anchor '{trigger_anchor['text']}' on page {p_num}"
                    )
                    debug_log.append(f"Transition at page {p_num}: {trigger_anchor['text']} (marker_only) -> New zone starts at {p_num+1}")
                
                elif trigger_anchor["type"] == "financial":
                    # Special Rule for Zone 3 (Financial): MUST be exactly 1 page
                    # Terminate current zone at PREVIOUS page
                    if current_zone:
                        current_zone.page_end = p_num - 1
                        if current_zone.page_start <= current_zone.page_end:
                            zones.append(current_zone)
                    
                    # Next zone starts on SAME page
                    current_zone = DocumentZone(
                        name=f"Zone 3 — Financial Section",
                        zone_type="financial",
                        page_start=p_num,
                        page_end=p_num, # Force 1 page
                        trigger_reason=f"Financial anchor '{trigger_anchor['text']}' on page {p_num}"
                    )
                    
                    # Spec says: "If more pages detected → truncate to first page → flag warning: financial_section_too_long"
                    # We check if the *next* page also contains financial markers OR if there's no marker before a lot of content
                    # For simplicity, if the current zone was forced to end because of this rule and the next marker is far,
                    # we could flag it. 
                    # But the requirement is about detecting if it *would* have been longer.
                    # Since our previous logic allowed it to be longer (until next marker),
                    # if the next marker is NOT on the next page, it would have been longer.
                    
                    next_marker_page = -1
                    for pa in detected_anchors:
                        if pa["page"] > p_num:
                            next_marker_page = pa["page"]
                            break
                    
                    if next_marker_page == -1 or next_marker_page > p_num + 1:
                         # It would have spanned at least until next_marker_page - 1
                         debug_log.append("Warning: financial_section_too_long detected. Truncating to 1 page.")
                         # We can't easily add it to DocumentZone without schema change, 
                         # but we can add to debug_log or a global warnings list if we had one.
                         # Actually, we can add it to the structure metadata.
                         if not structure.metadata: structure.metadata = {}
                         structure.metadata["financial_section_too_long"] = True
                    
                    zones.append(current_zone)
                    
                    # Start next zone (waiting for next marker or implicitly continuing?)
                    # According to spec, usually Zone 4 follows after some marker.
                    # We create a temporary "buffer" zone or wait for the next marker.
                    current_zone = DocumentZone(
                        name="Zone 4 — Transition / Unknown",
                        zone_type="unknown",
                        page_start=p_num + 1,
                        trigger_reason="Automatic transition after 1-page financial zone"
                    )
                    debug_log.append(f"Transition at page {p_num}: Financial (1-page rule applied)")
                
                else:
                    # CONTENT_PAGE behavior (e.g. description)
                    # Terminate current zone at PREVIOUS page
                    if current_zone:
                        current_zone.page_end = p_num - 1
                        if current_zone.page_start <= current_zone.page_end:
                            zones.append(current_zone)
                    
                    # Next zone starts on SAME page
                    current_zone = DocumentZone(
                        name=f"Zone — {trigger_anchor['text']}",
                        zone_type=trigger_anchor["type"],
                        page_start=p_num,
                        trigger_reason=f"Content anchor '{trigger_anchor['text']}' on page {p_num}"
                    )
                    debug_log.append(f"Transition at page {p_num}: {trigger_anchor['text']} (content) -> New zone starts at {p_num}")

        # Finalize last zone
        if current_zone:
            current_zone.page_end = page_count
            if current_zone.page_start <= current_zone.page_end:
                zones.append(current_zone)
        
        # Post-process: Remove zones with invalid ranges or duplicates
        valid_zones = []
        for z in zones:
            if z.page_start <= z.page_end and z.page_start <= page_count:
                # Ensure no zone includes a marker page if it's not supposed to
                # If a zone starts on a marker page due to consecutive markers, push its start forward
                while z.page_start in marker_pages and z.page_start <= z.page_end:
                    z.page_start += 1
                
                if z.page_start > z.page_end:
                    continue

                if any(m_p in range(z.page_start, z.page_end + 1) for m_p in marker_pages):
                     debug_log.append(f"Warning: Zone {z.name} ({z.page_start}-{z.page_end}) overlaps with marker pages {marker_pages}")
                
                # Canonical mapping
                name_map = {
                    "header": "Zone 1 — Header and Registration Data",
                    "description": "Zone 2 — Project Description",
                    "financial": "Zone 3 — Financial Section",
                    "institution": "Zone 4 — Institution / Partner Section",
                    "pi_profile": "Zone 5 — PI Profile Section",
                    "team": "Zone 6 — Co-authors Section",
                    "annex": "Zone 7 — Annexes / Certificates"
                }
                z.name = name_map.get(z.zone_type, z.name)
                valid_zones.append(z)
        
        # Merge consecutive zones of same type (e.g. Додатки + Довідки both are 'annex')
        merged_zones = []
        if valid_zones:
            curr = valid_zones[0]
            for i in range(1, len(valid_zones)):
                nxt = valid_zones[i]
                if curr.zone_type == nxt.zone_type and curr.zone_type == "annex":
                    curr.page_end = nxt.page_end
                elif curr.zone_type == "unknown" and nxt.zone_type != "unknown":
                    # Skip unknown transition zones if followed by a valid one
                    curr = nxt
                else:
                    merged_zones.append(curr)
                    curr = nxt
            merged_zones.append(curr)
        
        # FINAL ZONE DEDUPLICATION: Ensure no overlapping or redundant zones before populating blocks
        final_zones = []
        seen_ranges = set()
        for z in merged_zones:
            z_range = (z.page_start, z.page_end, z.zone_type)
            if z_range not in seen_ranges:
                final_zones.append(z)
                seen_ranges.add(z_range)

        structure.zones = final_zones
        structure.metadata = {"marker_pages": marker_pages}
        
        # 3. Section Mapping & Subsection Reconstruction
        # Subsections for Zone 2
        zone2_subsections = [
            (r"^3\.1\.?\s+", "3.1"),
            (r"^3\.2\.?\s+", "3.2"),
            (r"^3\.3\.?\s+", "3.3"),
            (r"^3\.4\.?\s+", "3.4"),
            (r"^3\.5\.?\s+", "3.5"),
            (r"^3\.6\.?\s+", "3.6"),
            (r"^3\.7\.?\s+", "3.7"),
            (r"^3\.8\.?\s+", "3.8"),
            (r"^3\.9\.?\s+", "3.9"),
            (r"^Список\s+використаних\s+джерел", "Список використаних джерел"),
        ]

        # 1. Block construction with refined types
        for z in structure.zones:
            zone_blocks = []
            seen_in_zone = set()
            for p_num in range(z.page_start, z.page_end + 1):
                p_idx = p_num - 1
                lines = cleaned_pages_lines[p_idx]
                
                if lines:
                    for line in lines:
                        # Deterministic deduplication check
                        # Skip if this line was already added to THIS zone on THIS page
                        # (prevents artifacts from multiple anchor matches or similar)
                        # We use (page, line) as a unique identifier for now
                        block_key = (p_num, line.strip())
                        if block_key in seen_in_zone:
                            continue
                        seen_in_zone.add(block_key)

                        b_type = "paragraph"
                        
                        # Task 1: Remove anchor text from all block content
                        is_exact_anchor = False
                        for a_def in anchor_defs:
                            for pattern in a_def["patterns"]:
                                if re.search(pattern, line, re.IGNORECASE):
                                    is_exact_anchor = True
                                    break
                            if is_exact_anchor: break
                        
                        if is_exact_anchor:
                            # Skip adding this line as a block content if it's exactly the anchor
                            # (unless it's a marker page, where we keep marker_block)
                            if p_num not in marker_pages:
                                continue

                        # Detect headers
                        is_header = False
                        if z.zone_type == "description":
                            if re.search(r"^\s*Опис\s+проєкту\s*$", line, re.IGNORECASE):
                                is_header = True
                            else:
                                for pattern, _ in zone2_subsections:
                                    if re.search(pattern, line, re.IGNORECASE):
                                        is_header = True
                                        break
                                        
                        # Task 2: Detect subsection headers
                        is_sub_header = False
                        if z.zone_type == "description" and not is_header:
                            if self._is_subsection_header(line):
                                is_sub_header = True

                        if is_header:
                            b_type = "section_header"
                        elif is_sub_header:
                            b_type = "subsection_header"
                        elif z.zone_type in ["pi_profile", "team"]:
                            b_type = "profile_block"
                        elif z.zone_type == "institution":
                            b_type = "metadata_block"
                        elif z.zone_type == "financial":
                            b_type = "table_block"
                        elif z.zone_type == "annex":
                            b_type = "scan_block"
                        elif p_num in marker_pages:
                            b_type = "marker_block"
                        
                        # Precision Refinement: Detect foreign structured blocks in narrative zones
                        if z.zone_type == "description" and b_type == "paragraph":
                            if self._is_foreign_structured_content(line):
                                b_type = "foreign_structured_block"
                            
                        zone_blocks.append(DocumentBlock(
                            block_type=b_type,
                            text=line,
                            page_number=p_num
                        ))
            
            z.blocks = zone_blocks

            # 1.5 Paragraph Reconstruction Layer
            reconstructed_blocks = []
            if z.blocks:
                current_block = None
                for b in z.blocks:
                    # Skip merging for special blocks (including headers)
                    if b.block_type != "paragraph":
                        if current_block:
                            # Task 4: Cleanup merged block (remove trailing hyphen or whitespace)
                            current_block.text = current_block.text.strip()
                            reconstructed_blocks.append(current_block)
                            current_block = None
                        reconstructed_blocks.append(b)
                        continue
                    
                    if current_block is None:
                        current_block = DocumentBlock(
                            block_type="paragraph",
                            text=b.text.strip(),
                            page_number=b.page_number
                        )
                    else:
                        # Logic to merge
                        prev_text = current_block.text.strip()
                        curr_text = b.text.strip()
                        
                        should_merge = False
                        is_narrative_zone = z.zone_type == "description"
                        if prev_text:
                            # 1. Merge if previous line does NOT end with sentence-ending punctuation
                            last_char = prev_text[-1]
                            ends_sentence = last_char in [".", "!", "?", ":", ";"]
                            
                            starts_with_lower = curr_text and curr_text[0].islower()
                            is_continuation = curr_text and curr_text[0] in ["(", ")", ",", "-", "–", "—"]
                            is_bullet = curr_text and (curr_text.startswith("•") or curr_text.startswith("-") or curr_text.startswith("*"))
                            
                            # Task 4: Merge line-by-line fragments (fragments shorter than ~30 chars)
                            is_short_fragment = len(prev_text) < 40 or len(curr_text) < 40
                            
                            # Aggressive merge is limited to narrative zones.
                            # Metadata-heavy zones (e.g. Zone 1) keep line boundaries for label/value parsing.
                            if is_narrative_zone and not is_bullet:
                                if not ends_sentence or starts_with_lower or is_continuation or is_short_fragment:
                                    if len(prev_text) < 3000: # Slightly larger max length
                                        should_merge = True
                                
                        # 2. Merge broken words
                        if prev_text and (prev_text.endswith("’") or prev_text.endswith("'") or prev_text.endswith("-")):
                             join_char = ""
                             if prev_text.endswith("-"):
                                 # Heuristic: if it's a long word split, remove hyphen.
                                 # We check if the last char before hyphen is a letter and first char of next is a letter.
                                 if len(prev_text) > 1 and prev_text[-2].isalpha() and curr_text and curr_text[0].isalpha():
                                     prev_text = prev_text[:-1]
                             
                             current_block.text = prev_text + join_char + curr_text
                        elif should_merge:
                            current_block.text = prev_text + " " + curr_text
                        else:
                            current_block.text = prev_text.strip()
                            reconstructed_blocks.append(current_block)
                            current_block = DocumentBlock(
                                block_type="paragraph",
                                text=b.text.strip(),
                                page_number=b.page_number
                            )
                
                if current_block:
                    current_block.text = current_block.text.strip()
                    reconstructed_blocks.append(current_block)
            
            z.blocks = reconstructed_blocks

            # 2. Hierarchical Section Reconstruction for Zone 2
            if z.zone_type == "description":
                main_section = DocumentSection(
                    title="Опис проєкту",
                    anchor="Опис проєкту",
                    page_number=z.page_start,
                    subsections=[]
                )
                
                current_sub = None
                for block in z.blocks:
                    # Group foreign_structured_blocks into a synthetic section
                    if block.block_type == "foreign_structured_block":
                        if not current_sub or current_sub.title != "Foreign Structured Forms":
                            current_sub = DocumentSection(
                                title="Foreign Structured Forms",
                                anchor="Synthetic",
                                page_number=block.page_number,
                                subsections=[]
                            )
                            main_section.subsections.append(current_sub)
                        current_sub.blocks.append(block)
                        continue

                    if block.block_type == "section_header":
                        # Check if it's a subsection or the main title
                        is_main = re.search(r"^\s*Опис\s+проєкту\s*$", block.text, re.IGNORECASE)
                        if is_main:
                            continue # Already have main_section
                        
                        # Find which subsection it is
                        sub_title = None
                        for pattern, title in zone2_subsections:
                            if re.search(pattern, block.text, re.IGNORECASE):
                                sub_title = title
                                break
                        
                        if sub_title:
                            # Create new subsection
                            current_sub = DocumentSection(
                                title=sub_title,
                                anchor=block.text,
                                page_number=block.page_number,
                                blocks=[block]
                            )
                            main_section.subsections.append(current_sub)
                        else:
                            # If it's a header but not in our list, add to main or current sub
                            if current_sub:
                                current_sub.blocks.append(block)
                            else:
                                main_section.blocks.append(block)
                    elif block.block_type == "subsection_header":
                        # Task 3: Attach narrative to header
                        # Treat subsection_header as a marker that starts a group of narrative blocks
                        # We don't necessarily create a DocumentSection for it if it's not a 3.x subsection,
                        # but we ensure it's in the blocks list before narrative.
                        if current_sub:
                            current_sub.blocks.append(block)
                        else:
                            main_section.blocks.append(block)
                    else:
                        # Narrative block
                        if current_sub:
                            current_sub.blocks.append(block)
                        else:
                            main_section.blocks.append(block)
                
                # CRITICAL: Clear z.blocks because they are now owned by sections
                z.blocks = []
                z.sections = [main_section]

        # 4. Metadata Extraction (titles, IDs, PI Name)
        # PI Name extraction from Zone 5
        pi_zone = next((z for z in structure.zones if z.zone_type == "pi_profile"), None)
        if pi_zone:
            # Look for "Пан <Name>" or "Пані <Name>" in Zone 5 pages
            pi_name_found = False
            for p_idx in range(pi_zone.page_start - 1, pi_zone.page_end):
                if pi_name_found: break
                p_text = pages_text[p_idx]
                # Filter out headers
                lines = [l.strip() for l in p_text.split("\n") if l.strip()]
                cleaned_lines = []
                for line in lines:
                    is_noise = False
                    for noise_pat in header_noise_patterns:
                        if re.search(noise_pat, line, re.IGNORECASE):
                            is_noise = True
                            break
                    if not is_noise:
                        cleaned_lines.append(line)
                
                for line in cleaned_lines:
                    # Match "Пан " or "Пані " or "Доктор " or "Професор "
                    match = re.search(r"^(Пан|Пані|Доктор|Професор)\s+([А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+\s*[А-ЯЄІЇҐ]?[а-яєіїґ]*)", line)
                    if match:
                        structure.pi_name = match.group(2).strip()
                        pi_name_found = True
                        break
                    # Fallback: first line that looks like a name if no Pan/Pani
                    elif not pi_name_found and re.match(r"^[А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+(\s+[А-ЯЄІЇҐ][а-яєіїґ]+)?$", line):
                        # Simple name check - only use if it's one of the first few lines
                        if cleaned_lines.index(line) < 5:
                            structure.pi_name = line
                            pi_name_found = True
                            break
            
            # Layer 2: Extract Person Profile (Zone 5)
            self._extract_pi_profile(pi_zone, pages_text, structure)

        # Layer 2: Extract Institution Profile (Zone 4)
        inst_zone = next((z for z in structure.zones if z.zone_type == "institution"), None)
        if inst_zone:
            self._extract_institution_profile(inst_zone, pages_text, structure)
        
        # Layer 2: Extract Co-authors (Zone 6)
        team_zone = next((z for z in structure.zones if z.zone_type == "team"), None)
        if team_zone:
            self._extract_team_profiles(team_zone, pages_text, structure)

        # Layer 2: Extract Financial Summary (Zone 3)
        financial_zone = next((z for z in structure.zones if z.zone_type == "financial"), None)
        if financial_zone:
            self._extract_financial_summary(financial_zone, structure)
            
        # Layer 2: Detect certificate / compliance document subtypes in Zone 7
        annex_zone = next((z for z in structure.zones if z.zone_type == "annex"), None)
        if annex_zone:
            self._detect_annex_subtypes(annex_zone, pages_text, structure)

        p1_text = pages_text[0] if page_count > 0 else ""
        structure.project_title = self._extract_project_title(p1_text)
        structure.application_id = self._extract_registration_number(p1_text)
        structure.call_title = self._extract_call_title(p1_text)
        
        # Deterministic Zone 1 metadata extraction from the actual Zone 1 blocks.
        zone1 = next((z for z in structure.zones if z.zone_type == "header"), None)
        z1_meta = self._extract_zone1_metadata(zone1, pages_text)

        if not structure.metadata: structure.metadata = {}
        structure.metadata.update(z1_meta)
        
        # Application ID might be found here too
        if z1_meta.get("application_id") and not structure.application_id:
            structure.application_id = z1_meta["application_id"]

        structure.debug_log = debug_log
        if zone1 and z1_meta:
            # Ensure Zone 1 has a structured metadata representation in its own block path.
            zone1_meta_block = next((b for b in zone1.blocks if b.block_type == "metadata_block"), None)
            if zone1_meta_block:
                zone1_meta_block.metadata = z1_meta
            else:
                zone1.blocks.insert(0, DocumentBlock(
                    block_type="metadata_block",
                    text="Zone 1 Metadata",
                    page_number=zone1.page_start,
                    metadata=z1_meta
                ))
            self._cleanup_zone1_blocks(zone1, z1_meta, structure.project_title)

        return structure

    def _extract_institution_profile(self, zone: DocumentZone, pages: List[str], structure: ParsedDocumentStructure):
        """Extract structured fields from Institution Profile (Zone 4) using regex."""
        if not structure.metadata: structure.metadata = {}
        inst_data = {}
        
        full_zone_text = "\n".join([block.text for block in zone.blocks])
        
        # Mapping labels to fields with improved regex
        regex_map = {
            "institution_name": r"(?:Назва\s+установи|Учасник\s+конкурсу(?:\/субвиконавці)?)\s*:?\s*(.+)",
            "edrpou_code": r"Код\s+ЄДРПОУ\s*:?\s*(\d{8})",
            "kved_code": r"Код\(и\)\s+КВЕД\s*:?\s*([\d\.,\s]{5,})",
            "legal_address": r"Юридична\s+адреса\s*:?\s*(.+)",
            "postal_address": r"Поштова\s+адреса\s*:?\s*(.+)",
            "physical_address": r"Фактична\s+адреса\s*:?\s*(.+)",
            "phone_number": r"Телефон\s*:?\s*([\+\d\s\-\(\)]+)",
            "email": r"Адреса\s+електронної\s+пошти\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "website": r"Посилання\s+на\s+веб\s*сторінку\s*:?\s*(https?://[^\s]+|www\.[^\s]+)",
        }

        labels_to_check = [
            "Назва установи", "Код ЄДРПОУ", "Код(и) КВЕД", "Юридична адреса", 
            "Поштова адреса", "Фактична адреса", "Телефон", "Адреса електронної пошти", 
            "Посилання на веб сторінку", "підприємства/установи/організації"
        ]

        for field, pattern in regex_map.items():
            match = re.search(pattern, full_zone_text, re.IGNORECASE | re.MULTILINE)
            if match:
                val = match.group(1).strip()
                # Semantic validation
                is_valid = True
                if field == "edrpou_code" and not re.match(r"^\d{8}$", val): is_valid = False
                if field in ["legal_address", "physical_address", "postal_address"]:
                    if not SemanticValidator.is_address(val, labels_to_check): is_valid = False
                if field == "phone_number" and not SemanticValidator.is_phone(val): is_valid = False
                if field == "email" and not SemanticValidator.is_email(val): is_valid = False
                
                if is_valid:
                    inst_data[field] = val
                else:
                    inst_data[field] = None

        # Fallback for institution name if not matched by label
        if "institution_name" not in inst_data:
            lines = [l.strip() for l in full_zone_text.split("\n") if l.strip()]
            if lines and len(lines[0]) > 10:
                inst_data["institution_name"] = lines[0]

        structure.metadata["institution_profile"] = inst_data

        # Update ONLY the first metadata_block to contain full metadata
        # and remove other metadata_blocks to avoid duplication
        metadata_blocks = [b for b in zone.blocks if b.block_type == "metadata_block"]
        if metadata_blocks:
            main_block = metadata_blocks[0]
            main_block.metadata = inst_data
            main_block.text = "[STRUCTURED DATA]"
            
            # Remove redundant metadata blocks
            zone.blocks = [b for b in zone.blocks if b.block_type != "metadata_block" or b == main_block]

    def _extract_pi_profile(self, zone: DocumentZone, pages: List[str], structure: ParsedDocumentStructure):
        """Extract structured fields from PI Profile (Zone 5)."""
        if not structure.metadata: structure.metadata = {}
        
        pi_name, pi_title, pi_degree = SemanticValidator.clean_name_and_get_titles(structure.pi_name or "")
        pi_data = {
            "name": pi_name,
            "title": pi_title,
            "degree": pi_degree
        }
        
        full_zone_text = ""
        for block in zone.blocks:
            full_zone_text += block.text + "\n"

        lines = [re.sub(r"\s+", " ", l.strip()) for l in full_zone_text.split("\n") if l.strip()]
        label_specs = [
            ("gender", r"Стать"),
            ("birth_date", r"Дата\s+народження"),
            ("citizenship", r"Громадянство"),
            ("orcid", r"ORCID"),
            ("h_index_scopus", r"Індекс\s+Хірша\s+\(SCOPUS\)"),
            ("total_publications", r"Загальна\s+кількість\s+публікацій"),
            ("degree_raw", r"Науковий\s+ступінь"),
            ("position", r"Посада"),
            ("phone", r"(?:Мобільний\s+)?телефон"),
            ("email", r"Електронна\s+пошта"),
        ]

        label_compiled = [(field, re.compile(r"^" + pat + r"\s*:?\s*(.*)$", re.IGNORECASE)) for field, pat in label_specs]
        any_label_re = re.compile(r"^(?:" + "|".join(["(?:%s)" % pat for _, pat in label_specs]) + r")\b", re.IGNORECASE)
        forbidden_values = {
            "мобільний телефон",
            "телефон",
            "електронна пошта",
            "громадянство",
            "посада",
            "науковий ступінь",
        }

        def _extract_value_window(start_idx: int, inline_value: str) -> str:
            values: List[str] = []
            if inline_value:
                values.append(inline_value.strip())

            # Take only a small local window to avoid cross-field bleed.
            for j in range(start_idx + 1, min(len(lines), start_idx + 4)):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                if any_label_re.search(candidate):
                    break
                # Stop on obvious table/header-like labels.
                if candidate.isupper() and len(candidate.split()) <= 4:
                    break
                values.append(candidate)
                if len(values) >= 2:
                    break

            combined = re.sub(r"\s+", " ", " ".join(values)).strip()
            return re.sub(r"^\-\s*", "", combined).strip()

        def _truncate_after_markers(value: str, markers: List[str]) -> str:
            if not value:
                return value
            truncated = value
            for marker in markers:
                truncated = re.sub(marker + r".*$", "", truncated, flags=re.IGNORECASE).strip()
            return truncated

        extracted_values: Dict[str, str] = {}
        for i, line in enumerate(lines):
            for field, cre in label_compiled:
                if field in extracted_values:
                    continue
                m = cre.match(line)
                if not m:
                    continue
                value = _extract_value_window(i, m.group(1) or "")
                if value:
                    extracted_values[field] = value

        for field, value in extracted_values.items():
            norm_value = value.strip()
            if not norm_value:
                continue

            low = norm_value.lower()
            if low in forbidden_values:
                continue

            if field == "orcid":
                norm_orcid, raw_orcid = SemanticValidator.normalize_orcid(norm_value)
                if norm_orcid:
                    pi_data["orcid"] = norm_orcid
                    pi_data["orcid_raw"] = norm_orcid
            elif field == "phone":
                norm_value = _truncate_after_markers(norm_value, [r"\bE-?mail\b", r"\bЕлектронна\s+пошта\b", r"\bПерсональна\s+інтернет-сторінка\b"])
                if SemanticValidator.is_phone(norm_value):
                    pi_data[field] = norm_value
            elif field == "email":
                norm_value = _truncate_after_markers(norm_value, [r"\bПерсональна\s+інтернет-сторінка\b", r"\bORCID\b", r"\bМобільний\s+телефон\b"])
                if SemanticValidator.is_email(norm_value):
                    pi_data[field] = norm_value
            elif field == "citizenship":
                norm_value = _truncate_after_markers(norm_value, [r"\bМобільний\s+телефон\b", r"\bЕлектронна\s+пошта\b", r"\bПосада\b"])
                if SemanticValidator.is_citizenship(norm_value):
                    pi_data[field] = norm_value
            elif field == "position":
                # Reject obvious label noise in position value.
                norm_value = _truncate_after_markers(norm_value, [r"\bМобільний\s+телефон\b", r"\bЕлектронна\s+пошта\b", r"\bПерсональна\s+інтернет-сторінка\b"])
                if not any(kw in low for kw in ["телефон", "пошта", "громадянство"]):
                    pi_data[field] = norm_value
            else:
                pi_data[field] = norm_value
        
        if not pi_data.get("degree") and pi_data.get("degree_raw"):
            pi_data["degree"] = pi_data["degree_raw"]

        structure.metadata["pi_profile"] = pi_data

        # Update ONLY the first profile_block to contain full metadata
        profile_blocks = [b for b in zone.blocks if b.block_type == "profile_block"]
        if profile_blocks:
            main_block = profile_blocks[0]
            main_block.metadata = pi_data
            main_block.text = "[STRUCTURED DATA]"
            
            # Remove redundant profile blocks
            zone.blocks = [b for b in zone.blocks if b.block_type != "profile_block" or b == main_block]

    def _extract_team_profiles(self, zone: DocumentZone, pages: List[str], structure: ParsedDocumentStructure):
        """Extract co-author list from Zone 6 and flatten into individual profile entities."""
        if not structure.metadata: structure.metadata = {}
        team = []
        seen_names = set()
        seen_orcids: Dict[str, int] = {}
        seen_name_companion: Dict[Tuple[str, str], int] = {}
        seen_short_companion: Dict[Tuple[str, str], int] = {}
        
        # We need to collect ALL profile information and then deduplicate
        all_text = "\n".join([b.text for b in zone.blocks])
        
        # Detect individual profile entries
        # Profiles usually start with "Пан" or "Пані" or "Професор" or "Доктор" or a name in Zone 6
        split_pattern = r"(?=(?:Пан|Пані|Професор|Доктор|К\.т\.н\.|Д\.т\.н\.|Ph\.D\.)\s+[А-ЯЄІЇҐ][а-яєіїґ']+\s+[А-ЯЄІЇҐ][а-яєіїґ']+)"
        profile_splits = re.split(split_pattern, all_text)
        
        # Count potential profile starts for completeness check
        detected_starts = len(re.findall(split_pattern, all_text))
        unresolved_starts = []
        unresolved_pages = []
        
        # Track pages for unresolved starts
        current_char = 0
        for m in re.finditer(split_pattern, all_text):
            start_pos = m.start()
            # Find page number for this position
            p_num = zone.page_start
            char_acc = 0
            for b in zone.blocks:
                char_acc += len(b.text) + 1
                if char_acc > start_pos:
                    p_num = b.page_number
                    break
            # We'll use this later to find unresolved ones
            
        def _normalize_identity_text(value: Optional[str]) -> str:
            if not value:
                return ""
            normalized = value.strip().lower().replace("’", "'")
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = re.sub(r"[^\w\s\-']", "", normalized)
            return normalized

        def _short_name_signature(name_value: str) -> str:
            parts = [p for p in name_value.split() if p]
            if len(parts) < 2:
                return ""
            surname = parts[0]
            initials = "".join(p[0] for p in parts[1:] if p and p[0].isalpha())
            return f"{surname} {initials}".strip()

        def _extract_entry_fields(entry_text: str) -> Dict[str, str]:
            entry_lines = [re.sub(r"\s+", " ", l.strip()) for l in entry_text.split("\n") if l.strip()]
            specs = [
                ("orcid", r"ORCID"),
                ("degree_raw", r"Науковий\s+ступінь"),
                ("position", r"Посада"),
                ("institution", r"Місце\s+роботи"),
            ]
            compiled = [(f, re.compile(r"^" + p + r"\s*:?\s*(.*)$", re.IGNORECASE)) for f, p in specs]
            any_label = re.compile(r"^(?:" + "|".join(["(?:%s)" % p for _, p in specs]) + r")\b", re.IGNORECASE)
            extracted: Dict[str, str] = {}

            for i, line in enumerate(entry_lines):
                for field, cre in compiled:
                    if field in extracted:
                        continue
                    m = cre.match(line)
                    if not m:
                        continue
                    values: List[str] = []
                    inline = (m.group(1) or "").strip()
                    if inline:
                        values.append(inline)
                    for j in range(i + 1, min(len(entry_lines), i + 4)):
                        nxt = entry_lines[j]
                        if any_label.search(nxt):
                            break
                        if nxt.isupper() and len(nxt.split()) <= 5:
                            break
                        values.append(nxt)
                        if len(values) >= 2:
                            break
                    value = re.sub(r"\s+", " ", " ".join(values)).strip()
                    value = re.sub(r"^\-\s*", "", value).strip()
                    if value:
                        extracted[field] = value
            return extracted

        def _trim_tail(value: str, patterns: List[str]) -> str:
            if not value:
                return value
            out = value
            for pat in patterns:
                out = re.sub(pat + r".*$", "", out, flags=re.IGNORECASE).strip()
            return out

        def _looks_like_label_noise(value: str) -> bool:
            low = value.strip().lower()
            if not low:
                return True
            if low in {"та посада", "orcid", "місце роботи", "науковий ступінь", "посада", "період роботи"}:
                return True
            if "мінімум два" in low or "google scholar" in low or "scopus authors" in low:
                return True
            return False

        def _is_institution_value(value: str) -> bool:
            if not value or _looks_like_label_noise(value):
                return False
            normalized = re.sub(r"^\s*та\s+посада\b[:\s\-]*", "", value, flags=re.IGNORECASE).strip()
            low = normalized.lower()
            inst_markers = ["університет", "інститут", "академ", "нац", "нан україни", "україни"]
            return any(m in low for m in inst_markers)

        def _is_position_value(value: str) -> bool:
            if not value or _looks_like_label_noise(value):
                return False
            low = value.lower()
            bad_markers = ["google scholar", "scopus", "orcid", "мінімум два", "та посада"]
            if any(m in low for m in bad_markers):
                return False
            # Position is usually role-like, not institution name.
            if _is_institution_value(value):
                return False
            return True

        for entry in profile_splits:
            if not entry.strip(): continue
            
            # Extract name
            name_match = re.search(r"(?:Пан|Пані|Професор|Доктор|К\.т\.н\.|Д\.т\.н\.|Ph\.D\.)\s+([А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+\s*[А-ЯЄІЇҐ]?[а-яєіїґ]*)", entry)
            if not name_match:
                # Fallback for names without honorifics - look for Uppercase Word + Uppercase Word
                name_match = re.search(r"^([А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+(?:\s+[А-ЯЄІЇҐ][а-яєіїґ]+)?)", entry.strip())
            
            if name_match:
                raw_name = name_match.group(1).strip()
                name, title, degree = SemanticValidator.clean_name_and_get_titles(name_match.group(0))
                norm_name = _normalize_identity_text(name)
                
                if 5 < len(name) < 100:
                    # Extract other fields for this person if possible
                    person_data = {
                        "name": name,
                        "title": title,
                        "degree": degree
                    }

                    extracted = _extract_entry_fields(entry)
                    orcid_val = extracted.get("orcid")
                    if orcid_val:
                        norm_orcid, raw_orcid = SemanticValidator.normalize_orcid(orcid_val)
                        if norm_orcid:
                            person_data["orcid"] = norm_orcid
                            person_data["orcid_raw"] = norm_orcid
                        else:
                            person_data["orcid"] = None
                            person_data["orcid_raw"] = None

                    degree_raw = extracted.get("degree_raw")
                    if degree_raw and not _looks_like_label_noise(degree_raw):
                        person_data["degree_raw"] = degree_raw

                    position_val = extracted.get("position")
                    institution_val = extracted.get("institution")
                    if position_val:
                        position_val = _trim_tail(position_val, [r"\bПеріод\s+роботи\b", r"\bМісце\s+роботи\b", r"\bORCID\b"])
                    if institution_val:
                        institution_val = re.sub(r"^\s*ТА\s+ПОСАДА\b[:\s\-]*", "", institution_val, flags=re.IGNORECASE).strip()
                        institution_val = _trim_tail(institution_val, [r"\bПеріод\s+роботи\b", r"\bПосада\b", r"\bORCID\b"])

                    # Conservative anti-bleed assignment.
                    if position_val and _is_position_value(position_val):
                        person_data["position"] = position_val
                    if institution_val and _is_institution_value(institution_val):
                        person_data["institution"] = institution_val

                    # Swap/fix if values landed in opposite field.
                    if not person_data.get("institution") and position_val and _is_institution_value(position_val):
                        person_data["institution"] = position_val
                    if not person_data.get("position") and institution_val and _is_position_value(institution_val):
                        person_data["position"] = institution_val
                    
                    # Ensure degree is populated if found in labels but not in name
                    if not person_data.get("degree") and person_data.get("degree_raw"):
                        person_data["degree"] = person_data["degree_raw"]

                    orcid_norm = _normalize_identity_text(person_data.get("orcid"))
                    companion_parts = [
                        _normalize_identity_text(person_data.get("position")),
                        _normalize_identity_text(person_data.get("degree")),
                        _normalize_identity_text(person_data.get("institution")),
                    ]
                    companion_key = "|".join([p for p in companion_parts if p])
                    short_name = _short_name_signature(norm_name)

                    duplicate_idx = None
                    if orcid_norm and orcid_norm in seen_orcids:
                        duplicate_idx = seen_orcids[orcid_norm]
                    elif companion_key and (norm_name, companion_key) in seen_name_companion:
                        duplicate_idx = seen_name_companion[(norm_name, companion_key)]
                    elif companion_key and short_name and (short_name, companion_key) in seen_short_companion:
                        # Conservative variant merge: same companion fields + same surname/initials signature.
                        duplicate_idx = seen_short_companion[(short_name, companion_key)]
                    elif norm_name in seen_names:
                        duplicate_idx = next(
                            (idx for idx, existing in enumerate(team)
                             if _normalize_identity_text(existing.get("name")) == norm_name),
                            None
                        )

                    if duplicate_idx is None:
                        team.append(person_data)
                        idx = len(team) - 1
                        seen_names.add(norm_name)
                        if orcid_norm:
                            seen_orcids[orcid_norm] = idx
                        if companion_key:
                            seen_name_companion[(norm_name, companion_key)] = idx
                            if short_name:
                                seen_short_companion[(short_name, companion_key)] = idx
                    else:
                        # Merge missing deterministic fields into the existing canonical profile.
                        existing = team[duplicate_idx]
                        for field_name, field_value in person_data.items():
                            if field_value and not existing.get(field_name):
                                existing[field_name] = field_value
                else:
                    # Could be a duplicate or invalid name
                    pass
            else:
                # Unresolved start
                unresolved_starts.append(entry[:50].strip())
                # Find page for this entry if possible
                
        structure.metadata["team_profiles"] = team
        structure.metadata["team_profile_completeness"] = {
            "detected_starts": detected_starts,
            "extracted_entities": len(team),
            "unresolved_starts": unresolved_starts,
            "unresolved_pages": list(set(unresolved_pages))
        }

        # RECONSTRUCT BLOCKS: One profile_block per person
        new_blocks = []
        # Keep non-profile blocks
        for b in zone.blocks:
            if b.block_type != "profile_block":
                new_blocks.append(b)
        
        # Add exactly one profile_block per person
        for person in team:
            # Deterministic ID for profile blocks to ensure they aren't duplicated by mistake
            new_blocks.append(DocumentBlock(
                block_type="profile_block",
                text="[STRUCTURED DATA]",
                metadata=person,
                page_number=zone.page_start,
                # Use name in text to make it unique for deduplication tests if needed
                # although metadata is already there.
            ))
            
        zone.blocks = new_blocks

    def _detect_annex_subtypes(self, zone: DocumentZone, pages: List[str], structure: ParsedDocumentStructure):
        """Detect subtypes of documents in Zone 7 (Annexes)."""
        if not structure.metadata: structure.metadata = {}
        annexes = []
        
        subtype_patterns = {
            "scientific_certificate": r"ДОВІДКА\s+про\s+наукову\s+та\s+науково-технічну\s+діяльність",
            "compliance_certificate": r"ДОВІДКА\s+про\s+відповідність\s+учасника\s+конкурсу",
            "grant_application": r"ЗАЯВА\s+на\s+отримання\s+грантової\s+підтримки",
            "consent_form": r"ЗГОДА\s+на\s+участь\s+у\s+виконанні\s+проєкту",
            "ethics_form": r"Етичні\s+питання\s+проєкту",
        }
        
        for p_idx in range(zone.page_start - 1, zone.page_end):
            p_text = pages[p_idx]
            for subtype, pattern in subtype_patterns.items():
                if re.search(pattern, p_text, re.IGNORECASE):
                    annexes.append({
                        "type": subtype,
                        "page": p_idx + 1,
                        "title": re.search(pattern, p_text, re.IGNORECASE).group(0)
                    })

        marker_pages = structure.metadata.get("marker_pages", []) if structure.metadata else []
        annex_marker_pages = []
        certificate_marker_pages = []
        for anchor in structure.detected_anchors:
            if not isinstance(anchor, dict):
                continue
            text = (anchor.get("text") or "").strip()
            if text == "Додатки":
                annex_marker_pages.append(anchor.get("page"))
            elif text == "Довідки":
                certificate_marker_pages.append(anchor.get("page"))

        zone_pages = list(range(zone.page_start, zone.page_end + 1))
        empty_content_pages = []
        content_text_pages = []
        page_status = []
        for p in zone_pages:
            page_text = pages[p - 1] if 0 <= (p - 1) < len(pages) else ""
            has_text = bool(page_text and page_text.strip())
            if has_text:
                content_text_pages.append(p)
                page_status.append({"page": p, "status": "text_content_detected"})
            else:
                empty_content_pages.append(p)
                page_status.append({"page": p, "status": "no_text_extracted"})

        annex_status = {
            "semantics_mode": "single_zone_labeled_by_markers",
            "marker_pages": {
                "annex": sorted([p for p in annex_marker_pages if isinstance(p, int)]),
                "certificate": sorted([p for p in certificate_marker_pages if isinstance(p, int)]),
            },
            "zone_pages": zone_pages,
            "content_text_pages": content_text_pages,
            "no_text_extracted_pages": empty_content_pages,
            "page_status": page_status,
            "marker_only_detected": bool(sorted(set(marker_pages) & set(annex_marker_pages + certificate_marker_pages))),
            "no_text_extracted": len(content_text_pages) == 0,
            "scanned_or_image_only_suspected": len(content_text_pages) == 0 and len(zone_pages) > 0,
        }

        structure.metadata["annex_subtypes"] = annexes
        structure.metadata["annex_status"] = annex_status

        # Update blocks
        for block in zone.blocks:
            if block.block_type == "scan_block":
                block.metadata = {
                    "annex_subtypes": annexes,
                    "annex_status": annex_status
                }

    def _recursive_clean_blocks(self, section: DocumentSection):
        """Recursively clean blocks in section and its subsections."""
        header_noise_patterns = [
            r"Національний\s+фонд\s+досліджень\s+України",
            r"Конкурс\s+проєктів\s+із\s+виконання\s+наукових\s+досліджень\s+і\s+розробок",
            r"Передова\s+наука\s+в\s+Україні",
            r"Реєстраційний\s+номер\s+проєкту:?\s*[\d\.\/]+",
            r"\"Передова\s+наука\s+в\s+Україні\"",
            r"Сторінка\s+\d+\s+із\s+\d+",
        ]
        
        cleaned_blocks = []
        for block in section.blocks:
            # 1. Structured data cleaning
            if block.block_type in ["metadata_block", "profile_block"] and block.metadata:
                block.text = "[STRUCTURED DATA]"
            
            # 2. Header noise cleaning (final pass)
            is_noise = False
            for pat in header_noise_patterns:
                if re.search(pat, block.text, re.IGNORECASE):
                    # If the block contains ONLY noise, we might want to discard it
                    # If it contains noise + content, we might want to scrub it.
                    # For simplicity, if it's a short block matching noise, discard.
                    if len(block.text) < 200:
                        is_noise = True
                        break
            
            if not is_noise:
                cleaned_blocks.append(block)
                
        section.blocks = cleaned_blocks
        
        for sub in section.subsections:
            self._recursive_clean_blocks(sub)

    def _save_debug_json(self, result: Dict[str, Any], filename: Optional[str] = None):
        """Save parsing result to a debug folder."""
        debug_dir = "debug/parsing"
        os.makedirs(debug_dir, exist_ok=True)
        if not filename:
            filename = f"parsing_debug_{os.getpid()}.json"
        with open(os.path.join(debug_dir, filename), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _legacy_extract_all_sections(self, doc: fitz.Document, pages_text: List[str]) -> List[Dict[str, Any]]:
        """Legacy method to extract all sections from all pages."""
        sections = []
        for page_num in range(len(pages_text)):
            page = doc.load_page(page_num)
            page_sections = self._extract_page_sections(page)
            for section in page_sections:
                section["page"] = page_num + 1
                sections.append(section)
        return sections

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
            "signature": [],
            "pi": [],
            "coauthor": []
        }
        
        budget_patterns = [r"бюджет", r"budget", r"кошторис", r"finances"]
        annex_patterns = [r"додаток", r"annex", r"appendix"]
        signature_patterns = [r"підпис", r"signature", r"signed", r"мп"]
        pi_patterns = [r"керівник проєкту", r"project leader", r"principal investigator", r"піб"]
        coauthor_patterns = [r"виконавці", r"co-authors", r"team members"]
        
        for i, page_text in enumerate(pages_text):
            page_num = i + 1
            low_text = page_text.lower()
            
            if any(re.search(p, low_text) for p in budget_patterns):
                markers["budget"].append(page_num)
            if any(re.search(p, low_text) for p in annex_patterns):
                markers["annex"].append(page_num)
            if any(re.search(p, low_text) for p in signature_patterns):
                markers["signature"].append(page_num)
            if any(re.search(p, low_text) for p in pi_patterns):
                markers["pi"].append(page_num)
            if any(re.search(p, low_text) for p in coauthor_patterns):
                markers["coauthor"].append(page_num)
                
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

    def _extract_zone1_metadata(self, zone: Optional[DocumentZone], pages_text: List[str]) -> Dict[str, str]:
        """Deterministically extract structured fields from Zone 1 (header/registration)."""
        if not zone:
            return {}

        lines = [b.text.strip() for b in zone.blocks if b.text and b.text.strip()]
        if not lines:
            return {}

        field_specs = [
            ("research_type", r"^Характер\s+досліджень\s*:?\s*(.*)$"),
            ("grant_type", r"^Вид\s+грантової\s+підтримки\s*:?\s*(.*)$"),
            ("support_direction", r"^Напрям\s+грантової\s+підтримки\s*:?\s*(.*)$"),
            ("scientific_direction", r"^Науковий\s+напрям\s*:?\s*(.*)$"),
            ("speciality", r"^Спеціальність\s*:?\s*(.*)$"),
            ("application_id", r"^Реєстраційний\s+номер\s+проєкту\s*:?\s*(.*)$"),
        ]

        label_patterns = [re.compile(pat, re.IGNORECASE) for _, pat in field_specs]

        def _normalize_value(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", value).strip()
            cleaned = re.sub(r"^\-\s*", "", cleaned).strip()
            return cleaned

        extracted: Dict[str, str] = {}
        for idx, line in enumerate(lines):
            for field, pattern in field_specs:
                if field in extracted:
                    continue

                match = re.match(pattern, line, re.IGNORECASE)
                if not match:
                    continue

                value = _normalize_value(match.group(1) or "")
                if not value and idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    is_next_label = any(lp.match(next_line) for lp in label_patterns)
                    if not is_next_label:
                        value = _normalize_value(next_line)

                if value:
                    extracted[field] = value

        # Deterministic multiline extraction for "Тематичний напрям конкурсу"
        thematic_anchor_re = re.compile(r"^Тематичний\s+напрям\s+конкурсу\s*:?\s*(.*)$", re.IGNORECASE)
        thematic_stop_res = [
            re.compile(r"^Характер\s+досліджень\b", re.IGNORECASE),
            re.compile(r"^Вид\s+грантової\s+підтримки\b", re.IGNORECASE),
            re.compile(r"^Напрям\s+грантової\s+підтримки\b", re.IGNORECASE),
            re.compile(r"^Науковий\s+напрям\b", re.IGNORECASE),
            re.compile(r"^Спеціальність\b", re.IGNORECASE),
            re.compile(r"^Реєстраційний\s+номер\s+проєкту\b", re.IGNORECASE),
            re.compile(r"^Назва\s+конкурсу\b", re.IGNORECASE),
        ]
        for i, line in enumerate(lines):
            match = thematic_anchor_re.match(line)
            if not match:
                continue

            collected = []
            inline = _normalize_value(match.group(1) or "")
            if inline:
                collected.append(inline)

            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if any(stop_re.search(nxt) for stop_re in thematic_stop_res):
                    break
                if not nxt:
                    continue
                collected.append(_normalize_value(nxt))

            thematic_value = re.sub(r"\s+", " ", " ".join(collected)).strip()
            if thematic_value:
                extracted["competition_thematic_direction"] = thematic_value
            break

        if "application_id" not in extracted:
            search_text = "\n".join(lines)
            fallback_id = self._extract_registration_number(search_text)
            if not fallback_id and pages_text:
                fallback_id = self._extract_registration_number(pages_text[0])
            if fallback_id:
                extracted["application_id"] = fallback_id

        return extracted

    def _extract_financial_summary(self, zone: DocumentZone, structure: ParsedDocumentStructure):
        """Extract deterministic structured financial fields from Zone 3 table blocks."""
        if not structure.metadata:
            structure.metadata = {}

        lines = [re.sub(r"\s+", " ", (b.text or "").strip()) for b in zone.blocks if (b.text or "").strip()]
        if not lines:
            structure.metadata["financial_summary"] = {}
            return

        def _parse_amount(text: str) -> Optional[int]:
            match = re.search(r"(\d[\d\s,\.]*)", text)
            if not match:
                return None
            raw = match.group(1)
            cleaned = re.sub(r"[^\d]", "", raw)
            if not cleaned:
                return None
            try:
                return int(cleaned)
            except ValueError:
                return None

        def _find_next_amount(start_idx: int, lookahead: int = 3) -> Tuple[Optional[int], Optional[str]]:
            end = min(len(lines), start_idx + lookahead + 1)
            for j in range(start_idx + 1, end):
                amount = _parse_amount(lines[j])
                if amount is not None:
                    return amount, lines[j]
            return None, None

        def _is_non_numeric_label(text: str) -> bool:
            if not text:
                return False
            low = text.lower()
            if "обсяг фінансування" in low or "термін реалізації" in low:
                return True
            return _parse_amount(text) is None

        def _to_cell(col_id: str, text_value: str, normalized_value: Any = None) -> CanonicalTableCell:
            return CanonicalTableCell(col_id=col_id, text=text_value, normalized=normalized_value)

        def _build_table_rows(entries: List[Tuple[str, Optional[str], Optional[Any]]]) -> List[CanonicalTableRow]:
            rows: List[CanonicalTableRow] = []
            for idx, (label, raw_value, norm_value) in enumerate(entries):
                value_text = raw_value or ""
                rows.append(
                    CanonicalTableRow(
                        row_id=f"r{idx}",
                        cells=[
                            _to_cell("c0", label, label),
                            _to_cell("c1", value_text, norm_value),
                        ],
                    )
                )
            return rows

        def _extract_year_stage(label_text: str) -> Tuple[Optional[int], Optional[int]]:
            low = (label_text or "").lower()
            year_num = None
            if "перш" in low:
                year_num = 1
            elif "друг" in low:
                year_num = 2
            elif "трет" in low:
                year_num = 3
            stage_match = re.search(r"\bетап\s*(\d+)\b", low)
            if not stage_match:
                stage_match = re.search(r"(\d+)\s*[-–—]?\s*етап\b", low)
            stage_num = int(stage_match.group(1)) if stage_match else None
            return year_num, stage_num

        # Split financial lines into summary and stage segments using the in-page heading.
        summary_heading_idx = next((i for i, l in enumerate(lines) if re.search(r"^обсяг\s+фінансування\s*$", l, re.IGNORECASE)), None)
        stage_heading_idx = next((i for i, l in enumerate(lines) if re.search(r"^етапи\s+фінансування\s*$", l, re.IGNORECASE)), None)
        summary_lines = lines[:]
        stage_lines: List[str] = []

        if summary_heading_idx is not None:
            summary_lines = lines[summary_heading_idx + 1:]
        if stage_heading_idx is not None:
            stage_lines = lines[stage_heading_idx + 1:]
            if summary_heading_idx is not None and stage_heading_idx > summary_heading_idx:
                summary_lines = lines[summary_heading_idx + 1:stage_heading_idx]
            else:
                summary_lines = [l for l in lines if not re.search(r"^етапи\s+фінансування\s*$", l, re.IGNORECASE)]

        # Build summary table rows (label -> value).
        summary_entries: List[Tuple[str, Optional[str], Optional[Any]]] = []
        i = 0
        while i < len(summary_lines):
            label = summary_lines[i]
            low = label.lower()
            if "обсяг фінансування" in low or re.search(r"^термін\s+реалізації\s+проєкту", low):
                raw_value = summary_lines[i + 1] if i + 1 < len(summary_lines) else None
                norm_value: Any = raw_value
                parsed = _parse_amount(raw_value or "")
                if parsed is not None:
                    norm_value = parsed
                summary_entries.append((label, raw_value, norm_value))
                i += 2
                continue
            i += 1

        # Build stage table rows (label -> amount, when available).
        stage_entries: List[Tuple[str, Optional[str], Optional[Any]]] = []
        i = 0
        while i < len(stage_lines):
            label = stage_lines[i]
            low = label.lower()
            if "етап" in low and "обсяг фінансування" in low:
                raw_value = None
                norm_value: Any = None
                if i + 1 < len(stage_lines) and not _is_non_numeric_label(stage_lines[i + 1]):
                    raw_value = stage_lines[i + 1]
                    parsed = _parse_amount(raw_value)
                    if parsed is not None:
                        norm_value = parsed
                    i += 2
                else:
                    i += 1
                stage_entries.append((label, raw_value, norm_value))
                continue
            i += 1

        page_cls = next((c for c in structure.page_table_classifications if c.page_number == zone.page_start), None)
        page_class = page_cls.page_class if page_cls else "native_text"
        page_confidence = page_cls.confidence if page_cls else 0.8
        source = {
            "parser": "pymupdf_native",
            "page_class": page_class,
            "confidence": page_confidence,
        }

        summary_table = CanonicalTable(
            table_id=f"tbl_financial_summary_p{zone.page_start:03d}_01",
            table_family="financial_funding_summary",
            zone_type="financial",
            title="Обсяг фінансування",
            page_start=zone.page_start,
            page_end=zone.page_end or zone.page_start,
            source=source,
            columns=[
                CanonicalTableColumn(col_id="c0", name="label", semantic_type="string"),
                CanonicalTableColumn(col_id="c1", name="value", semantic_type="string_or_amount"),
            ],
            rows=_build_table_rows(summary_entries),
            spans=[],
            validation={
                "row_count": len(summary_entries),
                "column_count": 2,
                "normalization_warnings": [],
            },
        )

        stage_table = CanonicalTable(
            table_id=f"tbl_financial_stages_p{zone.page_start:03d}_02",
            table_family="financial_stage_amounts",
            zone_type="financial",
            title="Етапи фінансування",
            page_start=zone.page_start,
            page_end=zone.page_end or zone.page_start,
            source=source,
            columns=[
                CanonicalTableColumn(col_id="c0", name="stage_label", semantic_type="string"),
                CanonicalTableColumn(col_id="c1", name="amount_uah", semantic_type="currency_uah_int"),
            ],
            rows=_build_table_rows(stage_entries),
            spans=[],
            validation={
                "row_count": len(stage_entries),
                "column_count": 2,
                "normalization_warnings": [],
            },
        )

        yearly_rows: List[CanonicalTableRow] = []
        for idx, (label, raw_value, norm_value) in enumerate(summary_entries):
            if "рік" not in (label or "").lower():
                continue
            year_num, stage_num = _extract_year_stage(label)
            yearly_rows.append(
                CanonicalTableRow(
                    row_id=f"r{idx}",
                    cells=[
                        CanonicalTableCell(col_id="c0", text=label, normalized=label),
                        CanonicalTableCell(col_id="c1", text=str(year_num) if year_num is not None else "", normalized=year_num),
                        CanonicalTableCell(col_id="c2", text=raw_value or "", normalized=norm_value),
                        CanonicalTableCell(col_id="c3", text=str(stage_num) if stage_num is not None else "", normalized=stage_num),
                    ],
                )
            )

        yearly_stage_table = CanonicalTable(
            table_id=f"tbl_financial_yearly_stages_p{zone.page_start:03d}_03",
            table_family="financial_yearly_stage_amounts",
            zone_type="financial",
            title="Річний розподіл фінансування за етапами",
            page_start=zone.page_start,
            page_end=zone.page_end or zone.page_start,
            source=source,
            columns=[
                CanonicalTableColumn(col_id="c0", name="year_stage_label", semantic_type="string"),
                CanonicalTableColumn(col_id="c1", name="year_number", semantic_type="int"),
                CanonicalTableColumn(col_id="c2", name="amount_uah", semantic_type="currency_uah_int"),
                CanonicalTableColumn(col_id="c3", name="stage_number", semantic_type="int"),
            ],
            rows=yearly_rows,
            spans=[],
            validation={
                "row_count": len(yearly_rows),
                "column_count": 4,
                "normalization_warnings": [],
            },
        )

        # Replace previous financial tables for this page (idempotent reruns), then append canonical tables.
        structure.tables = [
            t for t in structure.tables
            if not (
                t.zone_type == "financial"
                and t.page_start == zone.page_start
                and t.table_family in {
                    "financial_funding_summary",
                    "financial_stage_amounts",
                    "financial_yearly_stage_amounts",
                }
            )
        ]
        structure.tables.extend([summary_table, stage_table, yearly_stage_table])

        # Derive legacy financial_summary from canonical rows where available.
        stage_amounts: Dict[str, int] = {}
        stage_amounts_raw: Dict[str, str] = {}
        total_amount_uah: Optional[int] = None
        total_amount_raw: Optional[str] = None
        project_duration: Optional[str] = None

        for row in summary_table.rows:
            label = (row.cells[0].text if row.cells else "").lower()
            raw_value = row.cells[1].text if len(row.cells) > 1 else ""
            normalized = row.cells[1].normalized if len(row.cells) > 1 else None

            if "термін реалізації проєкту" in label and raw_value:
                project_duration = raw_value

            if re.search(r"^обсяг\s+фінансування\s+проєкту\s*$", label):
                if isinstance(normalized, int):
                    total_amount_uah = normalized
                    total_amount_raw = raw_value

            if "обсяг фінансування" in label:
                stage_num = None
                if re.search(r"\bперш", label) or re.search(r"\bетап\s*1\b", label):
                    stage_num = "1"
                elif re.search(r"\bдруг", label) or re.search(r"\bетап\s*2\b", label):
                    stage_num = "2"
                elif re.search(r"\bтрет", label) or re.search(r"\bетап\s*3\b", label):
                    stage_num = "3"
                if stage_num and isinstance(normalized, int):
                    stage_amounts[stage_num] = normalized
                    stage_amounts_raw[stage_num] = raw_value

        # Backward-compatible fallback to prior line-neighbor logic when canonical rows are incomplete.
        if total_amount_uah is None or not stage_amounts:
            for i, line in enumerate(lines):
                low = line.lower()
                if project_duration is None and re.search(r"^термін\s+реалізації\s+проєкту", low):
                    if i + 1 < len(lines):
                        project_duration = lines[i + 1]
                if total_amount_uah is None and re.search(r"^обсяг\s+фінансування\s+проєкту\s*$", low):
                    amt, raw_line = _find_next_amount(i)
                    if amt is not None:
                        total_amount_uah = amt
                        total_amount_raw = raw_line
                if "обсяг фінансування" in low:
                    stage_num = None
                    if re.search(r"\bперш", low) or re.search(r"\bетап\s*1\b", low):
                        stage_num = "1"
                    elif re.search(r"\bдруг", low) or re.search(r"\bетап\s*2\b", low):
                        stage_num = "2"
                    elif re.search(r"\bтрет", low) or re.search(r"\bетап\s*3\b", low):
                        stage_num = "3"
                    if stage_num and stage_num not in stage_amounts:
                        amt, raw_line = _find_next_amount(i)
                        if amt is not None:
                            stage_amounts[stage_num] = amt
                            stage_amounts_raw[stage_num] = raw_line or ""

        financial_summary = {
            "total_amount_uah": total_amount_uah,
            "total_amount_raw": total_amount_raw,
            "project_duration": project_duration,
            "stage_amounts_uah": stage_amounts,
            "stage_amounts_raw": stage_amounts_raw,
        }
        structure.metadata["financial_summary"] = financial_summary

        first_table_block = next((b for b in zone.blocks if b.block_type == "table_block"), None)
        if first_table_block:
            first_table_block.metadata = financial_summary

    def _cleanup_zone1_blocks(self, zone: DocumentZone, z1_meta: Dict[str, str], project_title: Optional[str] = None):
        """
        Remove Zone 1 metadata anchors and already-consumed value lines
        from ordinary paragraph output to reduce semantic noise.
        """
        anchor_patterns = [
            r"^Назва\s+конкурсу$",
            r"^Тематичний\s+напрям\s+конкурсу$",
            r"^Характер\s+досліджень$",
            r"^Вид\s+грантової\s+підтримки$",
            r"^Напрям\s+грантової\s+підтримки$",
            r"^Науковий\s+напрям$",
            r"^Спеціальність$",
            r"^Реєстраційний\s+номер\s+проєкту$",
        ]
        anchor_res = [re.compile(p, re.IGNORECASE) for p in anchor_patterns]

        def _norm_line(text: str) -> str:
            t = re.sub(r"\s+", " ", (text or "").strip()).lower()
            t = re.sub(r"^\-\s*", "", t).strip()
            return t

        def _norm_title_fragment(text: str) -> str:
            t = (text or "").strip().lower()
            t = t.replace("’", "'").replace("`", "'").replace("`", "'")
            t = re.sub(r"[\"“”«»]", "", t)
            t = re.sub(r"[–—-]", " ", t)
            t = re.sub(r"[^\w\s']", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        consumed_values = set()
        consumed_multiline_values: List[str] = []
        for v in z1_meta.values():
            if isinstance(v, str) and v.strip():
                nv = _norm_line(v)
                consumed_values.add(nv)
                if len(nv) >= 30:
                    consumed_multiline_values.append(_norm_title_fragment(v))

        normalized_project_title = _norm_title_fragment(project_title or "")
        known_header_identity = {"нфду", "nrfu"}

        def _is_title_fragment(line_norm: str, title_norm: str) -> bool:
            if not line_norm or not title_norm or len(line_norm) < 10:
                return False
            if line_norm in title_norm:
                return True

            line_tokens = line_norm.split()
            title_tokens = title_norm.split()
            if len(line_tokens) < 3 or len(title_tokens) < 3:
                return False

            # Exact token-subsequence match.
            window = len(line_tokens)
            if window <= len(title_tokens):
                for i in range(0, len(title_tokens) - window + 1):
                    if title_tokens[i:i + window] == line_tokens:
                        return True

            # Prefix/suffix fragment match (handles line-broken title pieces).
            prefix_len = min(4, len(line_tokens), len(title_tokens))
            if prefix_len >= 3 and line_tokens[:prefix_len] == title_tokens[:prefix_len]:
                return True
            if prefix_len >= 3 and line_tokens[-prefix_len:] == title_tokens[-prefix_len:]:
                return True
            return False

        cleaned_blocks = []
        for block in zone.blocks:
            if block.block_type != "paragraph":
                cleaned_blocks.append(block)
                continue

            text = (block.text or "").strip()
            if not text:
                continue

            if any(cre.match(text) for cre in anchor_res):
                continue

            normalized_line = _norm_line(text)
            if normalized_line in consumed_values:
                continue

            # Header-identity markers (already represented structurally elsewhere).
            if normalized_line in known_header_identity:
                continue

            # Zone 1 title fragments already represented in extracted project_title.
            line_for_title = _norm_title_fragment(text)
            if _is_title_fragment(line_for_title, normalized_project_title):
                continue

            # Remove line fragments already consumed into long structured Zone 1 values.
            if (
                len(line_for_title) >= 20
                and any(line_for_title in mv for mv in consumed_multiline_values if mv)
            ):
                continue

            cleaned_blocks.append(block)

        zone.blocks = cleaned_blocks

    def _is_subsection_header(self, text: str) -> bool:
        """
        Detect subsection headers (e.g., "Анотація проєкту", "Мета", "Завдання").
        Rules:
        - starts with uppercase
        - no sentence-ending punctuation
        - must match explicit known header patterns
        - avoid classifying short narrative starters as headers
        """
        text = text.strip()
        if not text:
            return False

        words = text.split()
        if len(words) < 1 or len(words) > 8:
            return False

        if not text[0].isupper():
            return False

        if text.endswith(".") or text.endswith(":") or text.endswith(";") or text.endswith("?") or text.endswith("!"):
            return False

        # Guardrail against common sentence starters that were previously over-classified.
        banned_single_word = {"При", "Це", "Етап", "Виготовлення", "Можливість"}
        if len(words) == 1 and text in banned_single_word:
            return False

        # Explicit known NRFU subsection headers (narrative metadata anchors).
        explicit_headers = [
            r"Анотація\s+проєкту",
            r"Короткий\s+опис\s+проєкту",
            r"Ключові\s+слова",
            r"Мета(?:\s+наукового)?\s+проєкту",
            r"Мета",
            r"Основні\s+завдання\s+проєкту",
            r"Завдання(?:\s+проєкту)?",
            r"Актуальність\s+проєкту",
            r"Новизна\s+проєкту",
            r"Методи\s+досліджень",
            r"Очікувані\s+результати",
            r"Практичне\s+значення",
            r"Ризики",
            r"Дорожня\s+карта",
        ]

        for pat in explicit_headers:
            if re.search(r"^" + pat + r"$", text, re.IGNORECASE):
                return True

        # Numeric-form headers are already handled earlier as section_header (3.x patterns).
        return False

    def _is_foreign_structured_content(self, text: str) -> bool:
        """Detect questionnaire/profile/table patterns in narrative blocks."""
        patterns = [
            r"АНКЕТА\s+відповідності",
            r"Чи\s+має",
            r"Чи\s+був",
            r"Прізвище,\s+ім’я,\s+по\s+батькові",
            r"Науковий\s+ступінь",
            r"Вчене\s+звання",
            r"Місце\s+роботи",
            r"Посада",
            r"ORCID",
            r"Scopus\s+Author\s+ID",
            r"Researcher\s+ID",
            r"Google\s+Scholar",
            r"Цитування",
            r"Індекс\s+Хірша",
            r"Рік\s+захисту",
            r"Дата\s+народження",
            r"Контактний\s+телефон",
            r"Адреса\s+електронної\s+пошти",
        ]
        
        # Explicit form headings
        if re.search(r"АНКЕТА\s+відповідності", text, re.IGNORECASE):
            return True
            
        matches = 0
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                matches += 1
        
        # Increased threshold or explicit questionnaire wording
        if matches >= 2:
            return True
            
        # Checklist patterns like repeated yes/no or numbered criteria
        if re.search(r"Так\s*/\s*Ні", text, re.IGNORECASE) and len(text) < 200:
            return True
            
        return False

    def _extract_project_title(self, text: str) -> str:
        # Heuristic: search for "Назва проєкту" or "Project title"
        # NRFU PDFs often have project titles after a label on the same line or next line
        head = text[:5000] # Increased head for better coverage
        
        # In Master Spec: project title is between НФДУ and first separator, unlabeled.
        # But we also support labeled ones for robustness.
        
        # Try position-based extraction (unlabeled) first
        # Pattern: НФДУ ... (Title) ... Реєстраційний номер проєкту
        pos_match = re.search(r"НФДУ\s*(.*?)\s*Реєстраційний\s+номер\s+проєкту", head, re.DOTALL | re.IGNORECASE)
        if pos_match:
            title = pos_match.group(1).strip()
            # Clean up: remove call titles if they leaked in
            title = re.sub(r"Передова наука в Україні", "", title, flags=re.IGNORECASE).strip()
            if len(title) > 10:
                return re.sub(r"\s+", " ", title)

        # Fallback to labeled
        labels = [
            r"Назва проєкту", r"НАЗВА ПРОЄКТУ", 
            r"Project title", r"PROJECT TITLE",
            r"Назва теми проєкту", r"НАЗВА ТЕМИ ПРОЄКТУ",
            r"Тема проєкту", r"ТЕМА ПРОЄКТУ"
        ]
        
        for label in labels:
            # Try multi-line with negative lookahead for other fields first (most comprehensive)
            pattern_multi = rf"{label}[:\s]*((?:(?!(?:Project acronym|Acronym|Short name|Registration|Competition|Короткий|Абревіатура|Реєстраційний|Назва конкурсу|Бюджет|Budget)).)+)"
            match = re.search(pattern_multi, head, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 5 and not extracted.startswith("("):
                    return re.sub(r"\s+", " ", extracted)

            # Try same line match
            pattern = rf"{label}[:\s]*([^\n\r]+)"
            match = re.search(pattern, head, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 5 and not extracted.startswith("("):
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
