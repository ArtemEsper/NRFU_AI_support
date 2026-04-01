import fitz  # PyMuPDF
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import os
import tempfile
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4
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

    async def parse_pdf(
        self,
        file_content: bytes,
        input_file_name: Optional[str] = None,
        input_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            self._augment_scanned_tables_with_liteparse(file_content, structure, pages_text)
        
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
        
        debug_artifact_path: Optional[str] = None
        # Save full debug JSON if needed (includes full text)
        if os.getenv("DEBUG_PARSING") == "true":
            page_inspection_summary = self._build_page_inspection_summary(structure)
            self._save_debug_json(page_inspection_summary, filename="page_inspection_summary.json")

            debug_artifact = self._build_canonical_debug_artifact(
                parsing_result=parsing_result,
                input_file_name=input_file_name,
                input_file_path=input_file_path,
            )
            debug_artifact_path = self._save_canonical_debug_artifact(debug_artifact)

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
            "debug_artifact_path": debug_artifact_path,
            "parsing_result": parsing_result
        }

    def _classify_pages_for_tables(self, doc: fitz.Document, pages_text: List[str]) -> List[TablePageClassification]:
        classifications: List[TablePageClassification] = []
        page_rows: List[Dict[str, Any]] = []

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

            page_rows.append({
                "page_number": idx + 1,
                "page_class": page_class,
                "confidence": confidence,
                "word_count": word_count,
                "digit_ratio": digit_ratio,
                "image_area_ratio": image_area_ratio,
                "is_landscape": bool(page_rect.width > page_rect.height),
                "page_aspect_ratio": float(page_rect.width / max(page_rect.height, 1.0)),
                "has_financial_keyword": has_financial_keyword,
            })

        scanned_pages = [int(r["page_number"]) for r in page_rows if r["page_class"] == "scanned_image_only"]
        scanned_ocr_text = self._get_scanned_page_ocr_snippets(doc, scanned_pages)
        family_scoring_texts = list(pages_text)
        for page_num, snippet in scanned_ocr_text.items():
            idx = page_num - 1
            if 0 <= idx < len(family_scoring_texts) and snippet.strip():
                family_scoring_texts[idx] = snippet

        for row in page_rows:
            page_num = int(row["page_number"])
            page_idx = page_num - 1
            page_class = str(row["page_class"])
            page_confidence = float(row["confidence"])
            signals = {
                "word_count": int(row["word_count"]),
                "digit_ratio": round(float(row["digit_ratio"]), 4),
                "image_area_ratio": round(float(row["image_area_ratio"]), 4),
                "is_landscape": bool(row["is_landscape"]),
                "page_aspect_ratio": round(float(row["page_aspect_ratio"]), 4),
                "has_financial_keyword": bool(row["has_financial_keyword"]),
            }

            if page_class == "scanned_image_only":
                scored_text = family_scoring_texts[page_idx] if 0 <= page_idx < len(family_scoring_texts) else ""
                scored_word_count = len(re.findall(r"\w+", scored_text, flags=re.UNICODE))
                scored_digit_ratio = sum(ch.isdigit() for ch in scored_text) / max(len(scored_text), 1)
                family_hint, family_confidence, family_signals = self._classify_scanned_page_family(
                    page_index=page_idx,
                    pages_text=family_scoring_texts,
                    word_count=scored_word_count,
                    digit_ratio=scored_digit_ratio,
                )
                signals["scanned_family_hint"] = family_hint
                signals["scanned_family_confidence"] = round(family_confidence, 3)
                signals["scanned_family_signals"] = family_signals
                signals["scanned_family_text_source"] = (
                    "liteparse_ocr_snippet" if page_num in scanned_ocr_text and bool(scanned_ocr_text[page_num].strip()) else "native_extracted_text"
                )
                signals["scanned_family_ocr_text_len"] = len(scanned_ocr_text.get(page_num, ""))
                cert_evidence_ocr = self._extract_scanned_certificate_reference_evidence(scored_text)
                vision_candidate = bool(
                    settings.USE_SCANNED_CERT_VISION_EVIDENCE
                    and (
                        family_hint == "certificate_reference_evidence"
                        or bool(family_signals.get("certificate_keyword_detected"))
                        or bool(family_signals.get("primary_employment_phrase_detected"))
                        or float(family_signals.get("certificate_score", 0.0) or 0.0) >= 0.35
                    )
                )
                signals["certificate_vision_candidate"] = vision_candidate
                cert_evidence_vision: Dict[str, Any] = {}
                if vision_candidate:
                    cert_evidence_vision = self._extract_scanned_certificate_reference_evidence_vision(doc, page_num)
                    signals["certificate_vision_evidence_source"] = cert_evidence_vision.get("evidence_source", "none")
                merged_cert_evidence = self._merge_certificate_evidence(cert_evidence_ocr, cert_evidence_vision)
                if merged_cert_evidence:
                    signals["certificate_reference_evidence"] = merged_cert_evidence
                    signals["certificate_evidence_source"] = merged_cert_evidence.get("evidence_source") or "ocr_text_patterns"
                    signals["certificate_evidence_lanes"] = {
                        "ocr_available": bool(cert_evidence_ocr),
                        "vision_available": bool(cert_evidence_vision),
                        "ocr_confidence": float(cert_evidence_ocr.get("confidence", 0.0) or 0.0) if cert_evidence_ocr else 0.0,
                        "vision_confidence": float(cert_evidence_vision.get("confidence", 0.0) or 0.0) if cert_evidence_vision else 0.0,
                    }
                    if signals.get("scanned_family_hint") != "certificate_reference_evidence":
                        ev_conf = float(merged_cert_evidence.get("confidence", 0.0) or 0.0)
                        if ev_conf >= 0.45:
                            signals["scanned_family_hint"] = "certificate_reference_evidence"
                            signals["scanned_family_confidence"] = round(max(float(signals.get("scanned_family_confidence", 0.0) or 0.0), ev_conf), 3)
                else:
                    signals["certificate_evidence_source"] = "none"
                    signals["certificate_evidence_lanes"] = {
                        "ocr_available": bool(cert_evidence_ocr),
                        "vision_available": bool(cert_evidence_vision),
                    }
                signals["routing_decision"] = "scanned_supporting_misc"
                signals["routing_reasons"] = ["default_scanned_classification"]
                signals["continuation_context_source"] = "none"

            classifications.append(TablePageClassification(
                page_number=page_num,
                page_class=page_class,
                confidence=round(page_confidence, 3),
                signals=signals,
            ))

        return classifications

    def _get_scanned_page_ocr_snippets(
        self,
        doc: fitz.Document,
        scanned_pages: List[int],
    ) -> Dict[int, str]:
        if not scanned_pages:
            return {}
        if not settings.LITEPARSE_OCR_ENABLED:
            return {}
        if not self.enrichment_service.cli_available:
            return {}

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()
        snippets: Dict[int, str] = {}
        try:
            doc.save(tmp_path)
            pages_arg = ",".join(str(p) for p in sorted(set(scanned_pages)))
            result = self.enrichment_service.run_liteparse_cli(
                file_path=tmp_path,
                pages=pages_arg,
                dpi=settings.LITEPARSE_HIGH_QUALITY_DPI,
                ocr_enabled=True,
            )
            for page_data in result.get("pages", []):
                p = page_data.get("page_number") or page_data.get("page")
                if not isinstance(p, int):
                    continue
                text = str(page_data.get("text") or "").strip()
                if text:
                    snippets[p] = text
        except Exception:
            return {}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return snippets

    def _collect_supporting_phrases(self, text: str) -> List[str]:
        if not text:
            return []
        cues = [
            "довідка",
            "за основним місцем роботи",
            "довідка видана",
            "на пред'явлення",
            "для подання",
            "національний фонд досліджень україни",
            "працює",
        ]
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln and ln.strip()]
        phrases: List[str] = []
        for line in lines:
            low = line.lower()
            if any(cue in low for cue in cues):
                phrases.append(line)
        deduped: List[str] = []
        for p in phrases:
            if p not in deduped:
                deduped.append(p)
        return deduped[:8]

    def _extract_text_with_liteparse_ocr_from_image(self, image_path: str) -> str:
        if not self.enrichment_service.cli_available:
            return ""
        try:
            result = self.enrichment_service.run_liteparse_cli(
                file_path=image_path,
                dpi=settings.LITEPARSE_HIGH_QUALITY_DPI,
                ocr_enabled=True,
            )
        except Exception:
            return ""
        texts: List[str] = []
        if isinstance(result, dict):
            for page_data in result.get("pages", []):
                if not isinstance(page_data, dict):
                    continue
                txt = str(page_data.get("text") or "").strip()
                if txt:
                    texts.append(txt)
            if not texts:
                txt = str(result.get("text") or "").strip()
                if txt:
                    texts.append(txt)
        return "\n".join(texts).strip()

    def _extract_scanned_certificate_reference_evidence_vision(
        self,
        doc: fitz.Document,
        page_number: int,
    ) -> Dict[str, Any]:
        if not settings.USE_SCANNED_CERT_VISION_EVIDENCE:
            return {}
        if not settings.LITEPARSE_OCR_ENABLED:
            return {}
        if not self.enrichment_service.cli_available:
            return {}
        if page_number <= 0 or page_number > len(doc):
            return {}

        page = doc.load_page(page_number - 1)
        rect = page.rect
        if rect.width <= 0 or rect.height <= 0:
            return {}

        scale = float(settings.LITEPARSE_HIGH_QUALITY_DPI) / 72.0
        matrix = fitz.Matrix(scale, scale)
        middle_clip = fitz.Rect(
            rect.x0,
            rect.y0 + (rect.height * 0.2),
            rect.x1,
            rect.y1 - (rect.height * 0.2),
        )

        tmp_full = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_mid = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        full_path = tmp_full.name
        mid_path = tmp_mid.name
        tmp_full.close()
        tmp_mid.close()
        try:
            page.get_pixmap(matrix=matrix, alpha=False).save(full_path)
            page.get_pixmap(matrix=matrix, clip=middle_clip, alpha=False).save(mid_path)
            full_text = self._extract_text_with_liteparse_ocr_from_image(full_path)
            mid_text = self._extract_text_with_liteparse_ocr_from_image(mid_path)
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)
            if os.path.exists(mid_path):
                os.remove(mid_path)

        combined = "\n".join(part for part in [mid_text, full_text] if part).strip()
        if not combined:
            return {}

        evidence = self._extract_scanned_certificate_reference_evidence(combined)
        if not evidence:
            return {}

        evidence["evidence_source"] = "vision_image_regions"
        evidence["supporting_phrases"] = self._collect_supporting_phrases(combined)
        evidence["vision_regions"] = {
            "full_text_len": len(full_text),
            "middle_text_len": len(mid_text),
            "used_regions": ["full_page", "middle_region"],
        }
        return evidence

    def _merge_certificate_evidence(
        self,
        ocr_evidence: Dict[str, Any],
        vision_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        ocr = dict(ocr_evidence or {})
        vis = dict(vision_evidence or {})
        if not ocr and not vis:
            return {}

        def _pick_text(key: str) -> Optional[str]:
            v = (vis.get(key) or "").strip() if isinstance(vis.get(key), str) else vis.get(key)
            o = (ocr.get(key) or "").strip() if isinstance(ocr.get(key), str) else ocr.get(key)
            if v:
                return v
            if o:
                return o
            return None

        merged_phrases: List[str] = []
        for src in (ocr, vis):
            for phrase in src.get("supporting_phrases", []) or []:
                phrase_norm = re.sub(r"\s+", " ", str(phrase)).strip()
                if phrase_norm and phrase_norm not in merged_phrases:
                    merged_phrases.append(phrase_norm)

        ocr_conf = float(ocr.get("confidence", 0.0) or 0.0)
        vis_conf = float(vis.get("confidence", 0.0) or 0.0)
        confidence = max(ocr_conf, vis_conf)
        if ocr and vis:
            confidence = min(1.0, confidence + 0.1)

        has_ocr = bool(ocr)
        has_vis = bool(vis)
        if has_ocr and has_vis:
            source = "ocr_text_patterns+vision_image_regions"
        elif has_vis:
            source = vis.get("evidence_source") or "vision_image_regions"
        else:
            source = ocr.get("evidence_source") or "ocr_text_patterns"

        return {
            "is_certificate_reference": bool(ocr.get("is_certificate_reference") or vis.get("is_certificate_reference")),
            "certificate_keyword_detected": bool(ocr.get("certificate_keyword_detected") or vis.get("certificate_keyword_detected")),
            "primary_employment_confirmed": bool(ocr.get("primary_employment_confirmed") or vis.get("primary_employment_confirmed")),
            "issuance_for_nrfu": bool(ocr.get("issuance_for_nrfu") or vis.get("issuance_for_nrfu")),
            "person_name": _pick_text("person_name"),
            "institution": _pick_text("institution"),
            "position": _pick_text("position"),
            "supporting_phrases": merged_phrases,
            "primary_employment_phrase_detected": bool(ocr.get("primary_employment_phrase_detected") or vis.get("primary_employment_phrase_detected")),
            "evidence_source": source,
            "confidence": round(confidence, 3),
        }

    def _extract_scanned_certificate_reference_evidence(self, text: str) -> Dict[str, Any]:
        src = re.sub(r"\s+", " ", text or "").strip()
        if not src:
            return {}

        low = src.lower()
        certificate_keyword_exact = bool(re.search(r"\bдовідк", low))
        certificate_keyword_fuzzy = bool(re.search(r"(dovid|jlosink|jlosinka)", low))
        certificate_keyword_detected = certificate_keyword_exact or certificate_keyword_fuzzy
        has_main_employment = "за основним місцем роботи" in low
        employment_fuzzy = bool(re.search(r"(прац|npai|prac)", low))
        has_issued_phrase = ("довідка видана" in low) or ("видана" in low and "довідка" in low)
        has_presentation_phrase = ("на пред'явлення" in low) or ("для подання" in low)
        issuance_for_nrfu = (
            ("національний фонд досліджень україни" in low)
            or bool(re.search(r"\bnrfu\b", low, re.IGNORECASE))
        )

        institution = None
        position = None
        person_name = None

        inst_match = re.search(r"(інститут[^,.]{0,80}|університет[^,.]{0,80}|академ[іїi][^,.]{0,80})", src, re.IGNORECASE)
        if inst_match:
            institution = inst_match.group(1).strip()
        elif re.search(r"(ihcth|institut|instytut|hah|nan)", low):
            institution = "institution_detected_from_ocr_fuzzy"

        pos_match = re.search(r"(посада[:\s-]*[^,.]{3,80}|завідувач[^,.]{0,80}|директор[^,.]{0,80}|професор[^,.]{0,80})", src, re.IGNORECASE)
        if pos_match:
            position = pos_match.group(1).strip()

        name_match = re.search(r"([А-ЯІЇЄҐ][а-яіїєґ']+\s+[А-ЯІЇЄҐ][а-яіїєґ']+\s+[А-ЯІЇЄҐ][а-яіїєґ']+)", src)
        if name_match:
            person_name = name_match.group(1).strip()

        score = 0
        if certificate_keyword_exact:
            score += 2
        elif certificate_keyword_fuzzy:
            score += 1
        if has_main_employment:
            score += 2
        elif employment_fuzzy:
            score += 1
        if has_issued_phrase:
            score += 1
        if has_presentation_phrase:
            score += 1
        if issuance_for_nrfu:
            score += 1
        if institution:
            score += 1
        if position:
            score += 1

        if score < 2 or not (certificate_keyword_detected or (has_main_employment and has_issued_phrase)):
            return {}

        primary_employment_confirmed = has_main_employment or (employment_fuzzy and bool(institution))

        return {
            "is_certificate_reference": True,
            "primary_employment_confirmed": primary_employment_confirmed,
            "issuance_for_nrfu": issuance_for_nrfu,
            "certificate_keyword_detected": certificate_keyword_detected,
            "primary_employment_phrase_detected": has_main_employment,
            "person_name": person_name,
            "institution": institution,
            "position": position,
            "has_main_employment_phrase": has_main_employment,
            "has_issued_phrase": has_issued_phrase,
            "has_presentation_phrase": has_presentation_phrase,
            "supporting_phrases": self._collect_supporting_phrases(src),
            "evidence_source": "ocr_text_patterns",
            "confidence": round(min(1.0, score / 7.0), 3),
        }

    def _classify_scanned_page_family(
        self,
        page_index: int,
        pages_text: List[str],
        word_count: int,
        digit_ratio: float,
    ) -> Tuple[str, float, Dict[str, Any]]:
        text = pages_text[page_index] if 0 <= page_index < len(pages_text) else ""
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines() if ln and ln.strip()]
        low = (text or "").lower()
        line_count = len(lines)
        text_density = (word_count / max(line_count, 1)) if line_count else float(word_count)

        cert_cues = [
            "працює",
            "за основним місцем роботи",
            "довідка видана",
            "на пред'явлення",
            "видана для подання",
            "місце роботи",
            "посада",
            "інститут",
            "національний фонд досліджень україни",
        ]
        table_cues = [
            "обсяг фінансування",
            "економічне обґрунтування",
            "найменування",
            "статті витрат",
            "етап виконання проєкту",
            "у випадку залучення",
            "сума",
            "грн",
            "№ з/п",
            "критер",
        ]

        cert_hits = sum(1 for cue in cert_cues if cue in low)
        table_hits = sum(1 for cue in table_cues if cue in low)
        cert_ev = self._extract_scanned_certificate_reference_evidence(text)
        certificate_keyword_detected = bool(cert_ev.get("certificate_keyword_detected")) or bool(re.search(r"\bдовідк", low))
        primary_employment_phrase_detected = bool(cert_ev.get("primary_employment_phrase_detected")) or ("за основним місцем роботи" in low)
        issuance_for_nrfu = bool(cert_ev.get("issuance_for_nrfu")) or (
            ("національний фонд досліджень україни" in low)
            or bool(re.search(r"\bnrfu\b", low, re.IGNORECASE))
        )
        heading_like_hits = sum(
            1
            for ln in lines[:8]
            if re.search(r"(^\d+\)|етап|статтею|обґрунтування|критер)", ln, re.IGNORECASE)
        )

        neighbor_table_continuity = 0
        for n_idx in (page_index - 1, page_index + 1):
            if n_idx < 0 or n_idx >= len(pages_text):
                continue
            n_low = (pages_text[n_idx] or "").lower()
            if not n_low:
                continue
            n_table_hits = sum(1 for cue in table_cues if cue in n_low)
            if n_table_hits >= 1 and (digit_ratio > 0.05 or sum(ch.isdigit() for ch in n_low) / max(len(n_low), 1) > 0.05):
                neighbor_table_continuity += 1

        certificate_score = 0.0
        if certificate_keyword_detected and primary_employment_phrase_detected:
            certificate_score += 0.75
        elif certificate_keyword_detected:
            certificate_score += 0.35
        elif primary_employment_phrase_detected:
            certificate_score += 0.35
        if cert_hits >= 2:
            certificate_score += 0.1
        if "працює" in low and primary_employment_phrase_detected:
            certificate_score += 0.25
        if ("довідка" in low and "видана" in low) or "на пред'явлення" in low:
            certificate_score += 0.2
        if issuance_for_nrfu:
            certificate_score += 0.1
        if cert_ev:
            certificate_score += min(0.2, float(cert_ev.get("confidence", 0.0)) * 0.25)
        if table_hits >= 2:
            certificate_score -= 0.2

        table_score = 0.0
        if table_hits >= 2:
            table_score += 0.35
        if heading_like_hits >= 1:
            table_score += 0.15
        if digit_ratio >= 0.08:
            table_score += 0.15
        if line_count >= 6:
            table_score += 0.1
        if text_density <= 9:
            table_score += 0.1
        if neighbor_table_continuity >= 1:
            table_score += 0.15

        certificate_score = max(0.0, min(1.0, certificate_score))
        table_score = max(0.0, min(1.0, table_score))

        if certificate_score >= 0.55 and certificate_score >= (table_score + 0.1):
            family = "certificate_reference_evidence"
            confidence = certificate_score
        elif table_score >= 0.55:
            family = "multi_page_table_family"
            confidence = table_score
        else:
            family = "scanned_supporting_misc"
            confidence = max(0.5, 1.0 - max(table_score, certificate_score))

        return family, confidence, {
            "line_count": line_count,
            "text_density": round(text_density, 3),
            "digit_ratio": round(digit_ratio, 4),
            "certificate_cue_hits": cert_hits,
            "table_cue_hits": table_hits,
            "certificate_keyword_detected": certificate_keyword_detected,
            "primary_employment_phrase_detected": primary_employment_phrase_detected,
            "issuance_for_nrfu": issuance_for_nrfu,
            "certificate_evidence_confidence": cert_ev.get("confidence") if cert_ev else 0.0,
            "heading_like_hits": heading_like_hits,
            "neighbor_table_continuity": neighbor_table_continuity,
            "certificate_score": round(certificate_score, 3),
            "table_score": round(table_score, 3),
        }

    def _build_line_layout_rows(self, raw_text: str) -> Tuple[List[CanonicalTableColumn], List[CanonicalTableRow], List[str]]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in (raw_text or "").splitlines() if ln and ln.strip()]
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

        columns = [
            CanonicalTableColumn(col_id="c0", name="line_number", semantic_type="int"),
            CanonicalTableColumn(col_id="c1", name="ocr_text", semantic_type="string"),
        ]
        return columns, rows, warnings

    def _build_light_table_from_text_items(
        self,
        page_data: Dict[str, Any],
        min_y: Optional[float] = None,
    ) -> Tuple[bool, List[CanonicalTableColumn], List[CanonicalTableRow], List[int], List[str]]:
        items_raw = page_data.get("textItems") or []
        if not isinstance(items_raw, list) or not items_raw:
            return False, [], [], [], ["no_text_items"]

        token_items = []
        heights: List[float] = []
        for item in items_raw:
            if not isinstance(item, dict):
                continue
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if not text:
                continue
            x = float(item.get("x") or 0.0)
            y = float(item.get("y") or 0.0)
            w = float(item.get("width") or 0.0)
            h = float(item.get("height") or 0.0)
            if min_y is not None and y < (min_y - 2.0):
                continue
            token_items.append({"text": text, "x": x, "y": y, "w": w, "h": h})
            if h > 0:
                heights.append(h)

        if len(token_items) < 6:
            return False, [], [], [], ["insufficient_text_items"]

        token_items.sort(key=lambda t: (t["y"], t["x"]))
        avg_h = (sum(heights) / len(heights)) if heights else 8.0
        row_tol = max(4.0, min(14.0, avg_h * 0.8))
        col_gap = max(16.0, min(60.0, avg_h * 2.2))

        row_groups: List[Dict[str, Any]] = []
        for tok in token_items:
            if not row_groups:
                row_groups.append({"y": tok["y"], "items": [tok]})
                continue
            if abs(tok["y"] - row_groups[-1]["y"]) <= row_tol:
                row_groups[-1]["items"].append(tok)
                row_groups[-1]["y"] = (row_groups[-1]["y"] + tok["y"]) / 2.0
            else:
                row_groups.append({"y": tok["y"], "items": [tok]})

        if len(row_groups) < 3:
            return False, [], [], [], ["insufficient_row_groups"]

        row_cells_text: List[List[str]] = []
        for group in row_groups:
            toks = sorted(group["items"], key=lambda t: t["x"])
            if not toks:
                continue
            current_chunks: List[str] = [toks[0]["text"]]
            prev_right = toks[0]["x"] + toks[0]["w"]
            row_cells: List[str] = []
            for tok in toks[1:]:
                gap = tok["x"] - prev_right
                if gap > col_gap:
                    row_cells.append(re.sub(r"\s+", " ", " ".join(current_chunks)).strip())
                    current_chunks = [tok["text"]]
                else:
                    current_chunks.append(tok["text"])
                prev_right = max(prev_right, tok["x"] + tok["w"])
            if current_chunks:
                row_cells.append(re.sub(r"\s+", " ", " ".join(current_chunks)).strip())
            row_cells = [c for c in row_cells if c]
            if row_cells:
                row_cells_text.append(row_cells)

        if not row_cells_text:
            return False, [], [], [], ["no_row_cells"]

        max_cols = max(len(r) for r in row_cells_text)
        multi_col_rows = sum(1 for r in row_cells_text if len(r) >= 2)
        if max_cols < 2 or multi_col_rows < max(2, int(len(row_cells_text) * 0.25)):
            return False, [], [], [], ["weak_cell_grouping"]
        row_cells_text, repair_warnings = self._repair_questionnaire_row_number_drift(row_cells_text, max_cols)

        columns = [
            CanonicalTableColumn(col_id=f"c{i}", name=f"col_{i + 1}", semantic_type="string")
            for i in range(max_cols)
        ]

        def _normalize_cell(text: str) -> Any:
            clean = text.strip()
            if re.fullmatch(r"[\d\s,\.]+", clean):
                digits = re.sub(r"[^\d]", "", clean)
                if digits:
                    try:
                        return int(digits)
                    except ValueError:
                        return clean
            return clean

        rows: List[CanonicalTableRow] = []
        header_rows: List[int] = []
        for idx, cells in enumerate(row_cells_text):
            row_cells: List[CanonicalTableCell] = []
            digit_cells = 0
            for c_idx in range(max_cols):
                text_val = cells[c_idx] if c_idx < len(cells) else ""
                if re.search(r"\d", text_val):
                    digit_cells += 1
                row_cells.append(
                    CanonicalTableCell(
                        col_id=f"c{c_idx}",
                        text=text_val,
                        normalized=_normalize_cell(text_val) if text_val else "",
                    )
                )
            if idx < 2 and digit_cells == 0:
                header_rows.append(idx)
            rows.append(CanonicalTableRow(row_id=f"r{idx}", cells=row_cells))

        return True, columns, rows, header_rows, repair_warnings

    def _repair_questionnaire_row_number_drift(
        self,
        row_cells_text: List[List[str]],
        max_cols: int,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Conservative repair for questionnaire-like tables where first-column row numbers
        drift to the end of the previous row's last cell (e.g., "... текст 10").
        """
        if max_cols < 4 or len(row_cells_text) < 4:
            return row_cells_text, []

        number_re = re.compile(r"^\d{1,2}$")
        leading_number_count = sum(
            1 for row in row_cells_text if row and number_re.fullmatch((row[0] or "").strip())
        )
        if leading_number_count < 3:
            return row_cells_text, []

        repaired = [list(r) for r in row_cells_text]
        warnings: List[str] = []
        last_seen_num: Optional[int] = None

        for idx, row in enumerate(repaired):
            if row and number_re.fullmatch((row[0] or "").strip()):
                try:
                    last_seen_num = int(row[0].strip())
                except ValueError:
                    last_seen_num = None
                continue
            if idx == 0 or last_seen_num is None:
                continue

            prev = repaired[idx - 1]
            if not prev:
                continue
            tail = (prev[-1] or "").strip()
            m = re.search(r"(?:^|\s)(\d{1,2})\s*$", tail)
            if not m:
                continue

            candidate = m.group(1)
            try:
                candidate_num = int(candidate)
            except ValueError:
                continue
            if candidate_num != (last_seen_num + 1):
                continue

            prev_tail_clean = tail[: m.start(1)].strip()
            prev[-1] = prev_tail_clean
            repaired[idx] = [candidate] + row
            if len(repaired[idx]) > max_cols:
                overflow = repaired[idx][max_cols - 1 :]
                repaired[idx] = repaired[idx][: max_cols - 1] + [" ".join([c for c in overflow if c]).strip()]
            warnings.append("questionnaire_row_number_repaired")
            last_seen_num = candidate_num

        if not warnings:
            return row_cells_text, []
        return repaired, sorted(set(warnings))

    def _apply_scanned_semantic_column_mapping(
        self,
        columns: List[CanonicalTableColumn],
        rows: List[CanonicalTableRow],
        header_rows: List[int],
    ) -> Tuple[List[CanonicalTableColumn], Dict[str, Any], List[str]]:
        warnings: List[str] = []
        mapping_meta: Dict[str, Any] = {
            "applied": False,
            "confidence": 0.0,
            "header_row_index": None,
            "mapped_columns": {},
        }

        if not columns or not rows or not header_rows:
            warnings.append("semantic_mapping_weak_no_header")
            return columns, mapping_meta, warnings

        header_idx = next((idx for idx in header_rows if 0 <= idx < len(rows)), None)
        if header_idx is None:
            warnings.append("semantic_mapping_weak_no_valid_header")
            return columns, mapping_meta, warnings

        header_cells = rows[header_idx].cells
        if not header_cells:
            warnings.append("semantic_mapping_weak_empty_header")
            return columns, mapping_meta, warnings

        def _semantic_name(text: str, fallback_idx: int) -> str:
            low = (text or "").lower()
            low = re.sub(r"\s+", " ", low).strip()
            if not low:
                return f"col_{fallback_idx + 1}"
            if re.search(r"\bдата\b|\bdate\b", low):
                return "date"
            if re.search(r"номер|№|no\\.?|id|код", low):
                return "identifier"
            if re.search(r"назва|заява|довідка|title|document", low):
                return "document_title"
            if re.search(r"установа|інститут|організац|institution|organization", low):
                return "institution"
            if re.search(r"email|e-mail|телефон|phone|контакт", low):
                return "contact"
            if re.search(r"адрес", low):
                return "address"
            if re.search(r"статус|відповідн|status", low):
                return "status"
            if re.search(r"приміт|коментар|note", low):
                return "note"
            return f"col_{fallback_idx + 1}"

        mapped: Dict[str, str] = {}
        mapped_count = 0
        non_empty_count = 0
        for idx, col in enumerate(columns):
            cell_text = header_cells[idx].text if idx < len(header_cells) else ""
            if cell_text.strip():
                non_empty_count += 1
            semantic = _semantic_name(cell_text, idx)
            mapped[col.col_id] = semantic
            if semantic != f"col_{idx + 1}":
                mapped_count += 1

        confidence = (mapped_count / max(non_empty_count, 1)) if non_empty_count else 0.0
        mapping_meta["confidence"] = round(confidence, 3)
        mapping_meta["header_row_index"] = header_idx
        mapping_meta["mapped_columns"] = mapped

        if mapped_count >= 2 and confidence >= 0.5:
            renamed_columns = [
                CanonicalTableColumn(col_id=col.col_id, name=mapped.get(col.col_id, col.name), semantic_type=col.semantic_type)
                for col in columns
            ]
            mapping_meta["applied"] = True
            return renamed_columns, mapping_meta, warnings

        warnings.append("semantic_mapping_weak")
        return columns, mapping_meta, warnings

    def _extract_scanned_financial_context(
        self,
        text: str,
        prev_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Optional[str]], int, Dict[str, bool]]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
        lines = [ln for ln in lines if ln]
        context: Dict[str, Optional[str]] = {
            "subsubsection_title": None,
            "subsection_anchor": None,
            "stage_label": None,
            "case_label": None,
        }
        detected = {
            "subsubsection_title": False,
            "subsection_anchor": False,
            "stage_label": False,
            "case_label": False,
        }
        context_start_idx = 0
        if not lines:
            return context, context_start_idx, detected

        numbered_sub_pat = re.compile(r"^\s*((?:\d+\.){2,}\d+)\s+(.+)$", re.IGNORECASE)
        unnumbered_sub_pat = re.compile(r"(економічне\s+обґрунтування\s+витрат\s+за\s+статтею)", re.IGNORECASE)
        stage_pat = re.compile(r"^\s*\d+\)\s*.+етап.+проєкту", re.IGNORECASE)
        case_pat = re.compile(r"у\s+випадку", re.IGNORECASE)

        for idx, line in enumerate(lines[:20]):
            numbered_match = numbered_sub_pat.match(line)
            if context["subsubsection_title"] is None and numbered_match:
                context["subsection_anchor"] = numbered_match.group(1).strip()
                context["subsubsection_title"] = line
                detected["subsection_anchor"] = True
                detected["subsubsection_title"] = True
                context_start_idx = idx
                continue
            if context["subsubsection_title"] is None and unnumbered_sub_pat.search(line):
                context["subsubsection_title"] = line
                detected["subsubsection_title"] = True
                context_start_idx = idx
                continue
            if context["stage_label"] is None and stage_pat.search(line):
                context["stage_label"] = line
                detected["stage_label"] = True
                if context["subsubsection_title"] is None:
                    context_start_idx = idx
                continue
            if context["case_label"] is None and case_pat.search(line):
                context["case_label"] = line
                detected["case_label"] = True
                if context["subsubsection_title"] is None and context["stage_label"] is None:
                    context_start_idx = idx

        if prev_context:
            for key in ("subsubsection_title", "subsection_anchor", "stage_label", "case_label"):
                if not context.get(key) and prev_context.get(key):
                    context[key] = prev_context.get(key)

        return context, context_start_idx, detected

    def _context_start_y_from_text_items(
        self,
        page_data: Dict[str, Any],
        context: Dict[str, Optional[str]],
    ) -> Optional[float]:
        items = page_data.get("textItems") or []
        if not isinstance(items, list) or not items:
            return None

        anchors: List[str] = []
        for key in ("subsubsection_title", "stage_label", "case_label"):
            val = (context.get(key) or "").strip()
            if val:
                anchors.append(val)

        if not anchors:
            return None

        def _norm(t: str) -> str:
            return re.sub(r"\s+", " ", t or "").strip().lower()

        item_hits: List[float] = []
        norm_anchors = [_norm(a) for a in anchors]
        for item in items:
            if not isinstance(item, dict):
                continue
            t = _norm(str(item.get("text") or ""))
            if not t:
                continue
            for anc in norm_anchors:
                probe = anc[:32]
                if probe and (probe in t or t in anc):
                    y = float(item.get("y") or 0.0)
                    item_hits.append(y)
                    break
        return min(item_hits) if item_hits else None

    def _trim_text_to_financial_context(
        self,
        raw_text: str,
        start_idx: int,
    ) -> str:
        lines = [ln for ln in (raw_text or "").splitlines()]
        if not lines:
            return raw_text or ""
        if 0 <= start_idx < len(lines):
            return "\n".join(lines[start_idx:])
        return raw_text or ""

    def _is_financial_detail_candidate_page(
        self,
        page_number: int,
        pages_text: List[str],
    ) -> bool:
        patterns = [
            r"3\.6\.\d+",
            r"економічне\s+обґрунтування\s+витрат",
            r"етап\s+виконання\s+проєкту",
            r"у\s+випадку\s+залучення",
            r"оплата\s+праці",
        ]
        idx = page_number - 1
        windows = [idx - 1, idx, idx + 1]
        for w in windows:
            if w < 0 or w >= len(pages_text):
                continue
            text = pages_text[w] or ""
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return True
        return False

    def _prepare_liteparse_pdf_with_orientation_normalization(
        self,
        file_content: bytes,
        target_pages: List[int],
    ) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        doc = fitz.open(stream=file_content, filetype="pdf")
        orientation_meta: Dict[int, Dict[str, Any]] = {}
        for p in target_pages:
            if p <= 0 or p > doc.page_count:
                continue
            page = doc.load_page(p - 1)
            rect = page.rect
            is_landscape = rect.width > rect.height
            original_rotation = int(page.rotation or 0)
            applied_rotation = 0
            if is_landscape:
                new_rotation = (original_rotation + 90) % 360
                page.set_rotation(new_rotation)
                applied_rotation = 90
            orientation_meta[p] = {
                "is_landscape": is_landscape,
                "original_rotation": original_rotation,
                "applied_rotation": applied_rotation,
                "width": round(rect.width, 2),
                "height": round(rect.height, 2),
            }

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = tmp.name
        tmp.close()
        doc.save(tmp_path)
        doc.close()
        return tmp_path, orientation_meta

    def _suppress_routed_financial_detail_paragraphs(
        self,
        structure: ParsedDocumentStructure,
        page_numbers: Set[int],
    ) -> None:
        if not page_numbers:
            return

        def _clean_section(sec: DocumentSection):
            sec.blocks = [
                b for b in sec.blocks
                if not (b.block_type == "paragraph" and b.page_number in page_numbers)
            ]
            for sub in sec.subsections:
                _clean_section(sub)

        for zone in structure.zones:
            if zone.zone_type != "description":
                continue
            zone.blocks = [
                b for b in zone.blocks
                if not (b.block_type == "paragraph" and b.page_number in page_numbers)
            ]
            for sec in zone.sections:
                _clean_section(sec)

    def _is_native_financial_detail_candidate_page(
        self,
        page_number: int,
        pages_text: List[str],
    ) -> bool:
        patterns = [
            r"обсяг\s+фінансування\s+за\s+окремими\s+статтями\s+витрат",
            r"економічне\s+обґрунтування\s+витрат\s+за\s+статтею",
            r"оплата\s+праці",
            r"матеріал(и|ів|и)?",
            r"обладнан(ня|ням)",
            r"непрям(і|их)\s+витрат",
            r"інші\s+витрат(и|)\b",
            r"етап\s+виконання\s+проєкту",
            r"у\s+випадку\s+залучення",
        ]
        idx = page_number - 1
        windows = [idx - 1, idx, idx + 1]
        for w in windows:
            if w < 0 or w >= len(pages_text):
                continue
            text = pages_text[w] or ""
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return True
        return False

    def _extract_native_financial_detail_context(
        self,
        text: str,
        prev_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Optional[str]], int, Dict[str, bool]]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
        lines = [ln for ln in lines if ln]
        context: Dict[str, Optional[str]] = {
            "table_title": None,
            "subsubsection_title": None,
            "subsection_anchor": None,
            "stage_label": None,
            "case_label": None,
        }
        detected = {
            "table_title": False,
            "subsubsection_title": False,
            "subsection_anchor": False,
            "stage_label": False,
            "case_label": False,
        }
        context_start_idx = 0
        if not lines:
            return context, context_start_idx, detected

        numbered_sub_pat = re.compile(r"^\s*((?:\d+\.){2,}\d+)\s+(.+)$", re.IGNORECASE)
        title_pat = re.compile(
            r"(обсяг\s+фінансування\s+за\s+окремими\s+статтями\s+витрат|економічне\s+обґрунтування\s+витрат\s+за\s+статтею)",
            re.IGNORECASE,
        )
        stage_pat = re.compile(r"^\s*\d+\)\s*.+етап.+проєкту", re.IGNORECASE)
        case_pat = re.compile(r"у\s+випадку", re.IGNORECASE)

        for idx, line in enumerate(lines[:28]):
            numbered_match = numbered_sub_pat.match(line)
            if context["subsubsection_title"] is None and numbered_match:
                title_tail = numbered_match.group(2).strip()
                if title_pat.search(title_tail):
                    context["subsection_anchor"] = numbered_match.group(1).strip()
                    context["subsubsection_title"] = line
                    context["table_title"] = title_tail
                    detected["subsection_anchor"] = True
                    detected["subsubsection_title"] = True
                    detected["table_title"] = True
                    context_start_idx = idx
                    continue

            if context["table_title"] is None and title_pat.search(line):
                context["table_title"] = line
                context["subsubsection_title"] = line
                detected["table_title"] = True
                detected["subsubsection_title"] = True
                context_start_idx = idx
                continue

            if context["stage_label"] is None and stage_pat.search(line):
                context["stage_label"] = line
                detected["stage_label"] = True
                if context["table_title"] is None:
                    context_start_idx = idx
                continue

            if context["case_label"] is None and case_pat.search(line):
                context["case_label"] = line
                detected["case_label"] = True
                if context["table_title"] is None and context["stage_label"] is None:
                    context_start_idx = idx

        if prev_context:
            for key in ("table_title", "subsubsection_title", "subsection_anchor", "stage_label", "case_label"):
                if not context.get(key) and prev_context.get(key):
                    context[key] = prev_context.get(key)

        return context, context_start_idx, detected

    def _build_light_table_from_native_lines(
        self,
        raw_text: str,
    ) -> Tuple[bool, List[CanonicalTableColumn], List[CanonicalTableRow], List[int], List[str]]:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in (raw_text or "").splitlines() if ln and ln.strip()]
        if len(lines) < 4:
            return False, [], [], [], ["insufficient_lines"]

        parsed_rows: List[List[str]] = []
        multi_col_rows = 0
        for line in lines:
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t") if c and c.strip()]
            else:
                cells = [c.strip() for c in re.split(r"\s{2,}", line) if c and c.strip()]
                if len(cells) <= 1:
                    cells = [c.strip() for c in re.split(r"\s+\|\s+", line) if c and c.strip()]
            if not cells:
                continue
            parsed_rows.append(cells)
            if len(cells) >= 2:
                multi_col_rows += 1

        if len(parsed_rows) < 3:
            return False, [], [], [], ["insufficient_parsed_rows"]

        max_cols = max(len(r) for r in parsed_rows)
        if max_cols < 2 or multi_col_rows < 2:
            return False, [], [], [], ["weak_native_cell_grouping"]
        parsed_rows, repair_warnings = self._repair_questionnaire_row_number_drift(parsed_rows, max_cols)

        columns = [
            CanonicalTableColumn(col_id=f"c{i}", name=f"col_{i + 1}", semantic_type="string")
            for i in range(max_cols)
        ]

        def _normalize_cell(text: str) -> Any:
            clean = text.strip()
            if re.fullmatch(r"[\d\s,\.]+", clean):
                digits = re.sub(r"[^\d]", "", clean)
                if digits:
                    try:
                        return int(digits)
                    except ValueError:
                        return clean
            return clean

        rows: List[CanonicalTableRow] = []
        header_rows: List[int] = []
        for idx, cells in enumerate(parsed_rows):
            digit_cells = 0
            row_cells: List[CanonicalTableCell] = []
            for c_idx in range(max_cols):
                text_val = cells[c_idx] if c_idx < len(cells) else ""
                if re.search(r"\d", text_val):
                    digit_cells += 1
                row_cells.append(
                    CanonicalTableCell(
                        col_id=f"c{c_idx}",
                        text=text_val,
                        normalized=_normalize_cell(text_val) if text_val else "",
                    )
                )
            if idx < 2 and digit_cells == 0:
                header_rows.append(idx)
            rows.append(CanonicalTableRow(row_id=f"r{idx}", cells=row_cells))

        return True, columns, rows, header_rows, repair_warnings

    def _augment_native_complex_description_tables(
        self,
        structure: ParsedDocumentStructure,
        pages_text: List[str],
    ) -> None:
        native_specs: List[int] = []
        for cls in structure.page_table_classifications:
            if cls.page_class != "native_text_complex_table":
                continue
            zone = next(
                (z for z in structure.zones if z.page_start <= cls.page_number <= (z.page_end or z.page_start)),
                None,
            )
            if not zone or zone.zone_type != "description":
                continue
            if self._is_native_financial_detail_candidate_page(cls.page_number, pages_text):
                native_specs.append(cls.page_number)

        prev_ctx: Dict[str, Optional[str]] = {}
        prev_page: Optional[int] = None
        continuation_group: Optional[str] = None
        routed_pages: Set[int] = set()
        table_family = "native_financial_detail_complex_table"
        emitted_native_by_page: Dict[int, CanonicalTable] = {}

        for page_num in sorted(set(native_specs)):
            if page_num <= 0 or page_num > len(pages_text):
                continue
            raw_text = pages_text[page_num - 1] or ""
            if not raw_text.strip():
                continue

            prev_ctx_snapshot = prev_ctx.copy()
            context, context_start_idx, detected_ctx = self._extract_native_financial_detail_context(raw_text, prev_ctx)
            trimmed_text = self._trim_text_to_financial_context(raw_text, context_start_idx)
            grouped, columns, rows, header_rows, grouping_warnings = self._build_light_table_from_native_lines(trimmed_text)
            fallback_warnings: List[str] = []
            semantic_warnings: List[str] = []
            semantic_meta: Dict[str, Any] = {
                "applied": False,
                "confidence": 0.0,
                "header_row_index": None,
                "mapped_columns": {},
            }
            if not grouped:
                columns, rows, fallback_warnings = self._build_line_layout_rows(trimmed_text)
            else:
                columns, semantic_meta, semantic_warnings = self._apply_scanned_semantic_column_mapping(columns, rows, header_rows)
            warnings = grouping_warnings + semantic_warnings + fallback_warnings
            extraction_mode = "native_financial_detail_light_table" if grouped else "native_financial_detail_line_fallback"

            cls = next((c for c in structure.page_table_classifications if c.page_number == page_num), None)
            page_class = cls.page_class if cls else "native_text_complex_table"
            page_confidence = cls.confidence if cls else 0.8

            has_fresh_context = any(detected_ctx.get(k, False) for k in ("table_title", "subsubsection_title", "stage_label", "case_label"))
            continuation = False
            if prev_page is not None and page_num == (prev_page + 1):
                same_context = (
                    context.get("table_title") == prev_ctx_snapshot.get("table_title")
                    and context.get("stage_label") == prev_ctx_snapshot.get("stage_label")
                    and context.get("case_label") == prev_ctx_snapshot.get("case_label")
                )
                if same_context or not has_fresh_context:
                    continuation = True

            if continuation and continuation_group:
                group_id = continuation_group
            else:
                group_id = f"native_fin_detail_grp_p{page_num:03d}"
                continuation_group = group_id

            table_title = (
                context.get("table_title")
                or context.get("subsubsection_title")
                or context.get("stage_label")
                or "Native financial detail table"
            )

            routed_pages.add(page_num)
            prev_page = page_num
            prev_ctx = context.copy()

            native_table = CanonicalTable(
                table_id=f"tbl_{table_family}_p{page_num:03d}_01",
                table_family=table_family,
                zone_type="description",
                title=table_title,
                page_start=page_num,
                page_end=page_num,
                source={
                    "parser": "pymupdf_native",
                    "page_class": page_class,
                    "confidence": page_confidence,
                    "extraction_mode": extraction_mode,
                    "semantic_mapping_confidence": semantic_meta.get("confidence", 0.0),
                    "warnings": warnings,
                },
                context={
                    "table_title": context.get("table_title"),
                    "subsubsection_title": context.get("subsubsection_title"),
                    "subsection_anchor": context.get("subsection_anchor"),
                    "stage_label": context.get("stage_label"),
                    "case_label": context.get("case_label"),
                    "table_group_id": group_id,
                    "continuation": continuation,
                },
                columns=columns,
                rows=rows,
                spans=[],
                validation={
                    "row_count": len(rows),
                    "column_count": len(columns),
                    "header_row_indices": header_rows,
                    "semantic_mapping": semantic_meta,
                    "normalization_warnings": warnings,
                },
            )

            structure.tables = [
                t for t in structure.tables
                if not (
                    t.table_family == table_family
                    and t.page_start == page_num
                )
            ]
            structure.tables.append(native_table)
            emitted_native_by_page[page_num] = native_table

        scanned_cont_pages: Set[int] = set()
        if True:
            class_by_page: Dict[int, TablePageClassification] = {
                c.page_number: c for c in structure.page_table_classifications
            }
            def _is_generic_native_table_candidate(page_num: int) -> bool:
                if page_num <= 0 or page_num > len(pages_text):
                    return False
                text = (pages_text[page_num - 1] or "")
                low = text.lower()
                if re.search(r"(анкета\s+відповідності|критер|наукового\s+керівника)", low):
                    return True
                lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln and ln.strip()]
                numeric_lead = sum(1 for ln in lines if re.match(r"^\d{1,2}\b", ln))
                return numeric_lead >= 5

            def _extract_family_label(page_num: int) -> Optional[str]:
                if page_num <= 0 or page_num > len(pages_text):
                    return None
                lines = [re.sub(r"\s+", " ", ln).strip() for ln in (pages_text[page_num - 1] or "").splitlines() if ln and ln.strip()]
                for ln in lines[:12]:
                    if len(ln) < 8:
                        continue
                    if re.search(r"(анкета|критер|відповідності)", ln.lower()):
                        return ln
                return lines[0] if lines else None

            def _has_tabular_density(page_num: int) -> bool:
                if page_num <= 0 or page_num > len(pages_text):
                    return False
                lines = [re.sub(r"\s+", " ", ln).strip() for ln in (pages_text[page_num - 1] or "").splitlines() if ln and ln.strip()]
                if len(lines) < 8:
                    return False
                numeric_lead = sum(1 for ln in lines if re.match(r"^\d{1,2}\b", ln))
                digit_heavy = sum(1 for ln in lines if sum(ch.isdigit() for ch in ln) >= 2)
                return (numeric_lead >= 2) or (digit_heavy >= max(3, int(len(lines) * 0.25)))

            native_terminal_groups: List[Dict[str, Any]] = []
            native_desc_pages = sorted(
                c.page_number for c in structure.page_table_classifications
                if c.page_class in {"native_text", "native_text_complex_table"}
                and any(
                    z.zone_type == "description" and z.page_start <= c.page_number <= (z.page_end or z.page_start)
                    for z in structure.zones
                )
            )
            if native_desc_pages:
                run: List[int] = []
                for p in native_desc_pages:
                    is_seed_candidate = _is_generic_native_table_candidate(p)
                    is_continuation_candidate = _has_tabular_density(p)
                    if run and p == run[-1] + 1 and (is_seed_candidate or is_continuation_candidate):
                        run.append(p)
                        continue
                    if is_seed_candidate:
                        if not run:
                            run = [p]
                        else:
                            if len(run) >= 3:
                                native_terminal_groups.append({
                                    "start": run[0],
                                    "end": run[-1],
                                    "label": _extract_family_label(run[0]),
                                })
                            run = [p]
                    else:
                        if len(run) >= 3:
                            native_terminal_groups.append({
                                "start": run[0],
                                "end": run[-1],
                                "label": _extract_family_label(run[0]),
                            })
                        run = []
                if len(run) >= 3:
                    native_terminal_groups.append({
                        "start": run[0],
                        "end": run[-1],
                        "label": _extract_family_label(run[0]),
                    })

            def _set_page_decision(
                page_num: int,
                decision: str,
                reasons: List[str],
                continuation_source: Optional[str] = None,
                certificate_source: Optional[str] = None,
            ) -> None:
                cls = class_by_page.get(page_num)
                if not cls:
                    return
                sig = dict(cls.signals or {})
                sig["routing_decision"] = decision
                sig["routing_reasons"] = reasons
                sig["continuation_context_source"] = continuation_source or "none"
                sig["certificate_evidence_source"] = certificate_source or sig.get("certificate_evidence_source", "none")
                cls.signals = sig

            desc_pages = sorted({
                c.page_number for c in structure.page_table_classifications
                if c.page_class == "scanned_image_only"
                and any(
                    z.zone_type == "description" and z.page_start <= c.page_number <= (z.page_end or z.page_start)
                    for z in structure.zones
                )
            })
            for p in desc_pages:
                _set_page_decision(p, "scanned_supporting_misc", ["default_scanned_description_page"])
            scanned_run_len_by_page: Dict[int, int] = {}
            if desc_pages:
                run: List[int] = [desc_pages[0]]
                for p in desc_pages[1:]:
                    if p == run[-1] + 1:
                        run.append(p)
                    else:
                        for rp in run:
                            scanned_run_len_by_page[rp] = len(run)
                        run = [p]
                for rp in run:
                    scanned_run_len_by_page[rp] = len(run)
            native_pages_sorted = sorted(emitted_native_by_page.keys())

            def _find_anchor_table(page_num: int) -> Optional[CanonicalTable]:
                next_candidates = [p for p in native_pages_sorted if p > page_num]
                prev_candidates = [p for p in native_pages_sorted if p < page_num]
                if next_candidates:
                    return emitted_native_by_page[next_candidates[0]]
                if prev_candidates:
                    return emitted_native_by_page[prev_candidates[-1]]
                return None

            for page_num in desc_pages:
                if page_num in emitted_native_by_page:
                    continue
                cls = class_by_page.get(page_num)
                cls_signals = cls.signals or {} if cls else {}
                family_hint = cls_signals.get("scanned_family_hint")
                terminal_group = next((g for g in native_terminal_groups if page_num == (int(g["end"]) + 1)), None)
                cert_ev = cls_signals.get("certificate_reference_evidence") or {}
                cert_conf = float(cert_ev.get("confidence") or 0.0)
                strong_certificate_signal = bool(
                    cert_ev.get("is_certificate_reference")
                    and cert_conf >= 0.35
                    and (
                        cert_ev.get("certificate_keyword_detected")
                        or cert_ev.get("primary_employment_phrase_detected")
                    )
                )
                if family_hint == "certificate_reference_evidence" or strong_certificate_signal:
                    _set_page_decision(
                        page_num,
                        "certificate_reference_evidence",
                        (
                            ["family_hint_certificate_reference_evidence"]
                            if family_hint == "certificate_reference_evidence"
                            else ["certificate_reference_evidence_signals"]
                        ),
                        certificate_source=(
                            cls_signals.get("certificate_evidence_source")
                            or cert_ev.get("evidence_source")
                            or cls_signals.get("scanned_family_text_source")
                            or "none"
                        ),
                    )
                    continue
                # Keep continuation placeholders conservative: require a scanned run,
                # unless this is a terminal scanned continuation page after a native multi-page table group.
                if scanned_run_len_by_page.get(page_num, 0) < 2 and not terminal_group:
                    _set_page_decision(page_num, "scanned_supporting_misc", ["scanned_run_too_short"])
                    continue
                prev_native = any((p < page_num and page_num - p <= 4) for p in native_pages_sorted)
                next_native = any((p > page_num and p - page_num <= 4) for p in native_pages_sorted)
                terminal_continuation = False
                if terminal_group:
                    prev_cls = class_by_page.get(int(terminal_group["end"]))
                    prev_sig = prev_cls.signals or {} if prev_cls else {}
                    curr_orient = cls_signals.get("is_landscape")
                    prev_orient = prev_sig.get("is_landscape")
                    same_orient_terminal = (
                        curr_orient is None
                        or prev_orient is None
                        or curr_orient == prev_orient
                    )
                    shape_ok_terminal = bool(cls_signals.get("image_area_ratio", 0.0) >= 0.35)
                    cont_sig = cls_signals.get("scanned_family_signals") or {}
                    raw_text = pages_text[page_num - 1] if 0 < page_num <= len(pages_text) else ""
                    raw_low = raw_text.lower()
                    continuation_like_text = bool(
                        (cont_sig.get("line_count", 0) >= 10)
                        or (float(cls_signals.get("digit_ratio", 0.0) or 0.0) >= 0.03)
                    )
                    signature_footer_like = bool(
                        re.search(r"\b(підпис|дата|прізвище|ініціали)\b", raw_low)
                        or "м.п." in raw_low
                    )
                    terminal_continuation = bool(
                        shape_ok_terminal
                        and same_orient_terminal
                        and (
                            continuation_like_text
                            or signature_footer_like
                        )
                    )

                if not (prev_native or next_native or terminal_continuation):
                    _set_page_decision(page_num, "scanned_supporting_misc", ["no_nearby_native_anchor_or_terminal_group"])
                    continue

                is_family_table = family_hint == "multi_page_table_family"

                structural_continuity = False
                for neighbor in (page_num - 1, page_num + 1):
                    n_cls = class_by_page.get(neighbor)
                    if not n_cls or n_cls.page_class != "scanned_image_only":
                        continue
                    n_signals = n_cls.signals or {}
                    same_orient = cls_signals.get("is_landscape") == n_signals.get("is_landscape")
                    img_curr = float(cls_signals.get("image_area_ratio", 0.0) or 0.0)
                    img_nei = float(n_signals.get("image_area_ratio", 0.0) or 0.0)
                    close_image_coverage = abs(img_curr - img_nei) <= 0.2
                    if same_orient and close_image_coverage:
                        structural_continuity = True
                        break

                if not (is_family_table or structural_continuity):
                    if not terminal_continuation:
                        _set_page_decision(page_num, "scanned_supporting_misc", ["weak_table_family_and_structure_continuity"])
                        continue

                anchor_table = _find_anchor_table(page_num)
                anchor_ctx: Dict[str, Any] = {}
                anchor_title = None
                group_id = None
                continuation_context_source = "native_financial_detail_anchor_context"

                if terminal_continuation and terminal_group:
                    group_id = f"native_multi_tbl_grp_p{int(terminal_group['start']):03d}_{int(terminal_group['end']):03d}"
                    anchor_title = terminal_group.get("label") or "Native multi-page table continuation"
                    anchor_ctx = {
                        "table_title": terminal_group.get("label"),
                        "subsubsection_title": terminal_group.get("label"),
                        "subsection_anchor": None,
                        "stage_label": None,
                        "case_label": None,
                        "table_group_id": group_id,
                    }
                    continuation_context_source = "native_multi_page_group_context"
                elif anchor_table:
                    anchor_ctx = anchor_table.context or {}
                    group_id = anchor_ctx.get("table_group_id") or f"native_fin_detail_grp_p{anchor_table.page_start:03d}"
                    anchor_title = anchor_table.title or "Scanned financial-detail continuation"
                else:
                    _set_page_decision(page_num, "scanned_supporting_misc", ["no_anchor_table_context"])
                    continue

                raw_text = pages_text[page_num - 1] if 0 < page_num <= len(pages_text) else ""
                columns, rows, fallback_warnings = self._build_line_layout_rows(raw_text)
                warnings = ["scanned_continuation_no_ocr"] + fallback_warnings
                table_family = (
                    "scanned_native_table_terminal_continuation_placeholder"
                    if continuation_context_source == "native_multi_page_group_context"
                    else "scanned_financial_detail_continuation_placeholder"
                )

                scanned_cont_table = CanonicalTable(
                    table_id=f"tbl_scanned_fin_cont_p{page_num:03d}_01",
                    table_family=table_family,
                    zone_type="description",
                    title=anchor_title or "Scanned table continuation",
                    page_start=page_num,
                    page_end=page_num,
                    source={
                        "parser": "pymupdf_native_bridge",
                        "page_class": "scanned_image_only",
                        "confidence": 0.7,
                        "extraction_mode": "scanned_continuation_placeholder_no_ocr",
                        "warnings": warnings,
                    },
                    context={
                        "table_title": anchor_ctx.get("table_title"),
                        "subsubsection_title": anchor_ctx.get("subsubsection_title"),
                        "subsection_anchor": anchor_ctx.get("subsection_anchor"),
                        "stage_label": anchor_ctx.get("stage_label"),
                        "case_label": anchor_ctx.get("case_label"),
                        "table_group_id": group_id,
                        "continuation": True,
                    },
                    columns=columns,
                    rows=rows,
                    spans=[],
                    validation={
                        "row_count": len(rows),
                        "column_count": len(columns),
                        "header_row_indices": [],
                        "semantic_mapping": {
                            "applied": False,
                            "confidence": 0.0,
                            "header_row_index": None,
                            "mapped_columns": {},
                        },
                        "normalization_warnings": warnings,
                    },
                )

                structure.tables = [
                    t for t in structure.tables
                    if not (t.page_start == page_num and t.zone_type == "description")
                ]
                structure.tables.append(scanned_cont_table)
                scanned_cont_pages.add(page_num)
                decision_reasons = []
                if is_family_table:
                    decision_reasons.append("family_hint_multi_page_table_family")
                if structural_continuity:
                    decision_reasons.append("structural_continuity_neighbor_scanned_pages")
                if terminal_continuation and terminal_group:
                    decision_reasons.append("terminal_scanned_continuation_after_native_multi_page_group")
                _set_page_decision(
                    page_num,
                    "routed_scanned_continuation",
                    decision_reasons or ["continuation_inferred"],
                    continuation_source=continuation_context_source,
                )

                # Ownership trimming: if preceding description paragraph includes table-start marker,
                # keep only the narrative tail before the first marker.
                prev_page = page_num - 1
                if prev_page >= 1:
                    markers = [
                        anchor_ctx.get("table_title") or "",
                        anchor_ctx.get("stage_label") or "",
                        anchor_ctx.get("case_label") or "",
                        "Обсяг фінансування за окремими статтями витрат",
                        "Економічне обґрунтування витрат за статтею",
                        "№ з/п",
                        "Найменування статті витрат",
                    ]
                    markers = [m for m in markers if m]

                    def _trim_page_paragraphs(page_to_trim: int):
                        def _trim_block_text(text: str) -> Optional[str]:
                            src = text or ""
                            cut_idx = None
                            for marker in markers:
                                idx = src.find(marker)
                                if idx >= 0 and (cut_idx is None or idx < cut_idx):
                                    cut_idx = idx
                            if cut_idx is None:
                                return src
                            kept = src[:cut_idx].strip()
                            return kept if kept else None

                        def _trim_in_section(sec: DocumentSection):
                            new_blocks: List[DocumentBlock] = []
                            for b in sec.blocks:
                                if b.block_type != "paragraph" or b.page_number != page_to_trim:
                                    new_blocks.append(b)
                                    continue
                                trimmed = _trim_block_text(b.text)
                                if trimmed is None:
                                    continue
                                b.text = trimmed
                                new_blocks.append(b)
                            sec.blocks = new_blocks
                            for sub in sec.subsections:
                                _trim_in_section(sub)

                        for z in structure.zones:
                            if z.zone_type != "description":
                                continue
                            new_zone_blocks: List[DocumentBlock] = []
                            for b in z.blocks:
                                if b.block_type != "paragraph" or b.page_number != page_to_trim:
                                    new_zone_blocks.append(b)
                                    continue
                                trimmed = _trim_block_text(b.text)
                                if trimmed is None:
                                    continue
                                b.text = trimmed
                                new_zone_blocks.append(b)
                            z.blocks = new_zone_blocks
                            for sec in z.sections:
                                _trim_in_section(sec)

                    _trim_page_paragraphs(prev_page)

        self._suppress_routed_financial_detail_paragraphs(structure, routed_pages | scanned_cont_pages)

    def _augment_scanned_tables_with_liteparse(
        self,
        file_content: bytes,
        structure: ParsedDocumentStructure,
        pages_text: Optional[List[str]] = None,
    ):
        def _zone_for_page(page_number: int) -> Optional[str]:
            zone = next(
                (z for z in structure.zones if z.page_start <= page_number <= (z.page_end or z.page_start)),
                None,
            )
            return zone.zone_type if zone else None

        pages_text = pages_text or []
        routed_specs: List[Dict[str, Any]] = []
        for cls in structure.page_table_classifications:
            if cls.page_class != "scanned_image_only":
                continue
            zone_type = _zone_for_page(cls.page_number)
            if zone_type == "annex":
                routed_specs.append({"page_number": cls.page_number, "route_family": "annex_scanned"})
                continue
            if zone_type == "description" and self._is_financial_detail_candidate_page(cls.page_number, pages_text):
                routed_specs.append({"page_number": cls.page_number, "route_family": "financial_detail_scanned"})

        if not routed_specs:
            return

        if not self.enrichment_service.cli_available:
            if not structure.metadata:
                structure.metadata = {}
            structure.metadata["liteparse_scanned_routing_warning"] = "LiteParse CLI not available; scanned routing skipped."
            return

        page_numbers = sorted({int(spec["page_number"]) for spec in routed_specs})
        pages_arg = ",".join(str(p) for p in page_numbers)
        spec_by_page = {int(spec["page_number"]): spec["route_family"] for spec in routed_specs}
        tmp_path, orientation_meta = self._prepare_liteparse_pdf_with_orientation_normalization(file_content, page_numbers)

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

        prev_ctx: Dict[str, Optional[str]] = {}
        prev_page: Optional[int] = None
        continuation_group: Optional[str] = None
        financial_detail_pages: Set[int] = set()

        for page_data in liteparse_result.get("pages", []):
            page_num = page_data.get("page_number") or page_data.get("page")
            if not isinstance(page_num, int):
                continue

            cls = next((c for c in structure.page_table_classifications if c.page_number == page_num), None)
            if not cls or cls.page_class != "scanned_image_only":
                continue

            route_family = spec_by_page.get(page_num)
            if not route_family:
                continue

            raw_text = page_data.get("text") or ""
            context: Dict[str, Optional[str]] = {}
            context_start_idx = 0
            prev_ctx_snapshot = prev_ctx.copy()
            detected_ctx: Dict[str, bool] = {}
            if route_family == "financial_detail_scanned":
                context, context_start_idx, detected_ctx = self._extract_scanned_financial_context(raw_text, prev_ctx)
                financial_detail_pages.add(page_num)
            trimmed_text = self._trim_text_to_financial_context(raw_text, context_start_idx)
            start_y = self._context_start_y_from_text_items(page_data, context) if context else None

            grouped, columns, rows, header_rows, grouping_warnings = self._build_light_table_from_text_items(
                page_data,
                min_y=start_y,
            )
            fallback_warnings: List[str] = []
            semantic_warnings: List[str] = []
            semantic_meta: Dict[str, Any] = {
                "applied": False,
                "confidence": 0.0,
                "header_row_index": None,
                "mapped_columns": {},
            }
            if not grouped:
                columns, rows, fallback_warnings = self._build_line_layout_rows(trimmed_text)
            else:
                columns, semantic_meta, semantic_warnings = self._apply_scanned_semantic_column_mapping(columns, rows, header_rows)
            warnings = grouping_warnings + semantic_warnings + fallback_warnings
            if route_family == "financial_detail_scanned":
                extraction_mode = (
                    "scanned_financial_detail_route_light_table"
                    if grouped else
                    "scanned_financial_detail_route_line_fallback"
                )
            else:
                extraction_mode = "scanned_page_only_route_light_table" if grouped else "scanned_page_only_route_line_fallback"

            page_zone = next(
                (z for z in structure.zones if z.page_start <= page_num <= (z.page_end or z.page_start)),
                None,
            )
            zone_type = page_zone.zone_type if page_zone else "unknown"

            orientation = orientation_meta.get(page_num, {})
            continuation = False
            if route_family == "financial_detail_scanned":
                has_fresh_context = any(detected_ctx.get(k, False) for k in ("subsubsection_title", "stage_label", "case_label"))
                if prev_page is not None and page_num == (prev_page + 1):
                    same_context = (
                        context.get("subsubsection_title") == prev_ctx_snapshot.get("subsubsection_title")
                        and context.get("stage_label") == prev_ctx_snapshot.get("stage_label")
                        and context.get("case_label") == prev_ctx_snapshot.get("case_label")
                    )
                    if same_context or not has_fresh_context:
                        continuation = True
                if continuation and continuation_group:
                    group_id = continuation_group
                else:
                    group_id = f"fin_detail_grp_p{page_num:03d}"
                    continuation_group = group_id
                prev_page = page_num
                prev_ctx = context.copy()
            else:
                group_id = None

            table_family = "scanned_page_ocr_layout" if route_family == "annex_scanned" else "scanned_financial_detail_ocr_layout"
            table_title = "LiteParse OCR Layout Lines"
            if route_family == "financial_detail_scanned":
                table_title = context.get("subsubsection_title") or context.get("stage_label") or "Scanned financial detail OCR layout"

            scanned_table = CanonicalTable(
                table_id=f"tbl_{table_family}_p{page_num:03d}_01",
                table_family=table_family,
                zone_type=zone_type,
                title=table_title,
                page_start=page_num,
                page_end=page_num,
                source={
                    "parser": "liteparse_cli",
                    "page_class": cls.page_class,
                    "confidence": cls.confidence,
                    "extraction_mode": extraction_mode,
                    "orientation": orientation,
                    "semantic_mapping_confidence": semantic_meta.get("confidence", 0.0),
                    "warnings": warnings,
                },
                context={
                    "subsubsection_title": context.get("subsubsection_title") if context else None,
                    "subsection_anchor": context.get("subsection_anchor") if context else None,
                    "stage_label": context.get("stage_label") if context else None,
                    "case_label": context.get("case_label") if context else None,
                    "table_group_id": group_id,
                    "continuation": continuation,
                },
                columns=columns,
                rows=rows,
                spans=[],
                validation={
                    "row_count": len(rows),
                    "column_count": len(columns),
                    "header_row_indices": header_rows,
                    "semantic_mapping": semantic_meta,
                    "normalization_warnings": warnings,
                },
            )

            structure.tables = [
                t for t in structure.tables
                if not (
                    t.table_family == table_family
                    and t.page_start == page_num
                )
            ]
            structure.tables.append(scanned_table)

        self._suppress_routed_financial_detail_paragraphs(structure, financial_detail_pages)

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

        # Layer 2: Add canonical tables for native_text_complex_table financial-detail pages in Zone 2.
        self._augment_native_complex_description_tables(structure, pages_text)
            
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
        """Extract structured fields from Institution Profile (Zone 4) as flat label->value pairs."""
        if not structure.metadata:
            structure.metadata = {}
        inst_data: Dict[str, Any] = {}

        zone_block_text = "\n".join([block.text for block in zone.blocks])
        zone_page_text = "\n".join(
            pages[p - 1]
            for p in range(zone.page_start, (zone.page_end or zone.page_start) + 1)
            if 0 <= (p - 1) < len(pages)
        ).strip()
        full_zone_text = zone_page_text if zone_page_text else zone_block_text
        lines = [re.sub(r"\s+", " ", l.strip()) for l in full_zone_text.split("\n") if l and l.strip()]

        if not lines:
            structure.metadata["institution_profile"] = inst_data
            return

        labels_to_check = [
            "Назва установи",
            "Учасник конкурсу",
            "Код ЄДРПОУ",
            "Код(и) КВЕД",
            "Юридична адреса",
            "Поштова адреса",
            "Фактична адреса",
            "Телефон",
            "Адреса електронної пошти",
            "Посилання на веб сторінку",
            "Організаційно-правова форма",
            "Підпорядкованість",
            "ПІБ керівника",
            "підприємства/установи/організації",
        ]

        label_specs = [
            ("organization_type", r"організац\w*[-\s]правов\w*\s+форм"),
            ("parent_organization", r"підпорядкован\w+"),
            ("institution_head_name", r"(?:піб|прізвище.*ім[я'’]).*керівник"),
            ("legal_address", r"юридичн\w+\s+адрес"),
            ("postal_address", r"поштов\w+\s+адрес"),
            ("physical_address", r"фактичн\w+\s+адрес"),
            ("edrpou_code", r"код\s+єдрпоу"),
            ("kved_code", r"код(?:\(и\))?\s+квед"),
            ("phone_number", r"\bтелефон\b"),
            ("email", r"адреса\s+електронн\w+\s+пошт"),
            ("website", r"посилання\s+на\s+веб"),
        ]

        compiled_specs = [(field, re.compile(pattern, re.IGNORECASE)) for field, pattern in label_specs]
        continuation_label_re = re.compile(
            r"^(?:/|підприємства/установи/організації|установи/організації|організації|де працює учасник)\b",
            re.IGNORECASE,
        )

        def _build_label_candidate(start_idx: int) -> Tuple[str, int]:
            candidate = lines[start_idx]
            consumed = 1
            if start_idx + 1 < len(lines) and continuation_label_re.search(lines[start_idx + 1]):
                candidate = f"{candidate} {lines[start_idx + 1]}"
                consumed = 2
            return candidate.lower(), consumed

        def _looks_like_label(line: str) -> bool:
            low = line.lower()
            if any(cre.search(low) for _, cre in compiled_specs):
                return True
            return continuation_label_re.search(low) is not None

        def _extract_inline_value(label_idx: int) -> str:
            line = lines[label_idx]
            if ":" in line:
                inline = line.split(":", 1)[1].strip()
                if inline and not _looks_like_label(inline):
                    return inline
            return ""

        label_hits: Dict[str, Tuple[int, int]] = {}
        for i in range(len(lines)):
            label_candidate, consumed = _build_label_candidate(i)
            for field, cre in compiled_specs:
                if field in label_hits:
                    continue
                if cre.search(label_candidate):
                    label_hits[field] = (i, consumed)

        # Institution name is usually the first non-label line on this flat page.
        for ln in lines[:6]:
            if (
                len(ln) >= 10
                and not _looks_like_label(ln)
                and not re.search(r"\bучасник\b", ln, re.IGNORECASE)
            ):
                inst_data["institution_name"] = ln
                break

        ordered_hits = sorted((idx, consumed, field) for field, (idx, consumed) in label_hits.items())
        for pos, (idx, consumed, field) in enumerate(ordered_hits):
            next_idx = ordered_hits[pos + 1][0] if pos + 1 < len(ordered_hits) else len(lines)
            inline = _extract_inline_value(idx)

            value_lines: List[str] = []
            if inline:
                value_lines.append(inline)

            start = idx + consumed
            while start < next_idx and continuation_label_re.search(lines[start]):
                start += 1

            for j in range(start, next_idx):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                if _looks_like_label(candidate):
                    break
                value_lines.append(candidate)
                # Most fields are one-line values; addresses and organization type may span lines.
                if field not in {"organization_type", "legal_address", "postal_address", "physical_address"} and len(value_lines) >= 1:
                    break
                if field in {"organization_type", "legal_address", "postal_address", "physical_address"} and len(value_lines) >= 3:
                    break

            value = re.sub(r"\s+", " ", " ".join(value_lines)).strip(" ,;")
            if not value:
                continue

            if field == "edrpou_code":
                m = re.search(r"\b(\d{8})\b", value)
                if m:
                    inst_data[field] = m.group(1)
                continue
            if field == "kved_code":
                kveds = re.findall(r"\b\d{2}\.\d{2}\b", value)
                if kveds:
                    inst_data[field] = ", ".join(dict.fromkeys(kveds))
                continue
            if field == "phone_number":
                m = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", value)
                if m:
                    candidate = re.sub(r"\s+", " ", m.group(1)).strip()
                    if SemanticValidator.is_phone(candidate):
                        inst_data[field] = candidate
                continue
            if field == "email":
                m = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", value)
                if m and SemanticValidator.is_email(m.group(1)):
                    inst_data[field] = m.group(1)
                continue
            if field == "website":
                m = re.search(r"(https?://[^\s]+|www\.[^\s]+)", value, re.IGNORECASE)
                if m:
                    inst_data[field] = m.group(1).rstrip(".,;")
                continue
            if field in {"legal_address", "postal_address", "physical_address"}:
                if SemanticValidator.is_address(value, labels_to_check):
                    inst_data[field] = value
                continue

            inst_data[field] = value

        # Keep legacy institution-name fallback for robustness.
        if "institution_name" not in inst_data:
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
        """Extract structured fields from PI Profile (Zone 5) using subsection-aware parsing."""
        if not structure.metadata:
            structure.metadata = {}

        pi_name, pi_title, pi_degree = SemanticValidator.clean_name_and_get_titles(structure.pi_name or "")
        pi_data: Dict[str, Any] = {"name": pi_name}
        if pi_title:
            pi_data["title"] = pi_title
        if pi_degree:
            pi_data["degree"] = pi_degree

        full_zone_text = "\n".join(block.text for block in zone.blocks if block.text)
        lines = [re.sub(r"\s+", " ", l.strip()) for l in full_zone_text.split("\n") if l and l.strip()]
        if not lines:
            structure.metadata["pi_profile"] = pi_data
            return

        def _clean_value(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", value or "").strip(" ,;:-")
            return re.sub(r"^\-\s*", "", cleaned).strip()

        def _extract_labeled_values(text: str, field_specs: Dict[str, List[str]]) -> Dict[str, str]:
            if not text:
                return {}
            compact = re.sub(r"\s+", " ", text).strip()
            if not compact:
                return {}
            all_label_patterns = [pat for patterns in field_specs.values() for pat in patterns]
            if not all_label_patterns:
                return {}
            any_label = "(?:" + "|".join(f"(?:{pat})" for pat in all_label_patterns) + ")"

            out: Dict[str, str] = {}
            for field, patterns in field_specs.items():
                for pat in patterns:
                    regex = re.compile(
                        rf"(?:^|\s){pat}\s*[:\-]?\s*(.+?)(?=\s*(?:{any_label})\s*[:\-]?|$)",
                        re.IGNORECASE,
                    )
                    match = regex.search(compact)
                    if not match:
                        continue
                    value = _clean_value(match.group(1))
                    if value:
                        out[field] = value
                        break
            return out

        def _split_profile_sections(profile_lines: List[str]) -> Dict[str, List[str]]:
            sections: Dict[str, List[str]] = {
                "personal_contact": [],
                "scientific_profile": [],
                "scientific_activity": [],
                "publications": [],
                "monographs_patents": [],
                "education": [],
                "workplaces": [],
                "scientific_degree": [],
                "academic_titles": [],
                "appendix_visual_proof": [],
            }
            heading_specs = [
                ("scientific_profile", [r"науковий\s+профіл"]),
                ("scientific_activity", [r"наукова\s+діяльн"]),
                ("publications", [r"перелік\s+праць", r"перелік\s+публікац"]),
                ("monographs_patents", [r"перелік\s+монограф", r"перелік\s+монографій\s+або\s+патент"]),
                ("education", [r"^\s*освіта\b"]),
                ("workplaces", [r"місце\s+роботи", r"та\s+посада"]),
                ("scientific_degree", [r"науковий\s+ступінь"]),
                ("academic_titles", [r"академічне\s+або\s+вчене\s+звання", r"вчене\s+звання"]),
                (
                    "appendix_visual_proof",
                    [
                        r"сертифікат",
                        r"диплом",
                        r"довідка",
                        r"додат",
                        r"curriculum\s+vitae",
                        r"контактна\s+інформац",
                        r"основні\s+наукові\s+досягнення",
                    ],
                ),
            ]
            current = "personal_contact"
            saw_structured_section = False
            cv_transition_patterns = [
                r"curriculum\s+vitae",
                r"контактна\s+інформац",
                r"основні\s+наукові\s+досягнення",
                r"досвід\s+професійн",
                r"науково[-\s]організаційн",
                r"участь\s+у\s+колективних",
                r"популяризац",
            ]

            def _is_cv_transition(line_text: str) -> bool:
                low_text = line_text.lower()
                if any(re.search(pat, low_text, re.IGNORECASE) for pat in cv_transition_patterns):
                    return True
                if re.match(r"^\d+\)\s+", line_text) and len(line_text) > 45:
                    return True
                if len(line_text) > 120 and sum(1 for _ in re.finditer(r"[,:;]", line_text)) >= 3:
                    return True
                return False

            for line in profile_lines:
                low = line.lower()
                matched = False
                for key, patterns in heading_specs:
                    if any(re.search(pat, low, re.IGNORECASE) for pat in patterns):
                        current = key
                        if key != "appendix_visual_proof":
                            saw_structured_section = True
                        else:
                            sections["appendix_visual_proof"].append(line)
                        matched = True
                        break
                if matched:
                    continue
                if (
                    current != "appendix_visual_proof"
                    and saw_structured_section
                    and current in {"academic_titles", "workplaces", "education", "scientific_degree"}
                    and _is_cv_transition(line)
                ):
                    current = "appendix_visual_proof"
                sections[current].append(line)
            return sections

        def _extract_list_items(section_lines: List[str]) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            current: List[str] = []
            for raw in section_lines:
                line = _clean_value(raw)
                if not line:
                    continue
                if re.search(r"конкурс[іу]|не\s+більше", line, re.IGNORECASE):
                    continue
                starts_new = bool(re.match(r"^\d{1,2}[.)]\s+", line)) or bool(re.search(r"\b10\.\d{4,9}/\S+", line, re.IGNORECASE))
                if starts_new and current:
                    joined = _clean_value(" ".join(current))
                    if joined and len(joined) > 12:
                        items.append({"text": joined})
                    current = []
                current.append(re.sub(r"^\d{1,2}[.)]\s+", "", line))
            if current:
                joined = _clean_value(" ".join(current))
                if joined and len(joined) > 12:
                    items.append({"text": joined})
            for idx, item in enumerate(items, start=1):
                item["item_index"] = idx
                doi_match = re.search(r"\b10\.\d{4,9}/\S+", item["text"], re.IGNORECASE)
                if doi_match:
                    item["doi"] = doi_match.group(0).rstrip(".,;")
                    body = _clean_value(item["text"].replace(item["doi"], "", 1))
                    year_matches = re.findall(r"\b((?:19|20)\d{2})\b", body)
                    if year_matches:
                        item["year"] = year_matches[-1]
                    title_match = re.search(r"\b([A-Z][A-Z0-9\-\(\),']+(?:\s+[A-Z][A-Z0-9\-\(\),']+){3,})\b", body)
                    if title_match:
                        authors = _clean_value(body[:title_match.start()])
                        title = _clean_value(title_match.group(1))
                        journal = _clean_value(body[title_match.end():])
                        if authors and "," in authors:
                            item["authors_list"] = authors
                        if title:
                            item["title"] = title
                        if journal:
                            item["journal"] = journal
            return items

        def _extract_typed_monograph_items(section_lines: List[str]) -> List[Dict[str, Any]]:
            items = _extract_list_items(section_lines)
            if len(items) <= 1:
                merged_text = _clean_value(" ".join(section_lines))
                split_starts = [
                    m.start()
                    for m in re.finditer(
                        r"(?:№\s*)?(?:\d{5,6}|97[89][\-\d]{8,20})\s*,?\s*(?:19|20)\d{2}\s*[:\-]",
                        merged_text,
                        flags=re.IGNORECASE,
                    )
                ]
                if len(split_starts) >= 2:
                    split_items: List[Dict[str, Any]] = []
                    for idx, start in enumerate(split_starts):
                        end = split_starts[idx + 1] if idx + 1 < len(split_starts) else len(merged_text)
                        chunk = _clean_value(merged_text[start:end])
                        if chunk:
                            split_items.append({"text": chunk})
                    if split_items:
                        items = split_items
            for item in items:
                low = item.get("text", "").lower()
                if "монограф" in low or re.search(r"\b97[89]-\d", item.get("text", "")):
                    item["item_type"] = "monograph"
                elif (
                    "патент" in low
                    or re.search(r"\b№\s*\S+", item.get("text", ""))
                    or re.search(r"^\s*\d{5,6}\s*,?\s*(?:19|20)\d{2}\s*[:\-]", item.get("text", ""))
                ):
                    item["item_type"] = "patent"
                else:
                    item["item_type"] = "other"
            for idx, item in enumerate(items, start=1):
                item["item_index"] = idx
            return items

        def _extract_academic_titles(section_lines: List[str]) -> List[str]:
            titles: List[str] = []
            title_markers = [
                "професор",
                "доцент",
                "старший науковий співробітник",
                "член-кореспондент",
                "академік",
            ]
            for line in section_lines:
                for chunk in re.split(r"[;,]", line):
                    value = _clean_value(chunk)
                    if (
                        value
                        and len(value) > 2
                        and not re.search(r"звання", value, re.IGNORECASE)
                        and any(marker in value.lower() for marker in title_markers)
                    ):
                        titles.append(value)
            deduped: List[str] = []
            seen = set()
            for title in titles:
                low = title.lower()
                if low in seen:
                    continue
                seen.add(low)
                deduped.append(title)
            return deduped

        def _extract_subsection_fields(
            section_lines: List[str],
            label_patterns: Dict[str, List[str]],
            multiline_fields: Optional[Set[str]] = None,
        ) -> Dict[str, str]:
            multiline_fields = multiline_fields or set()
            lines_local = [_clean_value(x) for x in section_lines if _clean_value(x)]
            if not lines_local:
                return {}

            def _line_is_label(line_text: str) -> bool:
                for patterns in label_patterns.values():
                    for pat in patterns:
                        if re.search(pat, line_text, re.IGNORECASE):
                            return True
                return False

            extracted: Dict[str, str] = {}
            idx = 0
            while idx < len(lines_local):
                line = lines_local[idx]
                matched_field: Optional[str] = None
                matched_end = -1
                for field, patterns in label_patterns.items():
                    for pat in patterns:
                        m = re.search(pat, line, re.IGNORECASE)
                        if not m:
                            continue
                        if matched_field is None or m.start() < matched_end:
                            matched_field = field
                            matched_end = m.end()
                if not matched_field:
                    idx += 1
                    continue

                values: List[str] = []
                inline = _clean_value(line[matched_end:])
                if inline and not _line_is_label(inline):
                    values.append(inline)

                j = idx + 1
                while j < len(lines_local):
                    nxt = lines_local[j]
                    if _line_is_label(nxt):
                        break
                    values.append(nxt)
                    if matched_field not in multiline_fields:
                        break
                    if len(values) >= 4:
                        break
                    j += 1

                value = _clean_value(" ".join(values))
                if value and matched_field not in extracted:
                    extracted[matched_field] = value
                idx = max(j, idx + 1)
            return extracted

        def _infer_institution_name(section_lines: List[str], label_patterns: Dict[str, List[str]]) -> Optional[str]:
            candidates: List[str] = []
            for raw in section_lines:
                line = _clean_value(raw)
                if not line:
                    continue
                if re.search(r"https?://|www\.", line, re.IGNORECASE):
                    continue
                if re.search(r"\b\d{4}\s*[-–]\s*\d{4}\b", line):
                    continue
                if any(re.search(pat, line, re.IGNORECASE) for pats in label_patterns.values() for pat in pats):
                    continue
                if re.search(r"інститут|університет|академ|нан\s+україни|кпі", line, re.IGNORECASE):
                    candidates.append(line)
            if not candidates:
                return None
            return max(candidates, key=len)

        def _normalize_ua_phone(value: str) -> Optional[str]:
            if not value:
                return None
            digits = re.sub(r"\D", "", value)
            if digits.startswith("80") and len(digits) == 11:
                digits = "3" + digits
            elif digits.startswith("0") and len(digits) == 10:
                digits = "38" + digits
            elif digits.startswith("44") and len(digits) == 9:
                digits = "380" + digits
            candidate = f"+{digits}" if digits else None
            return candidate if candidate and SemanticValidator.is_phone(candidate) else None

        sections = _split_profile_sections(lines)
        personal_text = " ".join(sections["personal_contact"])
        personal_fields = _extract_labeled_values(
            personal_text,
            {
                "gender": [r"стать"],
                "birth_date": [r"дата\s+народження"],
                "country_of_residence": [r"країна\s+постійного\s+проживання"],
                "citizenship": [r"громадянство"],
                "mobile_phone": [r"мобільний\s+телефон", r"(?<!робочий\s)телефон"],
                "email": [r"e-?mail", r"електронна\s+пошта"],
                "other_contacts": [r"інші\s+контакти(?:\s*\([^)]+\))?"],
            },
        )

        for key in ("gender", "birth_date", "country_of_residence"):
            if personal_fields.get(key):
                pi_data[key] = personal_fields[key]
        if personal_fields.get("citizenship"):
            cit = personal_fields["citizenship"]
            if re.search(r"мобіль|телефон|пошта|контакт", cit, re.IGNORECASE):
                cit = ""
            if cit and SemanticValidator.is_citizenship(cit):
                pi_data["citizenship"] = cit
        if personal_fields.get("mobile_phone") and SemanticValidator.is_phone(personal_fields["mobile_phone"]):
            pi_data["mobile_phone"] = personal_fields["mobile_phone"]
            # Keep backward-compatible alias while preserving split-contact schema.
            pi_data["phone"] = personal_fields["mobile_phone"]
        if personal_fields.get("email") and SemanticValidator.is_email(personal_fields["email"]):
            pi_data["email"] = personal_fields["email"].lower()
        if personal_fields.get("other_contacts"):
            pi_data["other_contacts"] = personal_fields["other_contacts"]

        scientific_profile_text = " ".join(sections["scientific_profile"])
        scientific_profile_fields = _extract_labeled_values(
            scientific_profile_text,
            {
                "scientific_experience_years": [r"науковий\s+стаж[,\s]+кількість\s+років"],
                "total_patents": [r"загальна\s+кількість\s+патентів"],
                "total_publications": [r"загальна\s+кількість\s+публікацій"],
                "q1_q2_publications_10years": [r"кількість\s+публікацій\s+у\s+виданнях\s+1-го\s+[—\-]\s*2-го\s+квартил"],
                "h_index_scopus": [r"індекс\s+хірша\s*\(?.{0,10}scopus.*?\)?"],
                "total_monographs": [r"кількість\s+монографій"],
                "total_grants": [r"гранти[, ]+отримані\s+на\s+дослідження"],
                "expert_experience": [r"досвід\s+проведення\s+експертизи"],
            },
        )
        profile_urls = re.findall(r"(https?://[^\s,;]+|www\.[^\s,;]+)", scientific_profile_text, re.IGNORECASE)
        profile_urls = [u.rstrip(".,;") for u in profile_urls]
        scientific_profile_obj: Dict[str, Any] = {}
        if profile_urls:
            scientific_profile_obj["profile_urls"] = list(dict.fromkeys(profile_urls))
        metrics = {k: v for k, v in scientific_profile_fields.items() if k != "expert_experience" and v}
        if metrics:
            scientific_profile_obj["metrics"] = metrics
        if scientific_profile_fields.get("expert_experience"):
            scientific_profile_obj["expert_experience"] = scientific_profile_fields["expert_experience"]
        if scientific_profile_obj:
            pi_data["scientific_profile"] = scientific_profile_obj
        if scientific_profile_fields.get("total_publications"):
            pi_data["total_publications"] = scientific_profile_fields["total_publications"]
        if scientific_profile_fields.get("h_index_scopus"):
            pi_data["h_index_scopus"] = scientific_profile_fields["h_index_scopus"]
        if scientific_profile_fields.get("q1_q2_publications_10years"):
            q = scientific_profile_fields["q1_q2_publications_10years"]
            q_nums = re.findall(r"\d+", q)
            if q_nums and scientific_profile_obj.get("metrics"):
                scientific_profile_obj["metrics"]["q1_q2_publications_10years"] = q_nums[-1]
        if scientific_profile_fields.get("total_grants"):
            grants = scientific_profile_fields["total_grants"]
            if not re.search(r"\d", grants) or re.search(r"зокрема\s+гранти", grants, re.IGNORECASE):
                if scientific_profile_obj.get("metrics"):
                    scientific_profile_obj["metrics"].pop("total_grants", None)

        orcid_norm, orcid_raw = SemanticValidator.normalize_orcid(f"{personal_text} {scientific_profile_text}")
        if orcid_norm:
            pi_data["orcid"] = orcid_norm
            pi_data["orcid_raw"] = orcid_norm

        activity_text = " ".join(sections["scientific_activity"])
        activity_fields = _extract_labeled_values(
            activity_text,
            {
                "scientific_direction": [r"науковий\s+напрям"],
                "science_branch": [r"галузь\s+науки"],
                "publications_in_expertise": [r"кількість\s+публікацій\s+за\s+галуззю"],
                "keywords": [r"ключові\s+слова"],
            },
        )
        if activity_fields:
            scientific_activity: Dict[str, Any] = {}
            if activity_fields.get("scientific_direction"):
                scientific_activity["scientific_direction"] = activity_fields["scientific_direction"]
            if activity_fields.get("science_branch"):
                scientific_activity["science_branch"] = activity_fields["science_branch"]
            if activity_fields.get("publications_in_expertise"):
                pub_exp = activity_fields["publications_in_expertise"]
                nums = re.findall(r"\d+", pub_exp)
                scientific_activity["publications_in_expertise"] = nums[-1] if nums else pub_exp
            if activity_fields.get("keywords"):
                keywords = [
                    _clean_value(x) for x in re.split(r"[;,]", activity_fields["keywords"]) if _clean_value(x)
                ]
                scientific_activity["keywords"] = keywords if len(keywords) > 1 else activity_fields["keywords"]
            if scientific_activity:
                pi_data["scientific_activity"] = scientific_activity

        publications_list = _extract_list_items(sections["publications"])
        if publications_list:
            pi_data["publications_list"] = publications_list

        monographs_patents = _extract_typed_monograph_items(sections["monographs_patents"])
        if monographs_patents:
            pi_data["monographs_patents"] = monographs_patents

        edu_label_patterns = {
            "institution_name": [r"навчальн\w*\s+заклад", r"місце\s+навчання", r"заклад\s+освіт"],
            "country": [r"країна"],
            "city": [r"місто"],
            "faculty": [r"факультет"],
            "speciality": [r"спеціальність"],
            "diploma_number": [r"номер\s+диплому"],
            "diploma_issue_date": [r"дата\s+видачі\s+диплому"],
        }
        education_fields = _extract_subsection_fields(
            sections["education"],
            edu_label_patterns,
            multiline_fields={"speciality", "institution_name"},
        )
        if education_fields or sections["education"]:
            education_item: Dict[str, Any] = dict(education_fields)
            inferred_edu_inst = _infer_institution_name(sections["education"], edu_label_patterns)
            if inferred_edu_inst and not education_item.get("institution_name"):
                education_item["institution_name"] = inferred_edu_inst

            speciality = education_item.get("speciality")
            if speciality:
                spec_match = re.search(
                    r"^(?P<spec>.+?)\s+(?P<inst>[А-ЯA-ZІЇЄҐ][^,]{6,}(?:інститут|університет|академ[^,]*)[^,]*)(?:,\s*(?P<period>\d{4}\s*[-–]\s*\d{4}))?$",
                    speciality,
                    re.IGNORECASE,
                )
                if spec_match:
                    education_item["speciality"] = _clean_value(spec_match.group("spec"))
                    if not education_item.get("institution_name"):
                        education_item["institution_name"] = _clean_value(spec_match.group("inst"))
                    if spec_match.group("period"):
                        education_item["study_period"] = _clean_value(spec_match.group("period"))
                else:
                    education_item["speciality"] = _clean_value(speciality)

            if education_item.get("country") in {"У", "у"}:
                education_item["country"] = "Україна"
            if education_item.get("country") in {"У", "у"}:
                education_item.pop("country", None)
            if not education_item.get("country") and pi_data.get("country_of_residence"):
                education_item["country"] = pi_data.get("country_of_residence")

            if education_item:
                pi_data["education"] = [education_item]

        work_label_patterns = {
            "institution_name": [r"місце\s+роботи(?:\s+та\s+посада)?", r"установа\s+де\s+працює"],
            "position": [r"посада"],
            "period": [r"період\s+роботи"],
            "subordination": [r"підпорядкован"],
            "edrpou": [r"єдрпоу"],
            "country": [r"країна"],
            "city": [r"місто"],
            "address": [r"адреса\s+установи", r"адреса"],
            "work_phone": [r"робочий\s+телефон", r"телефон"],
        }
        workplace_fields = _extract_subsection_fields(
            sections["workplaces"],
            work_label_patterns,
            multiline_fields={"institution_name", "address", "subordination"},
        )
        if workplace_fields or sections["workplaces"]:
            workplace_item: Dict[str, Any] = dict(workplace_fields)
            inferred_work_inst = _infer_institution_name(sections["workplaces"], work_label_patterns)
            if inferred_work_inst and not workplace_item.get("institution_name"):
                workplace_item["institution_name"] = inferred_work_inst

            sub = workplace_item.get("subordination")
            if sub:
                sub = _clean_value(sub)
                sub = re.sub(r"^[iі]?\s*сть\s+", "", sub, flags=re.IGNORECASE)
                sub = re.sub(r"^[^\s]{0,6}ість\s+", "", sub, flags=re.IGNORECASE)
                workplace_item["subordination"] = sub

            if workplace_item.get("country") in {"У", "у"}:
                workplace_item["country"] = "Україна"
            if workplace_item.get("country") in {"У", "у"}:
                workplace_item.pop("country", None)
            if workplace_item.get("country") and re.fullmatch(r"\d{4,}", workplace_item["country"]):
                workplace_item.pop("country", None)
            if not workplace_item.get("country") and any("укра" in ln.lower() for ln in sections["workplaces"]):
                workplace_item["country"] = "Україна"

            if workplace_item.get("address"):
                addr = _clean_value(workplace_item["address"])
                addr = re.sub(r"^установи\b[:\s\-]*", "", addr, flags=re.IGNORECASE).strip()
                workplace_item["address"] = addr
            if not workplace_item.get("address") or len(workplace_item.get("address", "")) < 10:
                addr_candidates = [
                    _clean_value(ln)
                    for ln in sections["workplaces"]
                    if re.search(r"просп|вул\.?|вулиц|київ|буд\.?|адрес", ln, re.IGNORECASE)
                    and not re.search(r"^адреса\s+установи\b", ln, re.IGNORECASE)
                ]
                if addr_candidates:
                    workplace_item["address"] = max(addr_candidates, key=len)

            if workplace_item.get("work_phone"):
                norm_work_phone = _normalize_ua_phone(workplace_item["work_phone"])
                if norm_work_phone:
                    workplace_item["work_phone"] = norm_work_phone

            if workplace_item:
                pi_data["workplaces"] = [workplace_item]
                if workplace_item.get("position"):
                    pi_data["position"] = workplace_item["position"]
        if not pi_data.get("position"):
            for line in sections["workplaces"]:
                m = re.search(r"\bпосада\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
                if m:
                    pi_data["position"] = _clean_value(m.group(1))
                    break

        degree_lines = sections["scientific_degree"]
        degree_text = " ".join(degree_lines)
        degree_fields = _extract_labeled_values(
            degree_text,
            {
                "diploma_number": [r"номер\s+диплому"],
                "diploma_issue_date": [r"дата\s+видачі\s+диплому"],
            },
        )
        degree_name = ""
        for line in degree_lines:
            if re.search(r"номер\s+диплому|дата\s+видачі", line, re.IGNORECASE):
                continue
            if line:
                degree_name = line
                break
        if not degree_name:
            m = re.search(r"науковий\s+ступінь\s*[:\-]?\s*(.+)$", degree_text, re.IGNORECASE)
            if m:
                degree_name = _clean_value(m.group(1))
        if degree_name or degree_fields:
            scientific_degree: Dict[str, Any] = {}
            if degree_name:
                scientific_degree["name"] = degree_name
            scientific_degree.update(degree_fields)
            pi_data["scientific_degree"] = scientific_degree
            if degree_name:
                pi_data["degree_raw"] = degree_name
                if not pi_data.get("degree"):
                    pi_data["degree"] = degree_name

        academic_titles = _extract_academic_titles(sections["academic_titles"])
        if academic_titles:
            pi_data["academic_titles"] = academic_titles

        appendix_lines = sections["appendix_visual_proof"]
        appendix_text = " ".join(appendix_lines)
        if pi_data.get("scientific_degree") and isinstance(pi_data["scientific_degree"], dict):
            degree_obj = pi_data["scientific_degree"]
            selected_diploma_start: Optional[int] = None
            selected_diploma_end: Optional[int] = None
            if not degree_obj.get("diploma_number"):
                diploma_candidates = []
                for m in re.finditer(r"(?:ДД|ДК)\s*№\s*\d{4,8}", appendix_text, re.IGNORECASE):
                    ctx = appendix_text[max(0, m.start() - 160): m.end() + 120]
                    score = 2 if re.search(r"доктор|наук", ctx, re.IGNORECASE) else 1
                    diploma_candidates.append((score, m.start(), m.end(), m.group(0).replace(" ", "")))
                if diploma_candidates:
                    diploma_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    selected_diploma_start = diploma_candidates[0][1]
                    selected_diploma_end = diploma_candidates[0][2]
                    degree_obj["diploma_number"] = diploma_candidates[0][3]
            else:
                diploma_pattern = re.escape(degree_obj["diploma_number"]).replace("\\№", r"\s*№\s*")
                m = re.search(diploma_pattern, appendix_text, re.IGNORECASE)
                if m:
                    selected_diploma_start = m.start()
                    selected_diploma_end = m.end()
            if not degree_obj.get("diploma_issue_date"):
                if selected_diploma_start is not None:
                    tail_from_diploma = appendix_text[selected_diploma_end or selected_diploma_start:]
                    nearby_after = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", tail_from_diploma)
                    if nearby_after and nearby_after.start() <= 120:
                        degree_obj["diploma_issue_date"] = nearby_after.group(0)
                if not degree_obj.get("diploma_issue_date"):
                    date_candidates = []
                    for m in re.finditer(r"\b\d{2}\.\d{2}\.\d{4}\b", appendix_text):
                        ctx = appendix_text[max(0, m.start() - 180): m.end() + 120]
                        score = 1
                        if re.search(r"дата\s+видачі\s+диплому|дата\s+видачі", ctx, re.IGNORECASE):
                            score += 4
                        if re.search(r"доктор|наук|дд\s*№|дк\s*№", ctx, re.IGNORECASE):
                            score += 2
                        distance = 10_000
                        if selected_diploma_start is not None:
                            distance = abs(m.start() - selected_diploma_start)
                            if distance <= 160:
                                score += 3
                            elif distance <= 300:
                                score += 1
                        date_candidates.append((score, -distance, m.start(), m.group(0)))
                    if date_candidates:
                        date_candidates.sort(reverse=True)
                        degree_obj["diploma_issue_date"] = date_candidates[0][3]
        visual_markers = ["сертифікат", "диплом", "довідка", "додаток", "certificate", "diploma", "curriculum vitae", "cv"]
        appendix_hits = [
            line for line in appendix_lines
            if any(marker in line.lower() for marker in visual_markers)
        ]
        if appendix_lines:
            pi_data["cv_appendix"] = {
                "detected": True,
                "line_count": len(appendix_lines),
                "preview_lines": appendix_lines[:20],
                "residual_text": " ".join(appendix_lines),
            }
        if appendix_hits:
            pi_data["visual_proof"] = {
                "possible_scanned_appendix": True,
                "supporting_phrases": appendix_hits[:5],
                "handling_note": "Structured CV/profile text parsed locally; appendix pages can be external-fallback candidates later.",
            }
        pi_data["structured_profile_local_parse"] = True

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
        """Extract Zone 6 co-authors as repeated Zone 5-like records with explicit ownership."""
        if not structure.metadata:
            structure.metadata = {}

        def _normalize_identity_text(value: Optional[str]) -> str:
            if not value:
                return ""
            normalized = value.strip().lower().replace("’", "'")
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = re.sub(r"[^\w\s\-']", "", normalized)
            return normalized

        def _extract_start_name(line: str) -> Optional[str]:
            clean_line = re.sub(r"\s+", " ", (line or "").strip())
            if not clean_line:
                return None
            non_name_tokens = {
                "кафедри", "технічних", "наук", "доктор", "професор", "старший",
                "науковий", "співробітник", "інститут", "академії", "академія",
                "університет", "університету", "відділу", "посада", "місце", "роботи",
            }
            # Strong start anchor: honorific/title + full person name.
            m = re.search(
                r"^(?:Пан|Пані|Професор|Доктор|К\.т\.н\.|Д\.т\.н\.|Ph\.D\.)\s+"
                r"([А-ЯЄІЇҐ][а-яєіїґ'`-]+\s+[А-ЯЄІЇҐ][а-яєіїґ'`-]+(?:\s+[А-ЯЄІЇҐ][а-яєіїґ'`-]+)?)",
                clean_line,
                re.IGNORECASE,
            )
            if not m:
                return None
            full = clean_line[: m.end()]
            name, _, _ = SemanticValidator.clean_name_and_get_titles(full)
            name_tokens = [t.lower() for t in name.split() if t]
            if (
                name
                and len(name_tokens) >= 2
                and not any(tok in non_name_tokens for tok in name_tokens)
            ):
                return name
            return None

        line_items: List[Dict[str, Any]] = []
        for b_idx, block in enumerate(zone.blocks):
            if not block.text:
                continue
            for raw_line in block.text.split("\n"):
                clean_line = re.sub(r"\s+", " ", raw_line.strip())
                if not clean_line:
                    continue
                line_items.append(
                    {
                        "text": clean_line,
                        "page_number": block.page_number or zone.page_start,
                        "block_index": b_idx,
                    }
                )

        starts: List[Tuple[int, str]] = []
        for idx, item in enumerate(line_items):
            start_name = _extract_start_name(item["text"])
            if start_name:
                starts.append((idx, start_name))

        records: List[Dict[str, Any]] = []
        for i, (start_idx, start_name) in enumerate(starts):
            end_idx = starts[i + 1][0] if i + 1 < len(starts) else len(line_items)
            seg_lines = line_items[start_idx:end_idx]
            if not seg_lines:
                continue
            seg_text = "\n".join(li["text"] for li in seg_lines)
            seg_pages = sorted({int(li["page_number"]) for li in seg_lines if li.get("page_number")})
            seg_blocks = sorted({int(li["block_index"]) for li in seg_lines if li.get("block_index") is not None})
            records.append(
                {
                    "name_hint": start_name,
                    "text": seg_text,
                    "pages": seg_pages,
                    "block_indices": seg_blocks,
                    "start_anchor": seg_lines[0]["text"],
                }
            )

        team: List[Dict[str, Any]] = []
        seen_orcids: Dict[str, int] = {}
        seen_names: Dict[str, int] = {}

        for rec in records:
            tmp_zone = DocumentZone(
                name="Zone 6 — Co-author Record",
                zone_type="pi_profile",
                page_start=rec["pages"][0] if rec["pages"] else zone.page_start,
                page_end=rec["pages"][-1] if rec["pages"] else zone.page_start,
                sections=[],
                blocks=[
                    DocumentBlock(
                        block_type="profile_block",
                        text=rec["text"],
                        page_number=rec["pages"][0] if rec["pages"] else zone.page_start,
                        metadata={},
                    )
                ],
            )
            tmp_structure = ParsedDocumentStructure(
                pi_name=rec["name_hint"],
                zones=[tmp_zone],
                sections=[],
                markers={},
                metadata={},
                tables=[],
                page_table_classifications=[],
            )
            self._extract_pi_profile(tmp_zone, pages, tmp_structure)
            profile = (tmp_structure.metadata or {}).get("pi_profile") or {}
            if not profile.get("name"):
                profile["name"] = rec["name_hint"]

            ownership = {
                "start_anchor": rec["start_anchor"],
                "pages": rec["pages"],
                "page_start": rec["pages"][0] if rec["pages"] else None,
                "page_end": rec["pages"][-1] if rec["pages"] else None,
                "source_block_count": len(rec["block_indices"]),
            }
            profile["record_ownership"] = ownership
            if isinstance(profile.get("cv_appendix"), dict):
                profile["cv_appendix"]["owner_pages"] = rec["pages"]

            key_orcid = _normalize_identity_text(profile.get("orcid"))
            key_name = _normalize_identity_text(profile.get("name"))
            duplicate_idx: Optional[int] = None
            if key_orcid and key_orcid in seen_orcids:
                duplicate_idx = seen_orcids[key_orcid]
            elif key_name and key_name in seen_names:
                duplicate_idx = seen_names[key_name]

            if duplicate_idx is None:
                team.append(profile)
                idx = len(team) - 1
                if key_orcid:
                    seen_orcids[key_orcid] = idx
                if key_name:
                    seen_names[key_name] = idx
            else:
                existing = team[duplicate_idx]
                for field_name, field_value in profile.items():
                    if field_value and not existing.get(field_name):
                        existing[field_name] = field_value
                if existing.get("record_ownership") and profile.get("record_ownership"):
                    existing_pages = set(existing["record_ownership"].get("pages") or [])
                    incoming_pages = set(profile["record_ownership"].get("pages") or [])
                    merged_pages = sorted(existing_pages | incoming_pages)
                    existing["record_ownership"]["pages"] = merged_pages
                    if merged_pages:
                        existing["record_ownership"]["page_start"] = merged_pages[0]
                        existing["record_ownership"]["page_end"] = merged_pages[-1]
                    existing["record_ownership"]["source_block_count"] = (
                        int(existing["record_ownership"].get("source_block_count") or 0)
                        + int(profile["record_ownership"].get("source_block_count") or 0)
                    )

        structure.metadata["team_profiles"] = team
        structure.metadata["team_profile_completeness"] = {
            "detected_starts": len(starts),
            "extracted_entities": len(team),
            "unresolved_starts": [],
            "unresolved_pages": [],
        }

        new_blocks = [b for b in zone.blocks if b.block_type != "profile_block"]
        for person in team:
            ownership = person.get("record_ownership") or {}
            page_num = ownership.get("page_start") or zone.page_start
            new_blocks.append(
                DocumentBlock(
                    block_type="profile_block",
                    text="[STRUCTURED DATA]",
                    metadata=person,
                    page_number=page_num,
                )
            )
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

    def _collect_parser_feature_flags(self) -> Dict[str, Any]:
        flags = {
            "DEBUG_PARSING": os.getenv("DEBUG_PARSING") == "true",
            "USE_LITEPARSE_ENRICHMENT": settings.USE_LITEPARSE_ENRICHMENT,
            "USE_LITEPARSE_SCANNED_TABLE_ROUTING": settings.USE_LITEPARSE_SCANNED_TABLE_ROUTING,
            "USE_SCANNED_CERT_VISION_EVIDENCE": settings.USE_SCANNED_CERT_VISION_EVIDENCE,
            "LITEPARSE_OCR_ENABLED": settings.LITEPARSE_OCR_ENABLED,
            "LITEPARSE_OCR_ENGINE": settings.LITEPARSE_OCR_ENGINE,
            "LITEPARSE_SCREENSHOT_ENABLED": settings.LITEPARSE_SCREENSHOT_ENABLED,
            "LITEPARSE_USE_CLI": settings.LITEPARSE_USE_CLI,
            "LITEPARSE_CLI_PATH_SETTING": settings.LITEPARSE_CLI_PATH,
        }
        try:
            flags.update(self.enrichment_service.get_runtime_info())
        except Exception:
            pass
        return flags

    def _build_canonical_debug_artifact(
        self,
        parsing_result: Dict[str, Any],
        input_file_name: Optional[str],
        input_file_path: Optional[str],
    ) -> Dict[str, Any]:
        timestamp_utc = datetime.now(timezone.utc)
        run_id = f"parse_{timestamp_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

        return {
            "run_id": run_id,
            "timestamp_utc": timestamp_utc.isoformat().replace("+00:00", "Z"),
            "input": {
                "file_name": input_file_name or "unknown",
                "file_path": input_file_path,
            },
            "feature_flags": self._collect_parser_feature_flags(),
            "parsing_result": parsing_result,
        }

    def _build_page_inspection_summary(
        self,
        structure: ParsedDocumentStructure,
    ) -> Dict[str, Any]:
        def _collect_section_blocks(sec: DocumentSection, collector: List[DocumentBlock]) -> None:
            collector.extend(sec.blocks)
            for sub in sec.subsections:
                _collect_section_blocks(sub, collector)

        page_zone: Dict[int, str] = {}
        for z in structure.zones:
            for p in range(z.page_start, (z.page_end or z.page_start) + 1):
                if p not in page_zone:
                    page_zone[p] = z.zone_type

        page_blocks: Dict[int, int] = {}
        for z in structure.zones:
            all_blocks: List[DocumentBlock] = []
            all_blocks.extend(z.blocks)
            for sec in z.sections:
                _collect_section_blocks(sec, all_blocks)
            for b in all_blocks:
                if not b.page_number:
                    continue
                page_blocks[b.page_number] = page_blocks.get(b.page_number, 0) + 1

        page_tables: Dict[int, Dict[str, Any]] = {}
        for t in structure.tables:
            start = t.page_start or 0
            end = t.page_end or start
            for p in range(start, end + 1):
                slot = page_tables.setdefault(p, {"table_families": [], "table_ids": []})
                if t.table_family not in slot["table_families"]:
                    slot["table_families"].append(t.table_family)
                if t.table_id and t.table_id not in slot["table_ids"]:
                    slot["table_ids"].append(t.table_id)

        page_class: Dict[int, str] = {
            c.page_number: c.page_class for c in structure.page_table_classifications
        }
        page_family_hint: Dict[int, Optional[str]] = {
            c.page_number: (c.signals or {}).get("scanned_family_hint")
            for c in structure.page_table_classifications
        }
        page_family_conf: Dict[int, Optional[float]] = {
            c.page_number: (c.signals or {}).get("scanned_family_confidence")
            for c in structure.page_table_classifications
        }

        all_pages: Set[int] = set(page_zone.keys()) | set(page_blocks.keys()) | set(page_tables.keys()) | set(page_class.keys())
        rows: List[Dict[str, Any]] = []
        for p in sorted(all_pages):
            table_meta = page_tables.get(p, {"table_families": [], "table_ids": []})
            rows.append({
                "page_number": p,
                "zone_type": page_zone.get(p),
                "page_class": page_class.get(p),
                "scanned_family_hint": page_family_hint.get(p),
                "scanned_family_confidence": page_family_conf.get(p),
                "block_count": page_blocks.get(p, 0),
                "has_table_object": bool(table_meta["table_ids"]),
                "table_families": sorted(table_meta["table_families"]),
                "table_ids": sorted(table_meta["table_ids"]),
            })

        return {
            "note": "Inspection aid only: scanned_image_only pages may still have canonical table objects.",
            "rows": rows,
        }

    def _save_canonical_debug_artifact(self, artifact: Dict[str, Any]) -> str:
        debug_dir = "debug/parsing"
        runs_dir = os.path.join(debug_dir, "runs")
        run_id = artifact["run_id"]
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        parse_output_path = os.path.join(run_dir, "parse_output.json")
        with open(parse_output_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)

        page_inspection_summary_path = os.path.join(debug_dir, "page_inspection_summary.json")
        run_page_inspection_summary_path = os.path.join(run_dir, "page_inspection_summary.json")
        if os.path.exists(page_inspection_summary_path):
            try:
                with open(page_inspection_summary_path, "r", encoding="utf-8") as src:
                    summary_obj = json.load(src)
                with open(run_page_inspection_summary_path, "w", encoding="utf-8") as dst:
                    json.dump(summary_obj, dst, ensure_ascii=False, indent=2)
            except Exception:
                run_page_inspection_summary_path = None
        else:
            run_page_inspection_summary_path = None

        latest_run = {
            "run_id": run_id,
            "timestamp_utc": artifact["timestamp_utc"],
            "parse_output_path": parse_output_path,
            "input": artifact.get("input", {}),
            "feature_flags": artifact.get("feature_flags", {}),
            "legacy_artifacts": {
                "debug_parse_output": os.path.join(debug_dir, "debug_parse_output.json"),
                "normalized_structure": os.path.join(debug_dir, "normalized_structure.json"),
                "page_inspection_summary": page_inspection_summary_path,
            },
            "run_artifacts": {
                "page_inspection_summary": run_page_inspection_summary_path,
            },
        }
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "latest_run.json"), "w", encoding="utf-8") as f:
            json.dump(latest_run, f, ensure_ascii=False, indent=2)

        return parse_output_path

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
