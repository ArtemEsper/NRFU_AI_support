import fitz  # PyMuPDF
from typing import Dict, Any, List, Optional
import re
import os
import tempfile
import json
import logging
from app.core.logger import logger
from app.services.layout_enrichment import LayoutEnrichmentService
from app.schemas.schemas import DocumentZone, DocumentSection, ParsedDocumentStructure, PageClassification, DocumentBlock

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

        # Deterministic Zone/Section Reconstruction
        structure = self._reconstruct_structure(doc, pages_text, full_text)
        
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
            
            # Save normalized structure without raw page text and with cleaned structure
            self._save_debug_json(parsing_result, filename="normalized_structure.json")

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

    def _reconstruct_structure(self, doc: fitz.Document, pages_text: List[str], full_text: str = "") -> ParsedDocumentStructure:
        """
        Implements deterministic zone/section reconstruction based on NRFU Spec (Master Spec).
        Strictly follows Zone Boundary Rules and page-based transition semantics.
        """
        structure = ParsedDocumentStructure()
        structure.pages = pages_text
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
                        
                        if is_header:
                            b_type = "section_header"
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
                    if b.block_type != "paragraph":
                        if current_block:
                            reconstructed_blocks.append(current_block)
                            current_block = None
                        reconstructed_blocks.append(b)
                        continue
                    
                    if current_block is None:
                        current_block = DocumentBlock(
                            block_type="paragraph",
                            text=b.text,
                            page_number=b.page_number
                        )
                    else:
                        # Logic to merge
                        prev_text = current_block.text.strip()
                        curr_text = b.text.strip()
                        
                        should_merge = False
                        if prev_text:
                            # 1. Merge if previous line does NOT end with sentence-ending punctuation
                            # and next line starts with lowercase or continuation
                            last_char = prev_text[-1]
                            ends_sentence = last_char in [".", "!", "?", ":", ";"]
                            
                            starts_with_lower = curr_text and curr_text[0].islower()
                            # Common Ukrainian/Russian continuation characters or specific NRFU markers
                            is_continuation = curr_text and curr_text[0] in ["(", ")", ",", "-", "–", "—"]
                            
                            # Bullet point detection - bullets should start a NEW paragraph
                            is_bullet = curr_text and (curr_text.startswith("•") or curr_text.startswith("-") or curr_text.startswith("*"))
                            
                            # 3. Paragraph merging (target 3-10 lines)
                            # We only merge if it's not a bullet and not already too long
                            if not is_bullet and (not ends_sentence or starts_with_lower or is_continuation):
                                # Heuristic: limit block length to avoid massive blocks
                                if len(prev_text) < 2000:
                                    should_merge = True
                                
                        # 2. Merge broken words (e.g. бар’ + єрного)
                        # Detect if prev ends with ' or - and merge without space
                        if prev_text and (prev_text.endswith("’") or prev_text.endswith("'") or prev_text.endswith("-")):
                             # For hyphen, we might want to keep it or remove it depending on if it's a real hyphenated word.
                             # But requirements say "join into one token".
                             # Usually trailing hyphen at EOL is a split word.
                             join_char = ""
                             if prev_text.endswith("-"):
                                 # Basic heuristic: if it's a long word split, remove hyphen. 
                                 # If it's a small word, it might be a real hyphen.
                                 # Spec says "join into one token".
                                 prev_text = prev_text[:-1]
                             
                             current_block.text = prev_text + join_char + curr_text
                        elif should_merge:
                            current_block.text = prev_text + " " + curr_text
                        else:
                            reconstructed_blocks.append(current_block)
                            current_block = DocumentBlock(
                                block_type="paragraph",
                                text=b.text,
                                page_number=b.page_number
                            )
                
                if current_block:
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
            
        # Layer 2: Detect certificate / compliance document subtypes in Zone 7
        annex_zone = next((z for z in structure.zones if z.zone_type == "annex"), None)
        if annex_zone:
            self._detect_annex_subtypes(annex_zone, pages_text, structure)

        p1_text = pages_text[0] if page_count > 0 else ""
        structure.project_title = self._extract_project_title(p1_text)
        structure.application_id = self._extract_registration_number(p1_text)
        structure.call_title = self._extract_call_title(p1_text)
        
        structure.debug_log = debug_log
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
            "kved_code": r"Код\(и\)\s+КВЕД\s*:?\s*([\d\.,\s]+)",
            "legal_address": r"Юридична\s+адреса\s*:?\s*(.+)",
            "postal_address": r"Поштова\s+адреса\s*:?\s*(.+)",
            "physical_address": r"Фактична\s+адреса\s*:?\s*(.+)",
            "phone_number": r"Телефон\s*:?\s*([\+\d\s\-\(\)]+)",
            "email": r"Адреса\s+електронної\s+пошти\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            "website": r"Посилання\s+на\s+веб\s*сторінку\s*:?\s*(https?://[^\s]+|www\.[^\s]+)",
        }

        for field, pattern in regex_map.items():
            match = re.search(pattern, full_zone_text, re.IGNORECASE | re.MULTILINE)
            if match:
                inst_data[field] = match.group(1).strip()

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
        
        pi_data = {"name": structure.pi_name}
        
        full_zone_text = ""
        for block in zone.blocks:
            full_zone_text += block.text + "\n"
        
        # Enforce label-value consistency and ignore line breaks for multi-line values
        labels = {
            "gender": r"Стать",
            "birth_date": r"Дата\s+народження",
            "citizenship": r"Громадянство",
            "orcid": r"ORCID",
            "h_index_scopus": r"Індекс\s+Хірша\s+\(SCOPUS\)",
            "total_publications": r"Загальна\s+кількість\s+публікацій",
            "degree": r"Науковий\s+ступінь",
            "position": r"Посада",
            "phone": r"(?:Мобільний\s+)?телефон",
            "email": r"Електронна\s+пошта"
        }
        
        lines = [l.strip() for l in full_zone_text.split("\n") if l.strip()]
        for field, label_pat in labels.items():
            for i, line in enumerate(lines):
                # Check for label match
                if re.search(r"^" + label_pat + r"\s*:", line, re.IGNORECASE):
                    val = line.split(":", 1)[1].strip()
                    if not val and i + 1 < len(lines):
                        val = lines[i+1]
                    
                    # Sanity check: citizenship shouldn't be a phone number
                    if field == "citizenship" and re.search(r"[\+\d\s\-\(\)]{7,}", val):
                        continue # Skip incorrect mapping
                    if field == "phone" and not re.search(r"[\+\d\s\-\(\)]{7,}", val):
                        continue
                        
                    pi_data[field] = val
                    break
                elif re.search(r"^" + label_pat + r"$", line, re.IGNORECASE):
                     if i + 1 < len(lines):
                        val = lines[i+1]
                        # Sanity checks
                        if field == "citizenship" and re.search(r"[\+\d\s\-\(\)]{7,}", val):
                            continue
                        pi_data[field] = val
                        break
        
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
        
        # We need to collect ALL profile information and then deduplicate
        all_text = "\n".join([b.text for b in zone.blocks])
        
        # Detect individual profile entries
        # Profiles usually start with "Пан" or "Пані" or a name in Zone 6
        profile_splits = re.split(r"(?=(?:Пан|Пані)\s+[А-ЯЄІЇҐ])", all_text)
        
        for entry in profile_splits:
            if not entry.strip(): continue
            
            # Extract name
            name_match = re.search(r"(?:Пан|Пані)\s+([А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+\s*[А-ЯЄІЇҐ]?[а-яєіїґ]*)", entry)
            if not name_match:
                # Fallback for names without honorifics
                name_match = re.search(r"^([А-ЯЄІЇҐ][а-яєіїґ]+\s+[А-ЯЄІЇҐ][а-яєіїґ]+(?:\s+[А-ЯЄІЇҐ][а-яєіїґ]+)?)", entry.strip())
            
            if name_match:
                name = name_match.group(1).strip()
                norm_name = name.lower()
                
                if norm_name not in seen_names and 5 < len(name) < 100:
                    # Extract other fields for this person if possible
                    person_data = {"name": name}
                    
                    labels = {
                        "orcid": r"ORCID",
                        "degree": r"Науковий\s+ступінь",
                        "position": r"Посада"
                    }
                    
                    for field, pat in labels.items():
                        m = re.search(pat + r"\s*:?\s*(.+)", entry, re.IGNORECASE)
                        if m:
                            person_data[field] = m.group(1).strip().split("\n")[0]
                    
                    team.append(person_data)
                    seen_names.add(norm_name)
        
        structure.metadata["team_profiles"] = team

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
        
        structure.metadata["annex_subtypes"] = annexes

        # Update blocks
        for block in zone.blocks:
            if block.block_type == "scan_block":
                block.metadata = {"annex_subtypes": annexes}

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
