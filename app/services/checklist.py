from typing import List, Dict, Any
from sqlalchemy.orm import Session, joinedload
from app.models.models import SubmittedFile, Call, ApplicationPackage, ChecklistItem, ReportFinding, Report, CallDocument
from app.core.logger import logger

class ChecklistService:
    def _extract_source_passage(self, document: CallDocument, section_keyword: str) -> str:
        """
        Extract a short snippet around the section keyword from the source document.
        Now uses structured parsing_result if available to find section content.
        """
        if not document or not document.extracted_text or not section_keyword:
            return None
        
        text = document.extracted_text
        parsing_result = document.parsing_result or {}
        sections = parsing_result.get("sections", [])
        
        # Try to find the section in structured data first
        if sections:
            for i, section in enumerate(sections):
                if section_keyword.lower() in section["title"].lower():
                    # Found the heading. Extract text from this heading to the next one (or a window)
                    start_page = section.get("page", 1)
                    # We don't have exact character offsets for sections yet, 
                    # so we fall back to finding the title in the full text
                    index = text.find(section["title"])
                    if index != -1:
                        # Find the start of the next section if possible
                        next_index = len(text)
                        if i + 1 < len(sections):
                            next_index = text.find(sections[i+1]["title"], index + len(section["title"]))
                            if next_index == -1:
                                next_index = index + 1000 # fallback window
                        
                        # Limit the window size for the snippet
                        end = min(index + 1000, next_index)
                        snippet = text[index:end].replace("\n", " ").strip()
                        if len(snippet) > 500:
                            snippet = snippet[:497] + "..."
                        return f"[Source: {section['title']}] {snippet}"

        # Fallback to simple keyword matching if structured search fails
        index = text.lower().find(section_keyword.lower())
        
        if index == -1:
            return f"Section '{section_keyword}' not matched exactly in source document."
        
        # Extract a window of 300 characters around the keyword
        start = max(0, index - 50)
        end = min(len(text), index + 250)
        snippet = text[start:end].replace("\n", " ").strip()
        
        return f"...{snippet}..."

    async def evaluate_package_rules(self, db: Session, call_id: int, package_id: int) -> List[Dict[str, Any]]:
        """
        Evaluate deterministic rules for a given call and application package.
        Returns a list of finding dictionaries.
        """
        logger.info(f"Evaluating checklist for call {call_id} and package {package_id}")
        
        # 1. Fetch the call and its active checklist items
        call = db.query(Call).filter(Call.id == call_id).first()
        if not call:
            logger.error(f"Call {call_id} not found")
            return []
            
        checklist_items = (
            db.query(ChecklistItem)
            .options(joinedload(ChecklistItem.source_document))
            .filter(ChecklistItem.call_id == call_id, ChecklistItem.is_active == True)
            .all()
        )
        
        # 2. Fetch the package and its files
        package = db.query(ApplicationPackage).filter(ApplicationPackage.id == package_id).first()
        if not package:
            logger.error(f"Package {package_id} not found")
            return []
            
        files = package.files
        findings = []
        
        # Map files by language for easy access
        files_by_lang = {f.language: f for f in files}
        
        for item in checklist_items:
            status = "review"
            explanation = "Manual review required"
            package_evidence = "Requires manual checking of files."
            file_id = None
            page_number = None
            
            # Rule: MANDATORY_UK_FILE
            if item.rule_code == "MANDATORY_UK_FILE":
                uk_file = files_by_lang.get("uk")
                if uk_file:
                    status = "pass"
                    explanation = "Mandatory Ukrainian file is present"
                    package_evidence = f"Found file: {uk_file.filename} (ID: {uk_file.id})"
                    file_id = uk_file.id
                else:
                    status = "fail"
                    explanation = "Mandatory Ukrainian file is missing"
                    package_evidence = "No file with language='uk' was found in the package."
            
            # Rule: CONDITIONAL_EN_FILE
            elif item.rule_code == "CONDITIONAL_EN_FILE":
                en_file = files_by_lang.get("en")
                if call.requires_english_mirror:
                    if en_file:
                        status = "pass"
                        explanation = "English mirror file is present as required by call"
                        package_evidence = f"Found file: {en_file.filename} (ID: {en_file.id})"
                        file_id = en_file.id
                    else:
                        status = "fail"
                        explanation = "English mirror file is missing but required by call"
                        package_evidence = "Call requires English mirror, but no file with language='en' was found."
                else:
                    status = "pass"
                    explanation = "English mirror file is not required by this call"
                    if en_file:
                        explanation += " (optional file provided)"
                        package_evidence = f"Optional file provided: {en_file.filename} (ID: {en_file.id})"
                        file_id = en_file.id
                    else:
                        package_evidence = "Call does not require English mirror; none provided."
            
            # Rule: PDF_VALIDATION (applies to all files in package)
            elif item.rule_code == "PDF_VALIDATION":
                all_pdf = True
                bad_files = []
                for f in files:
                    if not (f.content_type == "application/pdf" or f.filename.lower().endswith(".pdf")):
                        all_pdf = False
                        bad_files.append(f.filename)
                
                if all_pdf and files:
                    status = "pass"
                    explanation = "All submitted files are valid PDFs"
                    package_evidence = f"Validated {len(files)} files: " + ", ".join([f.filename for f in files])
                elif not files:
                    status = "fail"
                    explanation = "No files submitted"
                    package_evidence = "No files were found in this application package."
                else:
                    status = "fail"
                    explanation = "One or more files are not valid PDFs"
                    package_evidence = "Invalid files: " + ", ".join(bad_files)
            
            # Rule: PDF_PARSEABILITY_CHECK
            elif item.rule_code == "PDF_PARSEABILITY_CHECK":
                if not files:
                    status = "fail"
                    explanation = "No files to parse"
                    package_evidence = "Package contains no documents for parsing."
                else:
                    all_parseable = True
                    issues = []
                    details = []
                    for f in files:
                        details.append(f"File {f.filename} ({f.language}): {f.page_count} pages, {f.extracted_text_length} chars.")
                        if f.status != "completed":
                            all_parseable = False
                            issues.append(f"{f.language} file has processing status: {f.status}")
                        elif (f.page_count or 0) <= 0:
                            all_parseable = False
                            issues.append(f"{f.language} file has 0 pages")
                        elif not f.metadata_json or not f.metadata_json.get("preview") or len(f.metadata_json.get("preview")) < 10:
                            all_parseable = False
                            issues.append(f"{f.language} file has empty or too short extracted text")
                    
                    package_evidence = " | ".join(details)
                    if all_parseable:
                        status = "pass"
                        explanation = "All submitted files are parseable and contain text"
                    else:
                        status = "fail"
                        explanation = "Parseability issues: " + "; ".join(issues)
            
            # Rule: SECTION_CHECK
            elif item.rule_code == "SECTION_CHECK":
                if not files:
                    status = "fail"
                    explanation = "No files to check for sections"
                    package_evidence = "No documents provided for section analysis."
                else:
                    # Expanded dictionaries with wording variants
                    uk_markers = [
                        "Реєстраційний номер проєкту", "Реєстраційний номер", "Назва конкурсу", 
                        "Тематичний напрям конкурсу", "Тематичний напрям", "Бюджет", "Кошторис", 
                        "Додаток", "Додатки", "Підпис", "Підписи", "Згода", "Згоди"
                    ]
                    en_markers = [
                        "Application ID", "Registration number", "Competition title", "Subject area", 
                        "Budget", "Annex", "Annexes", "Signature", "Signatures", "Consent", "Consents"
                    ]
                    
                    found_sections = []
                    missing_sections = []
                    
                    # For cross-language fallback matching
                    all_markers = uk_markers + en_markers
                    
                    for f in files:
                        # Primary markers for the file language
                        primary_markers = uk_markers if f.language == "uk" else en_markers
                        # Use full extracted text
                        content = f.extracted_text or ""
                        
                        for marker in primary_markers:
                            if marker.lower() in content.lower():
                                found_sections.append(f"{f.language}:{marker}")
                            else:
                                # Fallback: check if other language markers match (mixed language docs)
                                fallback_match = False
                                for other_marker in all_markers:
                                    if other_marker.lower() in content.lower():
                                        # If it's a semantic equivalent, consider it a partial find
                                        # For MVP, we just record it as a found section to reduce false flags
                                        if (f.language == "uk" and other_marker in en_markers) or \
                                           (f.language == "en" and other_marker in uk_markers):
                                            found_sections.append(f"{f.language}:{other_marker}(fallback)")
                                            fallback_match = True
                                            break
                                if not fallback_match:
                                    missing_sections.append(f"{f.language}:{marker}")
                    
                    package_evidence = f"Found {len(found_sections)} unique section hits across {len(files)} files."
                    if not missing_sections:
                        status = "pass"
                        explanation = f"All primary NRFU sections detected."
                    else:
                        status = "review"
                        # Filter out variants of the same concept to keep explanation clean
                        unique_missing = sorted(list(set(missing_sections)))
                        explanation = f"Missing specific markers: {', '.join(unique_missing)}. Found: {', '.join(found_sections[:5])}..."
            
            # Rule: BILINGUAL_CONSISTENCY
            elif item.rule_code == "BILINGUAL_CONSISTENCY":
                uk_file = files_by_lang.get("uk")
                en_file = files_by_lang.get("en")
                
                if uk_file and en_file:
                    issues = []
                    evidence_parts = []
                    
                    # 1. Registration Number Consistency
                    uk_reg = uk_file.detected_project_registration_number
                    en_reg = en_file.detected_project_registration_number
                    evidence_parts.append(f"Reg numbers: UK='{uk_reg or 'N/A'}', EN='{en_reg or 'N/A'}'")
                    if uk_reg and en_reg:
                        if uk_reg != en_reg:
                            issues.append(f"Registration number mismatch: UK='{uk_reg}', EN='{en_reg}'")
                    elif not uk_reg or not en_reg:
                        issues.append("Registration number not detected in one or both files")
                        
                    # 2. Project Title Presence
                    evidence_parts.append(f"UK Title: '{uk_file.detected_project_title or 'N/A'}'")
                    evidence_parts.append(f"EN Title: '{en_file.detected_project_title or 'N/A'}'")
                    if not uk_file.detected_project_title:
                        issues.append("Project title not detected in Ukrainian file")
                    if not en_file.detected_project_title:
                        issues.append("Project title not detected in English file")
                        
                    package_evidence = " | ".join(evidence_parts)
                    
                    if not issues:
                        status = "pass"
                        explanation = "Bilingual consistency check passed (registration number, titles)"
                    else:
                        status = "review"
                        explanation = "Consistency review required: " + "; ".join(issues)
                elif call.requires_english_mirror:
                    status = "fail"
                    explanation = "Cannot perform consistency check: missing required files"
                    package_evidence = "Cannot compare UK and EN files because one or both are missing."
                else:
                    status = "pass"
                    explanation = "Bilingual consistency check skipped (single language submission)"
                    package_evidence = "Package only contains one language; no consistency check needed."
            
            # Extract source passage if source document is available
            source_passage = None
            if item.source_document_id and item.source_section:
                source_passage = self._extract_source_passage(item.source_document, item.source_section)
                if not source_passage or "not matched exactly" in source_passage:
                    explanation += " | Note: Source section not found in document."
            elif not item.source_document_id:
                explanation += " | Note: This rule is currently ungrounded."

            findings.append({
                "checklist_item_id": item.id,
                "status": status,
                "explanation": explanation,
                "package_evidence": package_evidence,
                "file_id": file_id,
                "page_number": page_number,
                # Include rule metadata for the report
                "rule_code": item.rule_code,
                "rule_text": item.title,
                "severity": item.severity,
                "source_document_id": item.source_document_id,
                "source_document_title": item.source_document.title if item.source_document else None,
                "source_section": item.source_section,
                "source_passage": source_passage
            })
            
        return findings

checklist_service = ChecklistService()
