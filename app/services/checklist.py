from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.models import SubmittedFile, Call, ApplicationPackage, ChecklistItem, ReportFinding, Report
from app.core.logger import logger

class ChecklistService:
    async def evaluate_package_rules(self, db: Session, call_id: int, package_id: int) -> List[Dict[str, Any]]:
        """
        Evaluate deterministic rules for a given call and application package.
        Returns a list of finding dictionaries.
        """
        logger.info(f"Evaluating checklist for call {call_id} and package {package_id}")
        
        # 1. Fetch the call and its checklist items
        call = db.query(Call).filter(Call.id == call_id).first()
        if not call:
            logger.error(f"Call {call_id} not found")
            return []
            
        checklist_items = db.query(ChecklistItem).filter(ChecklistItem.call_id == call_id).all()
        
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
            file_id = None
            page_number = None
            
            # Rule: MANDATORY_UK_FILE
            if item.rule_code == "MANDATORY_UK_FILE":
                uk_file = files_by_lang.get("uk")
                if uk_file:
                    status = "pass"
                    explanation = "Mandatory Ukrainian file is present"
                    file_id = uk_file.id
                else:
                    status = "fail"
                    explanation = "Mandatory Ukrainian file is missing"
            
            # Rule: CONDITIONAL_EN_FILE
            elif item.rule_code == "CONDITIONAL_EN_FILE":
                en_file = files_by_lang.get("en")
                if call.requires_english_mirror:
                    if en_file:
                        status = "pass"
                        explanation = "English mirror file is present as required by call"
                        file_id = en_file.id
                    else:
                        status = "fail"
                        explanation = "English mirror file is missing but required by call"
                else:
                    status = "pass"
                    explanation = "English mirror file is not required by this call"
                    if en_file:
                        explanation += " (optional file provided)"
                        file_id = en_file.id
            
            # Rule: PDF_VALIDATION (applies to all files in package)
            elif item.rule_code == "PDF_VALIDATION":
                all_pdf = True
                for f in files:
                    if not (f.content_type == "application/pdf" or f.filename.lower().endswith(".pdf")):
                        all_pdf = False
                        break
                if all_pdf and files:
                    status = "pass"
                    explanation = "All submitted files are valid PDFs"
                elif not files:
                    status = "fail"
                    explanation = "No files submitted"
                else:
                    status = "fail"
                    explanation = "One or more files are not valid PDFs"
            
            # Rule: PDF_PARSEABILITY_CHECK
            elif item.rule_code == "PDF_PARSEABILITY_CHECK":
                if not files:
                    status = "fail"
                    explanation = "No files to parse"
                else:
                    all_parseable = True
                    issues = []
                    for f in files:
                        if f.status != "completed":
                            all_parseable = False
                            issues.append(f"{f.language} file has processing status: {f.status}")
                        elif (f.page_count or 0) <= 0:
                            all_parseable = False
                            issues.append(f"{f.language} file has 0 pages")
                        elif not f.metadata_json or not f.metadata_json.get("preview") or len(f.metadata_json.get("preview")) < 10:
                            all_parseable = False
                            issues.append(f"{f.language} file has empty or too short extracted text")
                    
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
                else:
                    uk_markers = [
                        "Реєстраційний номер проєкту", "Назва конкурсу", 
                        "Тематичний напрям конкурсу", "Бюджет", "Кошторис", 
                        "Додаток", "Підпис", "Згода"
                    ]
                    en_markers = [
                        "Application ID", "Competition title", "Subject area", 
                        "Budget", "Annex", "Signature", "Consent"
                    ]
                    
                    found_sections = []
                    missing_sections = []
                    
                    for f in files:
                        markers = uk_markers if f.language == "uk" else en_markers
                        # Use full extracted text for section check
                        content = f.extracted_text or ""
                        for marker in markers:
                            if marker.lower() in content.lower():
                                found_sections.append(f"{f.language}:{marker}")
                            else:
                                missing_sections.append(f"{f.language}:{marker}")
                    
                    if not missing_sections:
                        status = "pass"
                        explanation = f"All expected NRFU sections found: {', '.join(found_sections)}"
                    else:
                        status = "review"
                        explanation = f"Some sections not detected in full text: {', '.join(missing_sections)}. Found: {', '.join(found_sections)}"

            # Rule: BILINGUAL_CONSISTENCY
            elif item.rule_code == "BILINGUAL_CONSISTENCY":
                uk_file = files_by_lang.get("uk")
                en_file = files_by_lang.get("en")
                
                if uk_file and en_file:
                    issues = []
                    
                    # 1. Registration Number Consistency
                    uk_reg = uk_file.detected_project_registration_number
                    en_reg = en_file.detected_project_registration_number
                    if uk_reg and en_reg:
                        if uk_reg != en_reg:
                            issues.append(f"Registration number mismatch: UK='{uk_reg}', EN='{en_reg}'")
                    elif not uk_reg or not en_reg:
                        issues.append("Registration number not detected in one or both files")
                        
                    # 2. Project Title Presence
                    if not uk_file.detected_project_title:
                        issues.append("Project title not detected in Ukrainian file")
                    if not en_file.detected_project_title:
                        issues.append("Project title not detected in English file")
                        
                    # 3. Call Title Presence
                    if not uk_file.detected_call_title:
                        issues.append("Call title not detected in Ukrainian file")
                    if not en_file.detected_call_title:
                        issues.append("Call title not detected in English file")
                        
                    if not issues:
                        status = "pass"
                        explanation = "Bilingual consistency check passed (registration number, titles)"
                    else:
                        status = "review"
                        explanation = "Consistency review required: " + "; ".join(issues)
                elif call.requires_english_mirror:
                    status = "fail"
                    explanation = "Cannot perform consistency check: missing required files"
                else:
                    status = "pass"
                    explanation = "Bilingual consistency check skipped (single language submission)"
            
            findings.append({
                "checklist_item_id": item.id,
                "status": status,
                "explanation": explanation,
                "file_id": file_id,
                "page_number": page_number
            })
            
        return findings

checklist_service = ChecklistService()
