from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.models import SubmittedFile, Report, Call, ChecklistItem, ReportFinding, ApplicationPackage, CallDocument
from app.services.storage import storage_service
from app.services.pdf_parsing import pdf_parsing_service
from app.services.checklist import checklist_service
from app.schemas.schemas import (
    ApplicationPackageCreate, ApplicationPackageSchema, CallDocumentSchema, 
    CallSchema, ChecklistItemSchema, ChecklistItemBase, CallCreate, 
    ChecklistItemUpdate, ReportSchema
)
import hashlib
import io

router = APIRouter()

@router.post("/packages", response_model=ApplicationPackageSchema)
async def create_package(package_in: ApplicationPackageCreate, db: Session = Depends(get_db)):
    # Check if call exists
    call = db.query(Call).filter(Call.id == package_in.call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Check if project identifier already exists
    existing = db.query(ApplicationPackage).filter(ApplicationPackage.project_identifier == package_in.project_identifier).first()
    if existing:
        raise HTTPException(status_code=400, detail="Project identifier already exists")
    
    db_package = ApplicationPackage(
        call_id=package_in.call_id,
        project_identifier=package_in.project_identifier,
        submission_mode=package_in.submission_mode
    )
    db.add(db_package)
    db.commit()
    db.refresh(db_package)
    return db_package

@router.post("/packages/{package_id}/upload")
async def upload_document(
    package_id: int,
    language: str = Form(..., description="'uk' or 'en'"),
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # Validate language
    if language not in ["uk", "en"]:
        raise HTTPException(status_code=400, detail="Language must be 'uk' or 'en'")
        
    # Check if package exists
    package = db.query(ApplicationPackage).filter(ApplicationPackage.id == package_id).first()
    if not package:
        raise HTTPException(status_code=404, detail="Application package not found")

    # Enforce one file per language per package
    existing_file = db.query(SubmittedFile).filter(
        SubmittedFile.package_id == package_id,
        SubmittedFile.language == language
    ).first()
    if existing_file:
        raise HTTPException(
            status_code=400, 
            detail=f"A {language} file already exists for this package. Duplicate uploads are not allowed."
        )

    # Validate PDF
    if not file.filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported at this stage.")

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    
    # Save file to storage
    file_path = await storage_service.save_file(io.BytesIO(content), file.filename)
    
    # Parse PDF
    try:
        parsing_results = await pdf_parsing_service.parse_pdf(content)
    except Exception as e:
        db_file = SubmittedFile(
            package_id=package_id,
            filename=file.filename,
            language=language,
            content_type=file.content_type,
            size=len(content),
            checksum=checksum,
            file_path=file_path,
            status="error",
            metadata_json={"error": str(e)}
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return {"id": db_file.id, "filename": db_file.filename, "status": "error", "detail": "Parsing failed"}

    # Store metadata in DB
    db_file = SubmittedFile(
        package_id=package_id,
        filename=file.filename,
        language=language,
        content_type=file.content_type,
        size=len(content),
        checksum=checksum,
        file_path=file_path,
        page_count=parsing_results["page_count"],
        extracted_text=parsing_results["full_text"],
        extracted_text_length=parsing_results["extracted_text_length"],
        detected_project_registration_number=parsing_results["detected_project_registration_number"],
        detected_call_title=parsing_results["detected_call_title"],
        detected_project_title=parsing_results["detected_project_title"],
        detected_language_note=parsing_results["detected_language_note"],
        status="completed",
        metadata_json={
            "pdf_metadata": parsing_results["metadata"],
            "preview": parsing_results["preview"]
        }
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    return {
        "id": db_file.id, 
        "package_id": package_id,
        "filename": db_file.filename, 
        "language": language,
        "status": db_file.status, 
        "page_count": db_file.page_count,
        "extracted_text_length": db_file.extracted_text_length,
        "detected_project_registration_number": db_file.detected_project_registration_number,
        "preview": parsing_results["preview"]
    }

@router.post("/reports/generate")
async def generate_report(package_id: int, db: Session = Depends(get_db)):
    # Check if package exists
    package = db.query(ApplicationPackage).filter(ApplicationPackage.id == package_id).first()
    if not package:
        raise HTTPException(status_code=404, detail="Application package not found")
    
    call = package.call
    if not call:
        raise HTTPException(status_code=404, detail="Call associated with package not found")
    
    # Evaluate deterministic checklist
    findings_data = await checklist_service.evaluate_package_rules(db, call.id, package_id)
    
    # Calculate overall status
    statuses = [f["status"] for f in findings_data]
    if "fail" in statuses:
        overall_status = "fail"
    elif "review" in statuses:
        overall_status = "review"
    else:
        overall_status = "pass"
    
    # Generate report record
    db_report = Report(
        package_id=package_id,
        call_id=call.id,
        status=overall_status,
        data={
            "package_identifier": package.project_identifier,
            "call_title": call.title,
            "call_code": call.code,
            "file_count": len(package.files),
            "generated_at": str(func.now())
        }
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Save findings to database
    for finding in findings_data:
        db_finding = ReportFinding(
            report_id=db_report.id,
            file_id=finding.get("file_id"),
            checklist_item_id=finding["checklist_item_id"],
            status=finding["status"],
            evidence=finding["explanation"],
            package_evidence=finding.get("package_evidence"),
            source_document_title=finding.get("source_document_title"),
            source_section=finding.get("source_section"),
            source_passage=finding.get("source_passage"),
            page_number=finding.get("page_number")
        )
        db.add(db_finding)
    
    db.commit()
    
    return {
        "report_id": db_report.id,
        "package_id": package_id,
        "project_identifier": package.project_identifier,
        "call_code": call.code,
        "overall_status": overall_status,
        "files_summary": [
            {
                "id": f.id, 
                "filename": f.filename, 
                "language": f.language, 
                "pages": f.page_count,
                "text_length": f.extracted_text_length,
                "detected_reg_number": f.detected_project_registration_number,
                "detected_project_title": f.detected_project_title
            }
            for f in package.files
        ],
        "findings": findings_data
    }

@router.get("/calls/{call_id}", response_model=CallSchema)
async def get_call(call_id: int, db: Session = Depends(get_db)):
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call

@router.post("/calls", response_model=CallSchema)
async def create_call(call_in: CallCreate, db: Session = Depends(get_db)):
    # Check if code already exists
    existing = db.query(Call).filter(Call.code == call_in.code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Call code already exists")
    
    db_call = Call(**call_in.model_dump())
    db.add(db_call)
    db.commit()
    db.refresh(db_call)
    return db_call

@router.post("/calls/{call_id}/documents", response_model=CallDocumentSchema)
async def upload_call_document(
    call_id: int,
    title: str = Form(...),
    document_type: str = Form(..., description="regulation, template, manual, formal_criteria"),
    language: str = Form(None, description="'uk' or 'en'"),
    version: str = Form(None),
    is_source_of_truth: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Check if call exists
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    # Validate PDF
    if not file.filename.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    checksum = hashlib.sha256(content).hexdigest()
    
    # Save file to storage
    file_path = await storage_service.save_file(io.BytesIO(content), file.filename)
    
    # Parse PDF
    parsing_results = await pdf_parsing_service.parse_pdf(content)
    
    db_doc = CallDocument(
        call_id=call_id,
        title=title,
        document_type=document_type,
        language=language,
        version=version,
        is_source_of_truth=is_source_of_truth,
        stored_path=file_path,
        checksum=checksum,
        extracted_text=parsing_results["full_text"],
        extracted_text_length=parsing_results["extracted_text_length"]
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

@router.get("/calls/{call_id}/rules", response_model=List[ChecklistItemSchema])
async def list_call_rules(call_id: int, db: Session = Depends(get_db)):
    # Check if call exists
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # List active rules
    rules = db.query(ChecklistItem).filter(ChecklistItem.call_id == call_id, ChecklistItem.is_active == True).all()
    return rules

@router.post("/calls/{call_id}/rules", response_model=ChecklistItemSchema)
async def create_call_rule(call_id: int, rule_in: ChecklistItemBase, db: Session = Depends(get_db)):
    # Check if call exists
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Validate source_document_id if provided
    if rule_in.source_document_id:
        source_doc = db.query(CallDocument).filter(
            CallDocument.id == rule_in.source_document_id,
            CallDocument.call_id == call_id
        ).first()
        if not source_doc:
            raise HTTPException(
                status_code=400, 
                detail="Source document not found or belongs to another call."
            )

    # Create rule
    db_rule = ChecklistItem(
        call_id=call_id,
        **rule_in.model_dump()
    )
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule

@router.put("/rules/{rule_id}", response_model=ChecklistItemSchema)
async def update_call_rule(rule_id: int, rule_in: ChecklistItemUpdate, db: Session = Depends(get_db)):
    db_rule = db.query(ChecklistItem).filter(ChecklistItem.id == rule_id).first()
    if not db_rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    update_data = rule_in.model_dump(exclude_unset=True)
    
    # Validate source_document_id if provided
    if "source_document_id" in update_data and update_data["source_document_id"] is not None:
        source_doc = db.query(CallDocument).filter(
            CallDocument.id == update_data["source_document_id"],
            CallDocument.call_id == db_rule.call_id
        ).first()
        if not source_doc:
            raise HTTPException(
                status_code=400, 
                detail="Source document not found or belongs to another call."
            )

    for key, value in update_data.items():
        setattr(db_rule, key, value)
    
    db.commit()
    db.refresh(db_rule)
    return db_rule

@router.delete("/rules/{rule_id}")
async def delete_call_rule(rule_id: int, db: Session = Depends(get_db)):
    db_rule = db.query(ChecklistItem).filter(ChecklistItem.id == rule_id).first()
    if not db_rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    db.delete(db_rule)
    db.commit()
    return {"detail": "Rule deleted"}

@router.get("/packages", response_model=List[ApplicationPackageSchema])
async def list_packages(db: Session = Depends(get_db)):
    return db.query(ApplicationPackage).all()

@router.get("/calls/{call_id}/documents", response_model=List[CallDocumentSchema])
async def list_call_documents(call_id: int, db: Session = Depends(get_db)):
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call.documents
