from typing import List, Optional, Any
from pydantic import BaseModel, model_validator
from datetime import datetime

class SubmittedFileBase(BaseModel):
    filename: str
    language: str
    content_type: Optional[str] = None
    size: Optional[int] = None

class SubmittedFileCreate(SubmittedFileBase):
    checksum: str

class SubmittedFileSchema(SubmittedFileBase):
    id: int
    package_id: int
    status: str
    page_count: Optional[int] = None
    extracted_text_length: Optional[int] = None
    detected_project_registration_number: Optional[str] = None
    detected_call_title: Optional[str] = None
    detected_project_title: Optional[str] = None
    detected_language_note: Optional[str] = None
    metadata_json: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True

class CallBase(BaseModel):
    title: Optional[str] = None
    title_uk: Optional[str] = None
    title_en: Optional[str] = None
    code: str
    description: Optional[str] = None
    requires_english_mirror: bool = False
    applications_received_count: Optional[int] = 0
    applications_expected_count: Optional[int] = None
    preliminary_evaluation_capacity: Optional[int] = None
    formal_check_deadline: Optional[datetime] = None
    preliminary_evaluation_deadline: Optional[datetime] = None
    status: Optional[str] = "draft"

class CallCreate(CallBase):
    @model_validator(mode='before')
    @classmethod
    def validate_titles(cls, data: Any) -> Any:
        if isinstance(data, dict):
            title = data.get("title")
            title_uk = data.get("title_uk")
            title_en = data.get("title_en")
            
            # If only legacy 'title' is provided, map it to 'title_en' 
            # and 'title_uk' if they are missing.
            if title and not title_uk:
                data["title_uk"] = title
            if title and not title_en:
                data["title_en"] = title
                
            # Ensure at least one of title_uk or title_en is present
            if not data.get("title_uk") and not data.get("title_en"):
                raise ValueError("At least one of 'title_uk' or 'title_en' must be provided.")
            
            # For the database 'title' field (legacy), ensure it is populated
            if not data.get("title"):
                data["title"] = data.get("title_uk") or data.get("title_en")
                
        return data

class CallDocumentBase(BaseModel):
    document_type: str
    title: str
    language: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[datetime] = None
    is_active: bool = True
    is_source_of_truth: bool = False

class CallDocumentCreate(CallDocumentBase):
    pass

class CallDocumentSchema(CallDocumentBase):
    id: int
    call_id: int
    stored_path: Optional[str] = None
    checksum: Optional[str] = None
    extracted_text: Optional[str] = None
    extracted_text_length: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CallSchema(CallBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    documents: List[CallDocumentSchema] = []

    class Config:
        from_attributes = True

class ApplicationPackageBase(BaseModel):
    call_id: int
    project_identifier: str
    submission_mode: Optional[str] = "online"

class ApplicationPackageCreate(ApplicationPackageBase):
    pass

class ApplicationPackageSchema(ApplicationPackageBase):
    id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    files: List[SubmittedFileSchema] = []

    class Config:
        from_attributes = True

class ChecklistItemBase(BaseModel):
    title: str
    description: Optional[str] = None
    rule_code: str
    severity: Optional[str] = "critical"
    source_document_id: Optional[int] = None
    source_section: Optional[str] = None
    is_active: bool = True

class ChecklistItemCreate(ChecklistItemBase):
    pass

class ChecklistItemUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    rule_code: Optional[str] = None
    severity: Optional[str] = None
    source_document_id: Optional[int] = None
    source_section: Optional[str] = None
    is_active: Optional[bool] = None

class ChecklistItemSchema(ChecklistItemBase):
    id: int
    call_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    source_document: Optional[CallDocumentSchema] = None

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    package_id: int
    call_id: Optional[int] = None
    status: str

class ReportFindingSchema(BaseModel):
    id: int
    report_id: int
    file_id: Optional[int] = None
    checklist_item_id: int
    status: str
    evidence: Optional[str] = None
    package_evidence: Optional[str] = None
    source_document_title: Optional[str] = None
    source_section: Optional[str] = None
    source_passage: Optional[str] = None
    page_number: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ReportSchema(ReportBase):
    id: int
    data: dict
    created_at: datetime
    findings: List[ReportFindingSchema] = []

    class Config:
        from_attributes = True
