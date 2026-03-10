from typing import List, Optional
from pydantic import BaseModel
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
    title: str
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
    pass

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

class ChecklistItemSchema(ChecklistItemBase):
    id: int
    call_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    package_id: int
    call_id: Optional[int] = None
    status: str

class ReportFindingSchema(BaseModel):
    id: int
    report_id: int
    file_id: int
    checklist_item_id: int
    status: str
    evidence: Optional[str] = None
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
