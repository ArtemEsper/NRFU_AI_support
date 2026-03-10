from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.session import Base

class SubmittedFile(Base):
    __tablename__ = "submitted_files"

    id = Column(Integer, primary_key=True, index=True)
    package_id = Column(Integer, ForeignKey("application_packages.id"))
    filename = Column(String, nullable=False)
    language = Column(String, nullable=False)  # "uk" or "en"
    content_type = Column(String)
    size = Column(Integer)
    checksum = Column(String, index=True)
    file_path = Column(String)
    page_count = Column(Integer)
    extracted_text = Column(Text)  # Full extracted text
    extracted_text_length = Column(Integer)
    detected_project_registration_number = Column(String)
    detected_call_title = Column(String)
    detected_project_title = Column(String)
    detected_language_note = Column(String)
    status = Column(String, default="pending")  # pending, processing, completed, error
    metadata_json = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    package = relationship("ApplicationPackage", back_populates="files")
    findings = relationship("ReportFinding", back_populates="file")

class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    title_uk = Column(String)
    title_en = Column(String)
    code = Column(String, unique=True, index=True)
    description = Column(Text)
    requires_english_mirror = Column(Boolean, default=False)
    
    # Operational metadata
    applications_received_count = Column(Integer, default=0)
    applications_expected_count = Column(Integer)
    preliminary_evaluation_capacity = Column(Integer)
    formal_check_deadline = Column(DateTime(timezone=True))
    preliminary_evaluation_deadline = Column(DateTime(timezone=True))
    status = Column(String, default="draft")  # draft, active, closed, evaluation, completed
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    items = relationship("ChecklistItem", back_populates="call")
    packages = relationship("ApplicationPackage", back_populates="call")
    documents = relationship("CallDocument", back_populates="call")

class CallDocument(Base):
    __tablename__ = "call_documents"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    document_type = Column(String)  # e.g., "regulation", "template", "manual", "formal_criteria"
    title = Column(String, nullable=False)
    language = Column(String)  # "uk" or "en"
    version = Column(String)
    effective_date = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    is_source_of_truth = Column(Boolean, default=False)
    
    stored_path = Column(String)
    checksum = Column(String)
    extracted_text = Column(Text)
    extracted_text_length = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    call = relationship("Call", back_populates="documents")

class ApplicationPackage(Base):
    __tablename__ = "application_packages"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    project_identifier = Column(String, unique=True, index=True)
    submission_mode = Column(String)  # "online", "manual", etc.
    status = Column(String, default="draft")  # draft, submitted, checking, completed

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    call = relationship("Call", back_populates="packages")
    files = relationship("SubmittedFile", back_populates="package")
    reports = relationship("Report", back_populates="package")

class ChecklistItem(Base):
    __tablename__ = "checklist_items"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"))
    title = Column(String, nullable=False)
    description = Column(Text)
    rule_code = Column(String, index=True)  # Internal code for the rule
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    call = relationship("Call", back_populates="items")
    findings = relationship("ReportFinding", back_populates="checklist_item")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    package_id = Column(Integer, ForeignKey("application_packages.id"))
    call_id = Column(Integer, ForeignKey("calls.id"))
    status = Column(String, default="pending")
    data = Column(JSON, default={})  # Summary or extra analysis result
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    package = relationship("ApplicationPackage", back_populates="reports")
    call = relationship("Call")
    findings = relationship("ReportFinding", back_populates="report")

class ReportFinding(Base):
    __tablename__ = "report_findings"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    file_id = Column(Integer, ForeignKey("submitted_files.id"))
    checklist_item_id = Column(Integer, ForeignKey("checklist_items.id"))
    status = Column(String)  # pass, fail, warning, manual_review
    evidence = Column(Text)  # Extracted text passage or explanation
    page_number = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    report = relationship("Report", back_populates="findings")
    file = relationship("SubmittedFile", back_populates="findings")
    checklist_item = relationship("ChecklistItem", back_populates="findings")
