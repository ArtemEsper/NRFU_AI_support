import pytest
import os
import shutil
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine, SessionLocal
from app.models.models import Call, ChecklistItem, CallDocument

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")

@pytest.fixture(scope="function")
def db(setup_db):
    session = SessionLocal()
    yield session
    session.close()

def test_create_rule_with_grounding(db):
    # 1. Create a Call
    call = Call(title="Test Call", code="CALL-RULE-TEST")
    db.add(call)
    db.commit()
    db.refresh(call)

    # 2. Create a Call Document for this call
    doc = CallDocument(
        call_id=call.id,
        title="Source Regulation",
        document_type="regulation",
        extracted_text="Section 1.1: Mandatory UK file is required.",
        extracted_text_length=42
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 3. Create a rule linked to this document via API
    rule_data = {
        "title": "Grounded Rule",
        "rule_code": "MANDATORY_UK_FILE",
        "severity": "critical",
        "source_document_id": doc.id,
        "source_section": "Section 1.1",
        "is_active": True
    }
    response = client.post(f"/api/v1/calls/{call.id}/rules", json=rule_data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["source_document_id"] == doc.id
    assert res_data["source_section"] == "Section 1.1"

def test_update_rule_with_grounding(db):
    # 1. Create a Call
    call = Call(title="Test Call", code="CALL-RULE-UPDATE")
    db.add(call)
    db.commit()
    db.refresh(call)

    # 2. Create a Call Document for this call
    doc = CallDocument(
        call_id=call.id,
        title="Source Regulation",
        document_type="regulation",
        extracted_text="Section 2.2: Some rule content.",
        extracted_text_length=30
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 3. Create an ungrounded rule
    rule = ChecklistItem(
        call_id=call.id,
        title="Ungrounded Rule",
        rule_code="SECTION_CHECK",
        severity="major"
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)

    # 4. Update the rule to be grounded via API
    update_data = {
        "source_document_id": doc.id,
        "source_section": "Section 2.2",
        "severity": "critical"
    }
    response = client.put(f"/api/v1/rules/{rule.id}", json=update_data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["source_document_id"] == doc.id
    assert res_data["source_section"] == "Section 2.2"
    assert res_data["severity"] == "critical"
    assert res_data["title"] == "Ungrounded Rule" # Preserved

def test_reject_rule_update_wrong_call_document(db):
    # 1. Create Call A and its document
    call_a = Call(title="Call A", code="CALL-A")
    db.add(call_a)
    db.commit()
    db.refresh(call_a)
    
    doc_a = CallDocument(call_id=call_a.id, title="Doc A", document_type="regulation")
    db.add(doc_a)
    db.commit()
    db.refresh(doc_a)

    # 2. Create Call B and a rule for Call B
    call_b = Call(title="Call B", code="CALL-B")
    db.add(call_b)
    db.commit()
    db.refresh(call_b)
    
    rule_b = ChecklistItem(call_id=call_b.id, title="Rule B", rule_code="MANDATORY_UK_FILE")
    db.add(rule_b)
    db.commit()
    db.refresh(rule_b)

    # 3. Try to link Rule B to Doc A (wrong call)
    update_data = {
        "source_document_id": doc_a.id
    }
    response = client.put(f"/api/v1/rules/{rule_b.id}", json=update_data)
    assert response.status_code == 400
    assert "Source document not found or belongs to another call" in response.json()["detail"]

def test_grounded_report_generation(db):
    # 1. Setup call, doc, grounded rule
    call = Call(title="Call Grounded", code="CALL-GR")
    db.add(call)
    db.commit()
    db.refresh(call)
    
    doc = CallDocument(
        call_id=call.id, 
        title="Authoritative Doc", 
        document_type="regulation",
        extracted_text="Section X: Requirement details.",
        is_source_of_truth=True
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    rule = ChecklistItem(
        call_id=call.id,
        title="Check X",
        rule_code="MANDATORY_UK_FILE",
        source_document_id=doc.id,
        source_section="Section X"
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    
    # 2. Create package
    from app.models.models import ApplicationPackage
    package = ApplicationPackage(call_id=call.id, project_identifier="PKG-GR-001")
    db.add(package)
    db.commit()
    db.refresh(package)
    
    # 3. Generate report (no file uploaded, should fail but show grounding)
    response = client.post(f"/api/v1/reports/generate?package_id={package.id}")
    assert response.status_code == 200
    data = response.json()
    
    finding = next(f for f in data["findings"] if f["checklist_item_id"] == rule.id)
    assert finding["status"] == "fail"
    assert finding["source_document_title"] == "Authoritative Doc"
    assert finding["source_section"] == "Section X"
    assert "Requirement details" in finding["source_passage"]
