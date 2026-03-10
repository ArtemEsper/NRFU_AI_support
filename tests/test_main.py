import pytest
import os
import shutil
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine, SessionLocal
from app.models.models import Call, ChecklistItem, ApplicationPackage

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Use Alembic or just create_all for tests
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Service is healthy"}

def test_full_flow():
    db = SessionLocal()
    # 1. Create a Call
    call = Call(title="Default Call", code="CALL-001", requires_english_mirror=True)
    db.add(call)
    db.commit()
    db.refresh(call)
    
    # 2. Add Checklist Items
    items = [
        ChecklistItem(call_id=call.id, title="Mandatory UK", rule_code="MANDATORY_UK_FILE"),
        ChecklistItem(call_id=call.id, title="Conditional EN", rule_code="CONDITIONAL_EN_FILE"),
        ChecklistItem(call_id=call.id, title="Valid PDFs", rule_code="PDF_VALIDATION")
    ]
    for item in items:
        db.add(item)
    db.commit()
    
    # Extract IDs before closing session
    item_ids = [item.id for item in items]
    call_id = call.id
    db.close()

    # 3. Create Application Package
    package_data = {
        "call_id": call_id,
        "project_identifier": "PROJ-2024-001",
        "submission_mode": "online"
    }
    response = client.post("/api/v1/packages", json=package_data)
    assert response.status_code == 200
    package_id = response.json()["id"]

    # 3.1 Try duplicate upload
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
    files = {"file": ("merged_uk.pdf", pdf_content, "application/pdf")}
    data = {"language": "uk"}
    client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data) # First upload
    
    dup_res = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data) # Duplicate
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"]

    # 4. Upload Ukrainian File (already done in 3.1)

    # 5. Generate Report (Should show missing EN file)
    report_res = client.post(f"/api/v1/reports/generate?package_id={package_id}")
    assert report_res.status_code == 200
    findings = report_res.json()["findings"]
    
    # Check findings
    uk_finding = next(f for f in findings if f["checklist_item_id"] == item_ids[0])
    en_finding = next(f for f in findings if f["checklist_item_id"] == item_ids[1])
    
    assert uk_finding["status"] == "pass"
    assert en_finding["status"] == "fail" # because call requires EN mirror

    # 6. Upload English File
    files = {"file": ("merged_en.pdf", pdf_content, "application/pdf")}
    data = {"language": "en"}
    upload_res = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data)
    assert upload_res.status_code == 200

    # 7. Generate Report again
    report_res = client.post(f"/api/v1/reports/generate?package_id={package_id}")
    assert report_res.status_code == 200
    findings = report_res.json()["findings"]
    en_finding = next(f for f in findings if f["checklist_item_id"] == item_ids[1])
    assert en_finding["status"] == "pass"
