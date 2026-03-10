import pytest
import os
import io
import shutil
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine, SessionLocal
from app.models.models import Call, ApplicationPackage

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    call = Call(title="Test Call", code="TEST-PDF", requires_english_mirror=False)
    db.add(call)
    db.commit()
    db.refresh(call)
    
    package = ApplicationPackage(call_id=call.id, project_identifier="PDF-TEST-001")
    db.add(package)
    db.commit()
    db.refresh(package)
    
    package_id = package.id
    db.close()
    
    yield package_id
    
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")

def create_dummy_pdf(text="Hello NRFU AI!"):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def test_pdf_upload_and_extraction(setup_db):
    package_id = setup_db
    pdf_content = create_dummy_pdf()
    files = {"file": ("test_real.pdf", pdf_content, "application/pdf")}
    data = {"language": "uk"}
    response = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data)
    
    assert response.status_code == 200
    data_res = response.json()
    assert data_res["filename"] == "test_real.pdf"
    assert data_res["status"] == "completed"
    assert data_res["page_count"] == 1
    assert "Hello NRFU AI!" in data_res["preview"]

def test_non_pdf_rejection(setup_db):
    package_id = setup_db
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    data = {"language": "uk"}
    # We use language='en' here to avoid duplicate upload conflict with previous test which used 'uk'
    response = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data={"language": "en"})
    
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

def test_multi_page_pdf_extraction(setup_db):
    package_id = setup_db
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "Page 1 content")
    doc.new_page().insert_text((50, 50), "Page 2 content")
    pdf_content = doc.write()
    doc.close()
    
    files = {"file": ("multi_page.pdf", pdf_content, "application/pdf")}
    data = {"language": "en"}
    response = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data)
    
    assert response.status_code == 200
    data_res = response.json()
    assert data_res["page_count"] == 2
    assert "Page 1 content" in data_res["preview"]
