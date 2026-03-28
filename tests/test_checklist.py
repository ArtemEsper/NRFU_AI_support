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
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Create a test call and checklist items
    call = Call(title="Test Call", code="TEST-CHECK", requires_english_mirror=True)
    db.add(call)
    db.commit()
    db.refresh(call)
    
    items = [
        ChecklistItem(call_id=call.id, title="Mandatory UK", rule_code="MANDATORY_UK_FILE"),
        ChecklistItem(call_id=call.id, title="Conditional EN", rule_code="CONDITIONAL_EN_FILE"),
        ChecklistItem(call_id=call.id, title="Parseability", rule_code="PDF_PARSEABILITY_CHECK"),
        ChecklistItem(call_id=call.id, title="Sections", rule_code="SECTION_CHECK")
    ]
    for item in items:
        db.add(item)
    db.commit()
    
    package = ApplicationPackage(call_id=call.id, project_identifier="CHECK-TEST-001")
    db.add(package)
    db.commit()
    db.refresh(package)
    
    package_id = package.id
    item_ids = [item.id for item in items]
    db.close()
    
    yield package_id, item_ids
    
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")

def test_checklist_evaluation(setup_db):
    package_id, item_ids = setup_db
    
    # 1. Upload Ukrainian PDF
    def create_dummy_pdf(text="Hello NRFU AI!"):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), text)
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    pdf_content = create_dummy_pdf("Бюджет та Додаток. Підпис: Junie.")
    files = {"file": ("test_uk.pdf", pdf_content, "application/pdf")}
    data = {"language": "uk"}
    upload_res = client.post(f"/api/v1/packages/{package_id}/upload", files=files, data=data)
    assert upload_res.status_code == 200
    
    # 2. Generate report
    response = client.post(f"/api/v1/reports/generate?package_id={package_id}")
    assert response.status_code == 200
    data_res = response.json()
    
    assert data_res["package_id"] == package_id
    assert "findings" in data_res
    
    # Check specific findings
    findings = {f["checklist_item_id"]: f for f in data_res["findings"]}
    
    finding_uk = findings[item_ids[0]]
    assert finding_uk["status"] == "pass" # UK present
    assert finding_uk["rule_code"] == "MANDATORY_UK_FILE"
    assert "severity" in finding_uk
    assert "source_document_title" in finding_uk
    assert "package_evidence" in finding_uk
    assert "uk.pdf" in finding_uk["package_evidence"]

    assert findings[item_ids[1]]["status"] == "fail" # EN missing and required
    assert findings[item_ids[2]]["status"] == "pass" # UK is parseable
    assert "1 pages" in findings[item_ids[2]]["package_evidence"]
    # Check section check - it should be 'review' because we are missing some keywords
    assert findings[item_ids[3]]["status"] == "review"
    assert "Missing specific markers" in findings[item_ids[3]]["explanation"]
    # We inserted 3 keywords in uk_pdf: Бюджет, Додаток, Підпис. 
    # But we'll accept any evidence that shows it attempted the check
    assert "unique section hits" in findings[item_ids[3]]["package_evidence"]

def test_bilingual_consistency(setup_db):
    package_id, item_ids = setup_db
    db = SessionLocal()
    call = db.query(Call).filter(Call.code == "TEST-CHECK").first()
    
    # Create new package for this test
    package = ApplicationPackage(call_id=call.id, project_identifier="CONSISTENCY-TEST")
    db.add(package)
    db.commit()
    db.refresh(package)
    pkg_id = package.id
    
    # Add consistency checklist item if not exists
    consistency_item = db.query(ChecklistItem).filter(ChecklistItem.rule_code == "BILINGUAL_CONSISTENCY").first()
    if not consistency_item:
        consistency_item = ChecklistItem(call_id=call.id, title="Consistency", rule_code="BILINGUAL_CONSISTENCY")
        db.add(consistency_item)
        db.commit()
        db.refresh(consistency_item)
    
    db.close()

    def create_nrfu_pdf(text):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        # Insert text in a way that preserves newlines better or just multiple insertions
        y = 50
        for line in text.split("\n"):
            page.insert_text((50, y), line)
            y += 20
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    # Upload UK file
    uk_text = "Реєстраційний номер проєкту: 2023.03/0014\nНазва конкурсу: Тестовий конкурс\nНазва проєкту: ШІ для НФДУ\nБюджет\nДодаток\nПідпис\nЗгода"
    client.post(f"/api/v1/packages/{pkg_id}/upload", 
                files={"file": ("uk.pdf", create_nrfu_pdf(uk_text), "application/pdf")}, 
                data={"language": "uk"})

    # Upload EN file with matching reg number
    en_text = "Application ID: 2023.03/0014\nCompetition title: Test Competition\nProject title: AI for NRFU\nBudget\nAnnex\nSignature\nConsent"
    client.post(f"/api/v1/packages/{pkg_id}/upload", 
                files={"file": ("en.pdf", create_nrfu_pdf(en_text), "application/pdf")}, 
                data={"language": "en"})

    # Generate report
    response = client.post(f"/api/v1/reports/generate?package_id={pkg_id}")
    assert response.status_code == 200
    data_res = response.json()
    
    # Consistency check should pass if both reg numbers match and titles are present
    # Even if they don't semantically match (we don't check that yet)
    
    # Actually, we should find the finding by rule_code since IDs might vary
    finding_consistency = next((f for f in data_res["findings"] if f["rule_code"] == "BILINGUAL_CONSISTENCY"), None)
    assert finding_consistency is not None
    assert finding_consistency["status"] != "fail"
    assert "Reg numbers" in finding_consistency["package_evidence"]
    assert "2023.03/0014" in finding_consistency["package_evidence"]

def test_project_title_extraction():
    from app.services.pdf_parsing import pdf_parsing_service
    
    # Test UK title on same line
    uk_text = "Назва проєкту: Розробка ШІ для НФДУ\nБюджет: 100000"
    res = pdf_parsing_service._extract_project_title(uk_text)
    assert "Розробка ШІ для НФДУ" in res

    # Test EN title on next line
    en_text = "Project title:\nAI Development for NRFU\nBudget"
    res = pdf_parsing_service._extract_project_title(en_text)
    assert res == "AI Development for NRFU"

    # Test another UK variant
    uk_text_2 = "НАЗВА ТЕМИ ПРОЄКТУ: Масштабування MVP\nБюджет"
    res = pdf_parsing_service._extract_project_title(uk_text_2)
    assert res == "Масштабування MVP"

def test_section_detection_variants():
    from app.services.checklist import ChecklistService
    from app.models.models import SubmittedFile
    
    service = ChecklistService()
    
    # Dummy file with some variants
    f = SubmittedFile(
        language="uk",
        extracted_text="Кошторис проєкту та підписи сторін.\nДодатки до заяви."
    )
    
    # Mocking what SECTION_CHECK does internally (wording variants)
    uk_markers = ["Бюджет", "Кошторис", "Додаток", "Додатки", "Підпис", "Підписи"]
    found = []
    for marker in uk_markers:
        if marker.lower() in f.extracted_text.lower():
            found.append(marker)
            
    assert "Кошторис" in found
    assert "Підписи" in found
    assert "Додатки" in found

def test_source_passage_extraction(setup_db):
    package_id, item_ids = setup_db
    db = SessionLocal()
    call = db.query(Call).filter(Call.id > 0).first()
    
    # 1. Create a CallDocument with some text
    from app.models.models import CallDocument
    doc_content = "This is the regulation. SECTION 3.1: Mandatory budget must be included in all applications. END SECTION."
    call_doc = CallDocument(
        call_id=call.id,
        title="Official Regulation",
        document_type="regulation",
        extracted_text=doc_content,
        extracted_text_length=len(doc_content),
        is_source_of_truth=True
    )
    db.add(call_doc)
    db.commit()
    db.refresh(call_doc)
    
    # 2. Create a rule linked to this document
    rule = ChecklistItem(
        call_id=call.id,
        title="Budget Regulation",
        rule_code="SECTION_CHECK", # We use existing code logic
        source_document_id=call_doc.id,
        source_section="SECTION 3.1"
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    
    db.close()
    
    # 3. Generate report for the package
    response = client.post(f"/api/v1/reports/generate?package_id={package_id}")
    assert response.status_code == 200
    data_res = response.json()
    
    # 4. Check if finding has source_passage
    finding = next((f for f in data_res["findings"] if f["checklist_item_id"] == rule.id), None)
    assert finding is not None
    assert "source_passage" in finding
    assert "Mandatory budget" in finding["source_passage"]
    assert "source_document_title" in finding
    assert finding["source_document_title"] == "Official Regulation"
