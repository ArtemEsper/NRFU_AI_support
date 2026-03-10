import asyncio
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal, engine, Base
from app.models.models import Call, ChecklistItem, CallDocument
from datetime import datetime, timedelta

def seed_data():
    db = SessionLocal()
    try:
        # 1. Create a Ukrainian-only call
        uk_only_call = db.query(Call).filter(Call.code == "CALL-UK-2024").first()
        if not uk_only_call:
            uk_only_call = Call(
                title="Ukrainian-only Funding Call 2024",
                title_uk="Конкурс НФДУ 2024 (Тільки українською)",
                title_en="NRFU Funding Call 2024 (Ukrainian only)",
                code="CALL-UK-2024",
                description="This call requires only a Ukrainian merged PDF submission.",
                requires_english_mirror=False,
                applications_expected_count=100,
                status="active",
                formal_check_deadline=datetime.now() + timedelta(days=30)
            )
            db.add(uk_only_call)
            db.flush()  # To get the ID
            print(f"Created call: {uk_only_call.title}")
        else:
            print(f"Call {uk_only_call.code} already exists")

        # 2. Create a call requiring English mirror
        bilingual_call = db.query(Call).filter(Call.code == "CALL-BI-2024").first()
        if not bilingual_call:
            bilingual_call = Call(
                title="International Funding Call 2024 (Bilingual)",
                title_uk="Міжнародний конкурс НФДУ 2024 (Двомовний)",
                title_en="International NRFU Funding Call 2024 (Bilingual)",
                code="CALL-BI-2024",
                description="This call requires both Ukrainian and English merged PDF submissions.",
                requires_english_mirror=True,
                applications_expected_count=50,
                status="active",
                formal_check_deadline=datetime.now() + timedelta(days=45)
            )
            db.add(bilingual_call)
            db.flush()  # To get the ID
            print(f"Created call: {bilingual_call.title}")
        else:
            print(f"Call {bilingual_call.code} already exists")

        # 3. Define default checklist rules
        default_rules = [
            ("MANDATORY_UK_FILE", "Mandatory Ukrainian File", "Check if the mandatory Ukrainian merged PDF is present."),
            ("CONDITIONAL_EN_FILE", "English Mirror Requirement", "Check if the English mirror PDF is present if required by the call."),
            ("PDF_VALIDATION", "PDF Format Validation", "Verify that all uploaded files are valid PDFs."),
            ("PDF_PARSEABILITY_CHECK", "PDF Parseability Check", "Verify that PDFs are not empty and text can be extracted."),
            ("SECTION_CHECK", "NRFU Section Presence", "Heuristic check for mandatory sections like Budget, Annex, etc."),
            ("BILINGUAL_CONSISTENCY", "Bilingual Metadata Consistency", "Verify that registration numbers and titles match between UK and EN files.")
        ]

        # 4. Attach rules to calls
        for call in [uk_only_call, bilingual_call]:
            for rule_code, title, desc in default_rules:
                existing_item = db.query(ChecklistItem).filter(
                    ChecklistItem.call_id == call.id,
                    ChecklistItem.rule_code == rule_code
                ).first()
                
                if not existing_item:
                    item = ChecklistItem(
                        call_id=call.id,
                        title=title,
                        description=desc,
                        rule_code=rule_code
                    )
                    db.add(item)
                    print(f"Added rule {rule_code} to call {call.code}")
                else:
                    print(f"Rule {rule_code} already exists for call {call.code}")

        db.commit()
        print("Seeding completed successfully!")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
