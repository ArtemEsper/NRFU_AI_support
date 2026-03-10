# NRFU AI Formal Criteria Checking MVP

This is an MVP for the National Research Foundation of Ukraine (NRFU) to support formal criteria checking of project application packages. It implements a package-centric submission model where each application consists of merged Ukrainian and (optionally) English PDFs.

## Core Features

- **Package-Centric Submission**: Group multiple files (UK/EN) under a single `ApplicationPackage`.
- **Call-Level Metadata & Authoritative Documents**:
  - **Operational Metadata**: Tracks `applications_received_count`, `applications_expected_count`, deadlines, and status per `Call`.
  - **Authoritative Call Documents**: Support for uploading and storing official call regulations, templates, and manuals (Source of Truth).
  - **Extracted Text Storage**: All call documents are parsed and full text is stored to enable future RAG-based grounding.
- **Enhanced PDF Ingestion**:
  - **Full-Text Extraction**: Preserves the entire extracted text for full-document rule evaluation.
  - **Heuristic Metadata Extraction**: Automatically detects project registration numbers, call titles, and project titles using deterministic regex and keyword matching.
  - **Validation**: Ensures files are valid PDFs, checks for minimum text length, and prevents duplicate uploads for the same language.
- **Deterministic Checklist Engine**:
  - **Mandatory File Checks**: Verifies presence of the mandatory Ukrainian PDF.
  - **Conditional English Checks**: Enforces English mirror requirement based on Funding Call rules.
  - **Bilingual Consistency**: Checks if registration numbers and titles match or are present in both Ukrainian and English files.
  - **Full-Text Section Detection**: Scans the entire document for NRFU-specific markers (e.g., "Реєстраційний номер проєкту", "Назва конкурсу", "Бюджет", "Додаток", "Підпис").
- **Structured Reporting**: Generates a detailed JSON report including package metadata, extracted file info, per-rule findings, and an overall status (`pass`, `fail`, `review`).

## Getting Started

### 1. Prerequisites
- Docker and Docker Compose.
- Python 3.11+ (for local development).

### 2. Launch the Environment
```bash
# Setup environment variables
cp .env.example .env

# Start the application and dependencies (PostgreSQL, Redis)
docker compose up --build
```
The API is available at [http://localhost:8000](http://localhost:8000).  
Interactive documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Initialize the Database
Run migrations to set up the schema:
```bash
docker compose exec app alembic upgrade head
```

### 4. Seed Initial Data
You must create a `Call` and relevant `ChecklistItem`s before you can create any submission packages. 

A developer-friendly seeding script is provided to create sample calls (Ukrainian-only and Bilingual) with all default checklist rules:

```bash
# Run the seeding script inside the container
docker compose exec app python scripts/seed_data.py
```

After seeding, you will have:
- `call_id: 1` (CALL-UK-2024): Ukrainian-only call with metadata (deadlines, expected counts).
- `call_id: 2` (CALL-BI-2024): Bilingual call (requires English mirror) with metadata.
- All standard checklist rules (Mandatory UK, Conditional EN, Parseability, Sections, Consistency) attached to both.

Alternatively, you can manually seed via SQL if needed:
```bash
docker compose exec db psql -U postgres -d nrfu_ai -c "INSERT INTO calls (title, code, requires_english_mirror) VALUES ('Test Call 2024', 'CALL-001', true) RETURNING id;"
```

---

## Example API Workflow

### 1. Create a Submission Package
Create a package linked to the `call_id` you seeded above.
```bash
curl -X POST "http://localhost:8000/api/v1/packages" \
     -H "Content-Type: application/json" \
     -d '{"call_id": 1, "project_identifier": "NRFU-PROJ-2024-ABC", "submission_mode": "online"}'
```
*Note the returned `"id"` (e.g., `1`).*

### 2. Upload Ukrainian Merged PDF
```bash
curl -X POST "http://localhost:8000/api/v1/packages/1/upload" \
     -F "file=@my_submission_uk.pdf" \
     -F "language=uk"
```

### 3. Upload English Merged PDF (if required)
```bash
curl -X POST "http://localhost:8000/api/v1/packages/1/upload" \
     -F "file=@my_submission_en.pdf" \
     -F "language=en"
```

### 4. Generate Formal Criteria Report
```bash
curl -X POST "http://localhost:8000/api/v1/reports/generate?package_id=1"
```

### 5. Manage Call Documents (Authoritative Sources)
Upload official regulations or templates to a specific `Call`.
```bash
curl -X POST "http://localhost:8000/api/v1/calls/1/documents" \
     -F "title=Call Regulations 2024" \
     -F "document_type=regulation" \
     -F "language=uk" \
     -F "version=1.0" \
     -F "is_source_of_truth=true" \
     -F "file=@regulations.pdf"
```

List all documents for a call:
```bash
curl -X GET "http://localhost:8000/api/v1/calls/1/documents"
```

---

## Testing
The project uses `pytest` and an in-memory SQLite database for testing.

```bash
# Run tests inside Docker
docker compose run --rm -e ENVIRONMENT=test app pytest

# Run tests locally (one by one due to SQLite concurrency limits)
ENVIRONMENT=test pytest tests/test_main.py
ENVIRONMENT=test pytest tests/test_pdf.py
ENVIRONMENT=test pytest tests/test_checklist.py
```

## Project Structure
- `app/api/v1`: Route handlers for packages, uploads, reports, and calls.
- `app/models`: SQLAlchemy domain models (`Call`, `ApplicationPackage`, `SubmittedFile`, `CallDocument`).
- `app/services`: 
  - `pdf_parsing.py`: Text extraction and heuristic metadata logic via PyMuPDF.
  - `checklist.py`: Deterministic rule evaluation engine.
  - `storage.py`: Local file storage abstraction.
- `migrations`: Alembic database versioning.

## Current Limitations
- **Synchronous Parsing**: PDF parsing occurs within the request cycle (suitable for MVP, but should be backgrounded later).
- **Keyword-Based Extraction**: Metadata extraction is heuristic (regex/keywords) and may require manual review for non-standard documents.
- **Local Storage**: Files are saved to the `uploads/` folder; no cloud storage integration yet.
- **Deterministic Rules**: Semantic LLM checks (e.g., checking content vs. regulations) are not yet implemented.
