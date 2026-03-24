# AI-Assisted Grant Application Validation System (NRFU)

This is an MVP for the National Research Foundation of Ukraine (NRFU) to support formal criteria checking of project application packages. It implements a package-centric submission model where each application consists of merged Ukrainian and (optionally) English PDFs.

## Why This Project Matters

Administrative validation of research grant applications is typically performed manually and requires significant staff effort. Funding agencies must verify whether submitted applications comply with formal requirements defined in funding call regulations. These checks may include verifying document completeness, required sections, formatting constraints, and bilingual submission requirements.

This project demonstrates how modern backend architecture and document processing pipelines can automate large parts of this process using:
- **Deterministic rule-based compliance evaluation**
- **Grounded findings** linked to authoritative regulations
- **Traceable evidence** extracted from submitted application documents
- **Modular architecture** designed for future AI-assisted evaluation

## Engineering Overview

AI-assisted backend system designed to support formal verification of research grant applications. The system models funding calls, regulatory documents, application packages, and formal rules as linked entities, enabling transparent and reproducible formal evaluation.

### Key Capabilities
- **Package-Centric Submission**: Group multiple files (UK/EN) under a single `ApplicationPackage`.
- **Authoritative Document Management**: Support for uploading and storing official call regulations and templates as the "Source of Truth".
- **Enhanced PDF Ingestion**: Full-text extraction and heuristic metadata detection (registration numbers, titles) using PyMuPDF.
- **Layout Enrichment Layer**: Optional LiteParse integration for handling complex layouts, tables, and OCR-needed pages.
- **Deterministic Rule Engine**: Call-specific rules (mandatory files, bilingual consistency, section detection) with grounded evaluation.
- **Grounded Reporting**: Structured JSON reports with references to regulatory sources, source passages, and package-side evidence.

### Design Principles
- **Deterministic First**: Formal checks are implemented as explicit, reproducible rules.
- **Grounded Evaluation**: Findings reference authoritative regulatory documents.
- **Traceable Decisions**: Reports contain evidence from both regulations and application files.
- **Extensible**: Architecture prepared for future RAG / LLM-based evaluation and vector search.

---

## System Architecture

```text
Users / Operators
(applicants, NRFU staff, scientific board members, reviewers)
                         |
                         v
+--------------------------------------------------------------+
|                        FastAPI API Layer                     |
|--------------------------------------------------------------|
|  /calls   /calls/{id}/documents   /calls/{id}/rules         |
|  /packages   /packages/{id}/upload   /reports/generate      |
+--------------------------------------------------------------+
                         |
                         v
+--------------------------------------------------------------+
|                    Application Service Layer                 |
|--------------------------------------------------------------|
|  Call Management                                             |
|  Application Package Management                             |
|  Call Document Management                                   |
|  Report Generation                                           |
|  Rule / Checklist Management                                |
+--------------------------------------------------------------+
              |                          |                     |
              v                          v                     v
+---------------------------+  +------------------------+  +----------------------+
|   PDF Parsing Service     |  | Deterministic Rule     |  | Source Passage /     |
|---------------------------|  | Evaluation Engine      |  | Evidence Assembly    |
| - full text extraction    |  |------------------------|  |----------------------|
| - page count              |  | - file presence checks |  | - source snippets    |
| - checksum                |  | - parseability checks  |  | - package evidence   |
| - metadata extraction     |  | - section checks       |  | - grounded findings  |
| - reg. number detection   |  | - bilingual checks     |  |                      |
| - title detection         |  | - call-specific rules  |  |                      |
+---------------------------+  +------------------------+  +----------------------+
              |                          |                     |
              +--------------------------+---------------------+
                                         |
                                         v
+--------------------------------------------------------------+
|                     Persistence / Storage Layer              |
|--------------------------------------------------------------|
| PostgreSQL                                                   |
| - Calls                                                      |
| - CallDocuments                                              |
| - ChecklistItems / Call Rules                                |
| - ApplicationPackages                                        |
| - SubmittedFiles                                             |
| - Reports / ReportFindings                                   |
|                                                              |
| Local File Storage (uploads/)                                |
| - merged Ukrainian PDFs                                      |
| - merged English PDFs                                        |
| - call regulations / manuals / templates                     |
+--------------------------------------------------------------+
                                         |
                                         v
+--------------------------------------------------------------+
|                    Future Extension Layer                    |
|--------------------------------------------------------------|
| - Reviewer workflow (approve / override / comments)          |
| - Full-text search over call documents                       |
| - Vector DB (pgvector or Qdrant)                             |
| - RAG / LLM-based explanation and support                    |
| - Background workers for heavy parsing                       |
| - Azure Stack VM deployment                                  |
+--------------------------------------------------------------+
```

## Data Flow

1️⃣ **Application Submission**: Applicants or operators create an `ApplicationPackage` linked to a specific funding call. Merged application PDFs (Ukrainian mandatory, English optional) are uploaded via `POST /api/v1/packages/{package_id}/upload`.

2️⃣ **PDF Ingestion and Parsing**: The system validates PDF structure, generates checksums (SHA256), detects page counts, and performs full-text and heuristic metadata extraction using **PyMuPDF**.
**Layout Enrichment (Optional)**: If enabled, **LiteParse** provides a secondary enrichment layer for difficult pages (tables, scans, complex layouts) using layout-aware extraction and OCR.

3️⃣ **Rule Evaluation**: The `ChecklistService` evaluates the package against deterministic call-specific rules (presence, parseability, bilingual consistency, section detection).

4️⃣ **Grounded Reporting**: A structured report is generated. Each finding includes a status, severity, and (if applicable) a link to the authoritative `CallDocument` (title, section, and source passage) alongside package-side evidence.

---

## Technologies Used

- **Backend**: Python, FastAPI
- **Data Processing**: PyMuPDF (PDF parsing), LiteParse (Layout Enrichment), Heuristic metadata extraction
- **Data Storage**: PostgreSQL, SQLAlchemy ORM, Alembic (migrations)
- **Infrastructure**: Docker, Docker Compose
- **Testing**: pytest

---

## Example API Workflow

### 1. Initialize the Environment
```bash
cp .env.example .env
docker compose up --build
docker compose exec app alembic upgrade head
```

### 2. Seed Initial Data
```bash
docker compose exec app python scripts/seed_data.py
```
This creates sample calls with default rules and linked Source of Truth documents.

### 3. Create a Submission & Upload Files
```bash
# Create Package
curl -X POST "http://localhost:8000/api/v1/packages" \
     -H "Content-Type: application/json" \
     -d '{"call_id": 1, "project_identifier": "NRFU-PROJ-2024-ABC", "submission_mode": "online"}'

# Upload PDF
curl -X POST "http://localhost:8000/api/v1/packages/1/upload" \
     -F "file=@my_submission_uk.pdf" \
     -F "language=uk"
```

### 4. Generate Report
```bash
curl -X POST "http://localhost:8000/api/v1/reports/generate?package_id=1"
```

### Example Report Response
```json
{
  "package_id": 1,
  "project_identifier": "NRFU-PROJ-2024-ABC",
  "call_code": "2023.03",
  "overall_status": "review",
  "findings": [
    {
      "rule_code": "MANDATORY_UK_FILE",
      "status": "pass",
      "severity": "critical"
    },
    {
      "rule_code": "SECTION_CHECK",
      "status": "review",
      "severity": "major",
      "source_document_title": "NRFU Call Regulations",
      "source_section": "Section 3.2",
      "source_passage": "Applications must include a detailed project budget..."
    }
  ]
}
```

---

## Getting Started

### Prerequisites
- Docker and Docker Compose.
- Python 3.11+ (for local development).

### Local Setup
1. **Launch Environment**: `docker compose up --build`
2. **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
3. **Database Migrations**: `docker compose exec app alembic upgrade head`

### Optional: Layout Enrichment (LiteParse)
To enable LiteParse enrichment for complex documents:
1. **Install LiteParse and Tesseract**:
   ```bash
   pip install liteparse
   # System dependency for OCR
   apt-get install tesseract-ocr
   ```
2. **Enable in `.env`**:
   ```env
   USE_LITEPARSE_ENRICHMENT=True
   LITEPARSE_OCR_ENABLED=True
   ```
3. **Run Evaluation Script**:
   ```bash
   python scripts/evaluate_liteparse.py
   ```

### Running Tests
```bash
docker compose run --rm -e ENVIRONMENT=test app pytest
```

---

## Roadmap

- [x] Grant call API and metadata model
- [x] Package-centric submission model
- [x] PDF ingestion, parsing, and metadata extraction
- [x] Call document management
- [x] Call-specific rule definitions
- [x] Deterministic formal checks
- [x] Grounded report generation
- [ ] Improve rule grounding and source passage quality
- [ ] Improve NRFU-specific title/section extraction
- [ ] Build reviewer workflow and override logic
- [ ] Add full-text search over call documents
- [ ] Add vector search and RAG over authoritative documents
- [ ] Add LLM-based support for staff and applicants
- [ ] Prepare Azure Stack production deployment

---

## Project Structure
- `app/api/v1`: Route handlers for packages, uploads, reports, and calls.
- `app/models`: SQLAlchemy domain models (`Call`, `ApplicationPackage`, `SubmittedFile`, `CallDocument`).
- `app/services`: 
  - `pdf_parsing.py`: Text extraction and heuristic metadata logic via PyMuPDF.
  - `layout_enrichment.py`: LiteParse-based layout enrichment and OCR fallback.
  - `checklist.py`: Deterministic rule evaluation engine.
  - `storage.py`: Local file storage abstraction.
- `migrations`: Alembic database versioning.

## Current Limitations
- **Synchronous Parsing**: PDF parsing occurs within the request cycle.
- **Keyword-Based Extraction**: Metadata extraction is heuristic and may require manual review.
- **Local Storage**: Files are saved to the `uploads/` folder; no cloud storage integration yet.
- **Deterministic Rules**: Semantic LLM checks (e.g., checking content vs. regulations) are not yet implemented.
