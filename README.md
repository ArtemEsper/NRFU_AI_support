# Engineering Overview

AI-assisted backend system designed to support formal verification of research grant applications submitted to the National Research Foundation of Ukraine (NRFU).

The system processes application packages consisting of merged Ukrainian and optional English PDFs. It extracts metadata and full document text, and evaluates the submission against formal criteria defined in the funding call regulations.

Evaluation results are returned as structured reports containing grounded evidence from both the submitted application and the authoritative call documentation.

The architecture prioritizes deterministic rule-based checks, while being designed for future AI-assisted evaluation and retrieval-augmented reasoning (RAG).

### Key Capabilities

- package-centric submission model for research applications
- document ingestion and validation pipeline for merged PDFs
- full-text extraction and heuristic metadata detection
- rule-based formal compliance engine linked to funding calls
- grounded reporting referencing official call regulations
- architecture prepared for future RAG / LLM-based evaluation

### Core Technologies

Python • FastAPI • PostgreSQL • Docker • PyMuPDF • SQLAlchemy • Alembic

### Design Principles

- **Deterministic first** — formal checks are implemented as explicit rules
- **Grounded evaluation** — findings reference authoritative regulatory documents
- **Traceable decisions** — reports contain evidence from both regulations and application files
- **Extensible architecture** — designed to support vector search, RAG, and AI assistants

### Future Extensions

- vector search over regulatory documents (pgvector / Qdrant)
- RAG-based explanation of formal requirements
- AI-assisted reviewer support tools
- automated reviewer workflows
- integration with NRFU submission platforms

# AI-Assisted Grant Application Validation System (NRFU)

This is an MVP for the National Research Foundation of Ukraine (NRFU) to support formal criteria checking of project application packages. It implements a package-centric submission model where each application consists of merged Ukrainian and (optionally) English PDFs.

## Core Features

1.  **Package-Centric Submission**: Group multiple files (UK/EN) under a single `ApplicationPackage`.
2.  **Call-Level Metadata & Authoritative Documents**:
    -   **Operational Metadata**: Tracks `applications_received_count`, `applications_expected_count`, deadlines, and status per `Call`.
    -   **Authoritative Call Documents**: Support for uploading and storing official call regulations, templates, and manuals (Source of Truth).
    -   **Extracted Text Storage**: All call documents are parsed and full text is stored to enable future RAG-based grounding.
3.  **Call-Specific Rule Definitions (Grounded Evaluation)**:
    -   **Rule Modeling**: Rules are explicitly linked to a `Call` and can point to a `CallDocument` (e.g., a specific Regulation PDF) as their source of truth.
    -   **Metadata**: Each rule includes a `rule_code`, `severity` (critical, major, minor), and `source_section` (e.g., "Section 3.1").
    -   **Source-Linked Findings**: Reports now include references to the authoritative documents, source passages, and package-side evidence.
4.  **Enhanced PDF Ingestion**:
  - **Full-Text Extraction**: Preserves the entire extracted text for full-document rule evaluation.
  - **Heuristic Metadata Extraction**: Automatically detects project registration numbers, call titles, and project titles using deterministic regex and keyword matching.
  - **Validation**: Ensures files are valid PDFs, checks for minimum text length, and prevents duplicate uploads for the same language.
- **Deterministic Checklist Engine**:
  - **Mandatory File Checks**: Verifies presence of the mandatory Ukrainian PDF.
  - **Conditional English Checks**: Enforces English mirror requirement based on Funding Call rules.
  - **Bilingual Consistency**: Checks if registration numbers and titles match or are present in both Ukrainian and English files.
  - **Full-Text Section Detection**: Scans the entire document for NRFU-specific markers (e.g., "Реєстраційний номер проєкту", "Назва конкурсу", "Бюджет", "Додаток", "Підпис").
- **Call-Specific Reporting**: 
  - Generates a detailed JSON report including package metadata, extracted file info, and per-rule findings.
  - **Source-Linked Findings**: Each finding now includes:
    - `severity`, `rule_text`, and `status`.
    - `source_document_title` and `source_section` (linkage to regulatory source).
    - `source_passage`: A 300-character snippet from the official `CallDocument` around the specified section.
    - `package_evidence`: Deterministic proof from the application package (e.g., matching file ID, detected registration number, or section count).
  - **Overall Status**: Calculates a final status (`pass`, `fail`, `review`) based on rule outcomes.

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

#### Option A: Use the Seeding Script (Quick Start)
A developer-friendly seeding script is provided to create sample calls (Ukrainian-only and Bilingual) with all default checklist rules:

```bash
# Run the seeding script inside the container
docker compose exec app python scripts/seed_data.py
```

After seeding, you will have:
- `call_id: 1` (CALL-UK-2024): Ukrainian-only call with metadata and a linked **Source of Truth** document.
- `call_id: 2` (CALL-BI-2024): Bilingual call (requires English mirror).

#### Option B: Create a Customized Call (Manual)
If you need to create a specific call manually before creating a package, follow this workflow:

**1. Create the Call**
```bash
curl -X POST "http://localhost:8000/api/v1/calls" \
     -H "Content-Type: application/json" \
     -d '{
       "title_uk": "Передова наука в Україні",
       "title_en": "Excellence Science in Ukraine",
       "code": "2023.03",
       "requires_english_mirror": true,
       "status": "active"
     }'
```
*Note the returned `"id"` (e.g., `3`).*

**2. Upload Authoritative Documents (Optional Grounding)**
Before you can ground your rules, you need to upload the authoritative documents for the call. Note the returned document `"id"` (e.g., `1`).
```bash
curl -X POST "http://localhost:8000/api/v1/calls/3/documents" \
     -F "title=Official Call Conditions" \
     -F "document_type=regulation" \
     -F "is_source_of_truth=true" \
     -F "file=@regulations.pdf"
```

**3. Add Rules to the Call**
A new call has no rules by default. You can add rules with full grounding (linking them to the document ID from Step 2) or minimal fields:
```bash
curl -X POST "http://localhost:8000/api/v1/calls/3/rules" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Mandatory Ukrainian PDF",
       "rule_code": "MANDATORY_UK_FILE",
       "severity": "critical",
       "source_document_id": 1,
       "source_section": "Section 3.1"
     }'
```
*(Repeat for other rule codes like `CONDITIONAL_EN_FILE`, `SECTION_CHECK`, etc. The `source_document_id` and `source_section` are optional at creation but recommended for grounded reporting.)*

---

## Example API Workflow

### 0. List Existing Calls
To see all currently available Funding Calls:
```bash
curl -X GET "http://localhost:8000/api/v1/calls"
```

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

    # 5. Call Rules Management
    List all rules for a call:
    ```bash
    curl -X GET "http://localhost:8000/api/v1/calls/1/rules"
    ```
    
    Add a custom rule to a call (with grounding):
    ```bash
    curl -X POST "http://localhost:8000/api/v1/calls/1/rules" \
         -H "Content-Type: application/json" \
         -d '{
           "title": "Specific Section Check",
           "description": "Check for a very specific project section.",
           "rule_code": "SECTION_CHECK",
           "severity": "major",
           "source_document_id": 1,
           "source_section": "Annex 4",
           "is_active": true
         }'
    ```

    Note: You can create rules with all fields at once, or create them with minimal fields and update them later.

    Update an existing rule (e.g., to link it to a source document):
    ```bash
    curl -X PUT "http://localhost:8000/api/v1/rules/15" \
         -H "Content-Type: application/json" \
         -d '{
           "source_document_id": 1,
           "source_section": "Clause 2.1"
         }'
    ```
    *Note: This will now correctly link the rule to the authoritative document and regenerate the report with grounded findings (source passage and document title).*

    Generate (or regenerate) a report to see grounded findings:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/reports/generate?package_id=1"
    ```

    Delete a rule:
    ```bash
    curl -X DELETE "http://localhost:8000/api/v1/rules/15"
    ```

### 6. Manage Call Documents (Authoritative Sources)
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
