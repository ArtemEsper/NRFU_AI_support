# Requirements

## Functional Requirements

### FR-001 Document Upload
The system must allow uploading project-related documents for analysis.

### FR-002 PDF Parsing
The system must extract text and page-level structure from PDF files.

### FR-003 Document Registry
The system must store metadata about uploaded documents:
- filename
- document type
- source
- upload timestamp
- status
- checksum

### FR-004 Knowledge Base Document Storage
The system must support storing and indexing NRFU regulations, call documents, annexes, and instructions.

### FR-005 Formal Criteria Checklist
The system must support a structured checklist of formal call requirements.

### FR-006 Criteria Evaluation
The system must evaluate whether required items are present or missing in a project package.

### FR-007 Grounded Retrieval
The system must retrieve supporting text passages from indexed documents.

### FR-008 Report Generation
The system must generate a structured report containing:
- detected files
- missing files
- possible issues
- retrieved evidence
- confidence or review-needed flags

### FR-009 Auditability
The system must preserve source references for each finding.

### FR-010 Health Monitoring
The system must expose a health endpoint.

## Non-Functional Requirements

### NFR-001 Local Development
The project must run locally via Docker Compose.

### NFR-002 Azure Stack Readiness
The project must be deployable on Linux VMs in Azure Stack Hub.

### NFR-003 Modularity
The vector backend must be abstracted so the system can later use pgvector or Qdrant.

### NFR-004 Configuration
All runtime configuration must be environment-based.

### NFR-005 Maintainability
The codebase must be structured for future extension with chatbot and agentic workflows.

### NFR-006 Resource Awareness
The MVP must remain lightweight enough to run on a single 4 vCPU / 16 GB RAM Linux VM.

## Out of Scope
- public-facing chatbot
- advanced authentication/SSO
- expert matching
- OCR-heavy pipelines for all files by default
- multilingual production UX
