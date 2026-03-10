# Project Instructions

## Project Name
NRFU AI Formal Criteria Checking MVP

## Purpose
This project is an MVP for the National Research Foundation of Ukraine (NRFU).  
Its initial goal is to support formal criteria checking of project application packages for funding calls.

The system must:
- ingest and parse uploaded documents (initially PDFs, later DOCX),
- index legal, procedural, and call-related documents,
- compare application packages against formal call requirements,
- generate a structured report with findings and source-based evidence,
- remain modular for later deployment to Azure Stack Hub on Linux VMs.

## Deployment Context
The project is developed locally in PyCharm and Docker first.
Target production is Azure Stack Hub / VM-based deployment, not full Azure managed services.

Constraints:
- assume Linux VM deployment,
- avoid dependencies on Azure-only managed AI services,
- use Docker containers,
- keep storage and app components modular,
- design for one main VM first, with optional second VM for database/vector services.

## MVP Scope
The first MVP only covers:
- document ingestion,
- metadata extraction,
- retrieval over NRFU regulations and call documents,
- formal criteria report generation for one call type.

Do NOT implement:
- public chatbot,
- multi-agent swarm,
- Kubernetes,
- complex frontend UI,
- advanced auth integrations at the first step.

## Architecture Principles
- modular monolith first, microservices later if needed,
- deterministic checks first, LLM second,
- all LLM outputs must be grounded in retrieved documents,
- preserve source traceability: file name, page number, chunk id,
- use clear interfaces between ingestion, retrieval, checklist, and reporting,
- keep code production-oriented and testable.

## Core Modules
1. API layer
2. Document ingestion/parsing
3. Metadata extraction
4. Checklist/rule evaluation
5. Retrieval (keyword + vector-ready)
6. LLM-based report drafting
7. Persistence layer
8. Background jobs

## Initial Tech Stack
- Python 3.11
- FastAPI
- SQLAlchemy
- PostgreSQL
- pgvector or Qdrant (vector support should be abstracted)
- Redis for background jobs/cache
- PyMuPDF for PDF parsing
- Docker Compose for local development
- Pydantic for settings and schemas
- Alembic for migrations
- pytest for tests

## Coding Rules
- Use type hints everywhere.
- Use small, testable service classes/functions.
- Avoid hardcoded paths.
- All configuration must come from environment variables.
- Add docstrings for nontrivial modules and services.
- Prefer explicit schemas over loose dicts.
- Use structured logging.
- Keep business logic out of route handlers.

## Output Requirements
When creating code:
- create production-like folder structure,
- include starter tests,
- create `.env.example`,
- create Docker support,
- create a README with run instructions,
- create an initial health endpoint,
- create a minimal document ingestion endpoint,
- create a minimal report-generation pipeline scaffold.

## Important
Do not overengineer.
Set up the minimal working backbone for local development and first launch.
