# Initial Tasks

## Goal
Set up the initial local development environment and scaffold the MVP backend.

## Phase 1
1. Create the project structure.
2. Create a FastAPI application with:
   - `/health`
   - `/api/v1/documents/upload`
   - `/api/v1/reports/generate`
3. Add environment-based settings.
4. Add Dockerfile and docker-compose.yml.
5. Add PostgreSQL and Redis services to docker-compose.
6. Add database session scaffolding and base models.
7. Add a document metadata model.
8. Add a report model.
9. Add a minimal PDF parsing service scaffold.
10. Add a retrieval interface abstraction.
11. Add a report service scaffold.
12. Add starter tests.
13. Add README with startup instructions.

## Phase 2
1. Add Alembic migrations. ✓
2. Add file storage abstraction and local storage service. ✓
3. Improve configuration management (development, test). ✓
4. Add background task/worker scaffolding. *
5. Add deterministic checklist engine. ✓
6. Refactor model to ApplicationPackage (merged PDF submission). ✓
7. Add vector backend abstraction:
   - pgvector adapter placeholder
   - Qdrant adapter placeholder
8. Add sample seed script for test documents.

## Done Criteria
The project is considered initialized when:
- `docker compose up --build` works,
- API starts successfully,
- health endpoint returns OK,
- upload endpoint accepts a PDF,
- generate report endpoint returns a scaffolded report response,
- tests can run successfully.
