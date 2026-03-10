# Scaffolds for remaining services. 
# ParsingService and ChecklistService have been replaced by real implementations 
# in app/services/pdf_parsing.py and app/services/checklist.py.

from typing import List, Dict, Any

class RetrievalService:
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        TODO: Implement grounded retrieval.
        Use pgvector or Qdrant for vector search.
        """
        return [{"source": "regulations.pdf", "page": 1, "text": "Sample retrieved text"}]

class ReportingService:
    async def generate_report(self, document_id: int) -> Dict[str, Any]:
        """
        TODO: Implement LLM-based report generation.
        Combine retrieval and checklist evaluation results.
        """
        return {
            "document_id": document_id,
            "status": "completed",
            "findings": [
                {"category": "Formal", "status": "pass", "evidence": "All required files found"}
            ],
            "confidence": 0.95
        }

retrieval_service = RetrievalService()
reporting_service = ReportingService()
