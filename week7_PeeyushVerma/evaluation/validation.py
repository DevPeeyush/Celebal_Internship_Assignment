import json
import os
from utils.logger import get_logger
from retrieval.retriever import VectorStoreManager
from llm.generator import LLMGenerator
from utils.metrics import metrics_instance

logger = get_logger(__name__)

class Validator:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.llm_generator = LLMGenerator()
        
    def run_validation(self, questions, report_path="evaluation_report.json"):
        """Run validation on a set of sample questions."""
        
        if metrics_instance.vector_db_size == 0:
            logger.error("Vector database is empty. Cannot run validation.")
            return None
            
        logger.info(f"Running validation on {len(questions)} questions...")
        
        results = []
        total_retrieval_latency = 0
        total_generation_latency = 0
        
        for q in questions:
            # Retrieve
            retrieved_chunks = self.vector_manager.search(q, k=3)
            retrieval_latency = metrics_instance.last_retrieval_latency
            total_retrieval_latency += retrieval_latency
            
            # Extract retrieved sources for report
            sources = []
            for doc, score in retrieved_chunks:
                sources.append({
                    "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
                    "score": float(score),
                    "preview": doc.page_content[:100]
                })
                
            # Generate
            answer = self.llm_generator.generate_answer(q, retrieved_chunks)
            generation_latency = metrics_instance.last_generation_latency
            total_generation_latency += generation_latency
            
            # Store result
            results.append({
                "question": q,
                "generated_answer": answer,
                "retrieved_sources": sources,
                "retrieval_latency_s": retrieval_latency,
                "generation_latency_s": generation_latency
            })
            
        # Compile System Report
        report = {
            "system_metrics": {
                "vector_db_size": metrics_instance.vector_db_size,
                "total_chunks": metrics_instance.chunk_count,
                "average_retrieval_latency_s": total_retrieval_latency / len(questions) if questions else 0,
                "average_generation_latency_s": total_generation_latency / len(questions) if questions else 0
            },
            "queries": results
        }
        
        # Save report
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved validation report to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            
        return report

if __name__ == "__main__":
    validator = Validator()
    sample_questions = [
        "What is the main idea of the document?",
        "How does Retrieval-Augmented Generation work?",
        "What embedding models are used?"
    ]
    validator.run_validation(sample_questions)
