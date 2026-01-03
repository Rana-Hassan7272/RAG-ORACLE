"""
Example usage of the RAG Pipeline
Run this script to test the RAG pipeline.
"""

from rag_pipeline import RAGPipeline
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Example usage of RAG pipeline."""
    
    # Get the script directory and resolve documents path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    documents_path = project_root / "documents"
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        document_source=str(documents_path),
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="huggingface",
        model_type=None,
        top_k=5,
        temperature=0.7
    )
    
    # Query the pipeline
    question = "TELL BELIEFS AND THEIR IMPACT ON OUR LIVES"
    result = pipeline.query(question)
    
    # ---------------------------------------------------------
    # FINAL SIMPLIFIED OUTPUT (STRICT TEMPLATE)
    # ---------------------------------------------------------
    
    print(f"Answer:\n{result.get('answer', 'No answer produced')}")
    print("\n" + "-" * 50 + "\n")

    root_causes_result = result.get("root_causes", [])
    
    output = {
        "query_id": result.get("trace_id", "N/A"),
        "question": result.get("question", ""),
        "root_causes": root_causes_result
    }
    
    print(json.dumps(output, indent=2))
    
    print("\n" + "=" * 50 + "\n")
    print("RAG HEALTH REPORT (All Query History):")
    print("=" * 50 + "\n")
    
    report = pipeline.root_cause_oracle.get_report(last_n=None)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
