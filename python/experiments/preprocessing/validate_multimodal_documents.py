"""
Validate multimodal documents from documents.jsonl.

This script:
1. Loads documents from documents.jsonl
2. Validates image paths exist and are readable
3. Validates text content is non-empty
4. Validates metadata fields conform to schema
5. Generates validation report (JSON or HTML)

Author: s Bostan
Created on: Jan, 2026
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


REQUIRED_FIELDS = {"id", "content", "image_path", "metadata"}
REQUIRED_METADATA_FIELDS = {"split", "image_id", "caption_id", "coco_file"}


def load_documents(documents_file: Path) -> List[Dict[str, Any]]:
    """
    Load documents from JSONL file.
    
    Args:
        documents_file: Path to documents.jsonl file
        
    Returns:
        List of document dictionaries
        
    Raises:
        FileNotFoundError: If documents file doesn't exist
    """
    if not documents_file.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_file}")
    
    logger.info(f"Loading documents from: {documents_file}")
    documents = []
    
    with open(documents_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def validate_image_path(image_path_str: str, doc_id: str) -> Dict[str, Any]:
    """
    Validate that image file exists and is readable.
    
    Args:
        image_path_str: Path to image file (as string)
        doc_id: Document ID for error reporting
        
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": False,
        "exists": False,
        "readable": False,
        "error": None
    }
    
    try:
        image_path = Path(image_path_str)
        
        # Check if path exists
        if not image_path.exists():
            result["error"] = "Image file does not exist"
            return result
        
        result["exists"] = True
        
        # Try to verify it's a valid image file
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.verify()
            result["readable"] = True
            result["valid"] = True
        except Exception as e:
            result["error"] = f"Image file is not readable or invalid: {str(e)}"
            return result
        
    except Exception as e:
        result["error"] = f"Error validating image path: {str(e)}"
    
    return result


def validate_text_content(content: Any, doc_id: str) -> Dict[str, Any]:
    """
    Validate that text content is non-empty.
    
    Args:
        content: Text content (can be any type)
        doc_id: Document ID for error reporting
        
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": False,
        "is_string": False,
        "non_empty": False,
        "error": None
    }
    
    # Check if content is a string
    if not isinstance(content, str):
        result["error"] = f"Content is not a string, got type: {type(content).__name__}"
        return result
    
    result["is_string"] = True
    
    # Check if content is non-empty after stripping
    if not content.strip():
        result["error"] = "Content is empty after stripping whitespace"
        return result
    
    result["non_empty"] = True
    result["valid"] = True
    
    return result


def validate_metadata(metadata: Any, doc_id: str) -> Dict[str, Any]:
    """
    Validate metadata fields.
    
    Args:
        metadata: Metadata dictionary
        doc_id: Document ID for error reporting
        
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": False,
        "is_dict": False,
        "missing_fields": [],
        "error": None
    }
    
    # Check if metadata is a dictionary
    if not isinstance(metadata, dict):
        result["error"] = f"Metadata is not a dictionary, got type: {type(metadata).__name__}"
        return result
    
    result["is_dict"] = True
    
    # Check for required fields
    missing_fields = []
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            missing_fields.append(field)
    
    if missing_fields:
        result["missing_fields"] = missing_fields
        result["error"] = f"Missing required metadata fields: {missing_fields}"
        return result
    
    result["valid"] = True
    return result


def validate_document(doc: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validate a single document.
    
    Args:
        doc: Document dictionary
        strict: Whether to use strict validation (all fields required)
        
    Returns:
        Validation result dictionary
    """
    validation = {
        "valid": False,
        "doc_id": doc.get("id", "unknown"),
        "errors": [],
        "warnings": [],
        "image_validation": None,
        "text_validation": None,
        "metadata_validation": None
    }
    
    doc_id = validation["doc_id"]
    
    # Check required fields
    missing_fields = REQUIRED_FIELDS - set(doc.keys())
    if missing_fields:
        validation["errors"].append(f"Missing required fields: {missing_fields}")
        return validation
    
    # Validate image path
    image_path = doc.get("image_path")
    if image_path:
        image_validation = validate_image_path(image_path, doc_id)
        validation["image_validation"] = image_validation
        if not image_validation["valid"]:
            validation["errors"].append(f"Image validation failed: {image_validation['error']}")
    elif strict:
        validation["errors"].append("Image path is missing (strict mode)")
    
    # Validate text content
    content = doc.get("content")
    if content is not None:
        text_validation = validate_text_content(content, doc_id)
        validation["text_validation"] = text_validation
        if not text_validation["valid"]:
            validation["errors"].append(f"Text validation failed: {text_validation['error']}")
    elif strict:
        validation["errors"].append("Content is missing (strict mode)")
    
    # Validate metadata
    metadata = doc.get("metadata")
    if metadata is not None:
        metadata_validation = validate_metadata(metadata, doc_id)
        validation["metadata_validation"] = metadata_validation
        if not metadata_validation["valid"]:
            validation["errors"].append(
                f"Metadata validation failed: {metadata_validation['error']}"
            )
    elif strict:
        validation["errors"].append("Metadata is missing (strict mode)")
    
    # Document is valid if no errors
    validation["valid"] = len(validation["errors"]) == 0
    
    return validation


def validate_all_documents(
    documents: List[Dict[str, Any]],
    strict: bool = True,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate all documents.
    
    Args:
        documents: List of document dictionaries
        strict: Whether to use strict validation
        sample_size: Optional sample size for faster validation (None = all)
        
    Returns:
        Validation report dictionary
    """
    logger.info(f"Validating {len(documents)} documents (strict={strict})...")
    
    if sample_size:
        import random
        random.seed(42)
        documents = random.sample(documents, min(sample_size, len(documents)))
        logger.info(f"Sampling {len(documents)} documents for validation")
    
    valid_count = 0
    invalid_count = 0
    error_counts = {}
    validation_results = []
    
    missing_images = []
    empty_captions = []
    invalid_metadata = []
    
    for i, doc in enumerate(documents):
        if (i + 1) % 1000 == 0:
            logger.info(f"Validated {i + 1}/{len(documents)} documents...")
        
        validation = validate_document(doc, strict=strict)
        validation_results.append(validation)
        
        if validation["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            
            # Track specific error types
            for error in validation["errors"]:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            # Track specific issues
            if validation["image_validation"] and not validation["image_validation"]["valid"]:
                missing_images.append(validation["doc_id"])
            
            if validation["text_validation"] and not validation["text_validation"]["valid"]:
                empty_captions.append(validation["doc_id"])
            
            if validation["metadata_validation"] and not validation["metadata_validation"]["valid"]:
                invalid_metadata.append(validation["doc_id"])
    
    report = {
        "validation_summary": {
            "total_documents": len(documents),
            "valid_documents": valid_count,
            "invalid_documents": invalid_count,
            "validation_rate": valid_count / len(documents) if documents else 0.0,
            "strict_mode": strict
        },
        "error_summary": {
            "missing_images": len(missing_images),
            "empty_captions": len(empty_captions),
            "invalid_metadata": len(invalid_metadata),
            "error_counts": error_counts
        },
        "samples": {
            "missing_images": missing_images[:10],
            "empty_captions": empty_captions[:10],
            "invalid_metadata": invalid_metadata[:10]
        },
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    return report


def save_report_json(report: Dict[str, Any], output_file: Path) -> None:
    """
    Save validation report as JSON.
    
    Args:
        report: Validation report dictionary
        output_file: Path to output JSON file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving validation report to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Validation report saved successfully")


def save_report_html(report: Dict[str, Any], output_file: Path) -> None:
    """
    Save validation report as HTML.
    
    Args:
        report: Validation report dictionary
        output_file: Path to output HTML file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = report["validation_summary"]
    error_summary = report["error_summary"]
    samples = report["samples"]
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Document Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-box.error {{
            border-left-color: #f44336;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .error-list {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .error-item {{
            padding: 5px;
            margin: 5px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .valid {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .invalid {{
            color: #f44336;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Document Validation Report</h1>
        <p>Generated at: {report["generated_at"]}</p>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="stat-box {'error' if summary['validation_rate'] < 1.0 else ''}">
                <div class="stat-label">Total Documents</div>
                <div class="stat-value">{summary['total_documents']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Valid Documents</div>
                <div class="stat-value" class="valid">{summary['valid_documents']}</div>
            </div>
            <div class="stat-box error">
                <div class="stat-label">Invalid Documents</div>
                <div class="stat-value">{summary['invalid_documents']}</div>
            </div>
            <div class="stat-box {'error' if summary['validation_rate'] < 0.95 else ''}">
                <div class="stat-label">Validation Rate</div>
                <div class="stat-value">{summary['validation_rate']:.2%}</div>
            </div>
        </div>
        
        <h2>Error Summary</h2>
        <table>
            <tr>
                <th>Error Type</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>Missing Images</td>
                <td>{error_summary['missing_images']}</td>
            </tr>
            <tr>
                <td>Empty Captions</td>
                <td>{error_summary['empty_captions']}</td>
            </tr>
            <tr>
                <td>Invalid Metadata</td>
                <td>{error_summary['invalid_metadata']}</td>
            </tr>
        </table>
        
        <h2>Error Details</h2>
        <h3>Error Counts</h3>
        <table>
            <tr>
                <th>Error Message</th>
                <th>Count</th>
            </tr>
"""
    
    for error, count in sorted(error_summary['error_counts'].items(), key=lambda x: x[1], reverse=True):
        html_content += f"""
            <tr>
                <td>{error}</td>
                <td>{count}</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h3>Sample Issues</h3>
"""
    
    if samples['missing_images']:
        html_content += f"""
        <h4>Missing Images (first 10)</h4>
        <div class="error-list">
"""
        for doc_id in samples['missing_images']:
            html_content += f'            <div class="error-item">{doc_id}</div>\n'
        html_content += "        </div>\n"
    
    if samples['empty_captions']:
        html_content += f"""
        <h4>Empty Captions (first 10)</h4>
        <div class="error-list">
"""
        for doc_id in samples['empty_captions']:
            html_content += f'            <div class="error-item">{doc_id}</div>\n'
        html_content += "        </div>\n"
    
    html_content += """
    </div>
</body>
</html>
"""
    
    logger.info(f"Saving HTML validation report to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML validation report saved successfully")


def main(
    documents_file: Path,
    output_file: Optional[Path] = None,
    strict: bool = True,
    sample_size: Optional[int] = None,
    format: str = "json"
) -> None:
    """
    Main function to validate documents.
    
    Args:
        documents_file: Path to documents.jsonl file
        output_file: Path to output validation report file
        strict: Whether to use strict validation (all fields required)
        sample_size: Optional sample size for faster validation
        format: Output format (json or html)
    """
    logger.info("=" * 70)
    logger.info("Multimodal Document Validator")
    logger.info("=" * 70)
    logger.info(f"Documents file: {documents_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Strict mode: {strict}")
    logger.info(f"Format: {format}")
    
    # Load documents
    documents = load_documents(documents_file)
    
    if not documents:
        logger.error("No documents loaded. Cannot validate.")
        sys.exit(1)
    
    # Validate documents
    report = validate_all_documents(
        documents=documents,
        strict=strict,
        sample_size=sample_size
    )
    
    # Print summary
    summary = report["validation_summary"]
    error_summary = report["error_summary"]
    
    logger.info("=" * 70)
    logger.info("Validation Complete")
    logger.info("=" * 70)
    logger.info(f"Total documents: {summary['total_documents']}")
    logger.info(f"Valid documents: {summary['valid_documents']}")
    logger.info(f"Invalid documents: {summary['invalid_documents']}")
    logger.info(f"Validation rate: {summary['validation_rate']:.2%}")
    logger.info(f"Missing images: {error_summary['missing_images']}")
    logger.info(f"Empty captions: {error_summary['empty_captions']}")
    logger.info(f"Invalid metadata: {error_summary['invalid_metadata']}")
    
    # Save report
    if output_file:
        if format.lower() == "html":
            save_report_html(report, output_file)
        else:
            save_report_json(report, output_file)
    
    # Exit with error code if validation failed
    if summary['validation_rate'] < 1.0:
        logger.warning("Validation completed with errors. Please review the report.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate multimodal documents from documents.jsonl"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="experiments/datasets/mscoco/processed/pipeline/documents.jsonl",
        help="Path to documents.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output validation report file (optional)"
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict validation mode"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for faster validation"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "html"],
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    main(
        documents_file=Path(args.documents),
        output_file=Path(args.output) if args.output else None,
        strict=not args.no_strict,
        sample_size=args.sample_size,
        format=args.format
    )

