import sys
import argparse
from gui import PDFAnalyzerGUI, QApplication
import os
import json
from datetime import datetime
from pdf_extractor import PDFExtractor
from analyzers import PlagiarismAnalyzer, AIContentAnalyzer, AITeacherAnalyzer

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Analyze PDFs for plagiarism and AI-generated content')
    parser.add_argument('--gui', action='store_true', 
                       help='Launch in GUI mode')
    parser.add_argument('--folder-path', 
                       help='Path to the folder containing PDF files')
    parser.add_argument('--doc-path', 
                       help='Path to the folder containing documentation and books for training')
    parser.add_argument('--output', default='analysis_report.json', 
                       help='Output file name for the report')
    return parser.parse_args()

def find_pdf_files(folder_path: str):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def analyze_folder(folder_path: str, doc_path: str, output_path: str):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    if not os.path.exists(doc_path):
        print(f"Error: Documentation folder '{doc_path}' does not exist.")
        return

    # Get list of PDF files
    pdf_files = find_pdf_files(folder_path)
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting analysis...")

    # Extract text from PDFs
    pdf_extractor = PDFExtractor()
    documents = {}
    for pdf_file in pdf_files:
        try:
            text = pdf_extractor.extract_text(pdf_file)
            documents[pdf_file] = text
            print(f"Processed {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"Error extracting text from {pdf_file}: {str(e)}")

    # Extract text from documentation
    doc_files = find_pdf_files(doc_path)
    doc_texts = []
    for doc_file in doc_files:
        try:
            text = pdf_extractor.extract_text(doc_file)
            doc_texts.append(text)
            print(f"Processed documentation {os.path.basename(doc_file)}")
        except Exception as e:
            print(f"Error extracting text from {doc_file}: {str(e)}")

    # Analyze for plagiarism
    print("\nPerforming plagiarism analysis...")
    plagiarism_analyzer = PlagiarismAnalyzer()
    plagiarism_results = plagiarism_analyzer.analyze_documents(documents)

    # Analyze for AI-generated content
    print("Performing AI content analysis...")
    ai_analyzer = AIContentAnalyzer()
    ai_content_results = {}
    for file_path, text in documents.items():
        ai_content_results[file_path] = ai_analyzer.analyze_text(text)

    # AI Teacher functionality
    print("Performing AI teacher evaluation...")
    ai_teacher_analyzer = AITeacherAnalyzer(doc_texts)
    scores = ai_teacher_analyzer.compare_pdfs(documents)
    best_pdf, best_score = ai_teacher_analyzer.assign_best_qualification(scores)

    # Generate report
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_files_analyzed": len(pdf_files),
        "plagiarism_analysis": plagiarism_results,
        "ai_content_analysis": ai_content_results,
        "ai_teacher_evaluation": {
            "best_pdf": best_pdf,
            "best_score": best_score,
            "scores": scores
        },
        "summary": {
            "total_files_analyzed": len(pdf_files),
            "high_plagiarism_count": sum(1 for score in 
                plagiarism_results["similarity_scores"].values() if score > 0.8),
            "likely_ai_generated_count": sum(1 for score in 
                ai_content_results.values() if score > 0.9)
        }
    }

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis complete. Report saved to {output_path}")

def main():
    args = parse_arguments()
    
    if args.gui:
        app = QApplication(sys.argv)
        window = PDFAnalyzerGUI()
        window.show()
        sys.exit(app.exec())
    else:
        if not args.folder_path or not args.doc_path:
            print("Error: Please provide --folder-path and --doc-path or use --gui")
            sys.exit(1)
        analyze_folder(args.folder_path, args.doc_path, args.output)

if __name__ == "__main__":
    main()