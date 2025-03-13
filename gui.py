from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                          QTableWidget, QTableWidgetItem, QProgressBar,
                          QMessageBox, QTabWidget, QHeaderView, QComboBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

import sys
import os
import json
from datetime import datetime
from pdf_extractor import PDFExtractor
from analyzers import PlagiarismAnalyzer, AIContentAnalyzer, AITeacherAnalyzer

def find_pdf_files(folder_path: str):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder_path, doc_path, model_name):
        super().__init__()
        self.folder_path = folder_path
        self.doc_path = doc_path
        self.model_name = model_name

    def run(self):
        try:
            # Initialize analyzers
            pdf_extractor = PDFExtractor()
            plagiarism_analyzer = PlagiarismAnalyzer()
            ai_analyzer = AIContentAnalyzer(self.model_name)
            ai_teacher_analyzer = AITeacherAnalyzer(self._load_training_texts())

            # Get PDF files
            pdf_files = find_pdf_files(self.folder_path)
            
            if not pdf_files:
                self.error.emit("No PDF files found in the selected folder.")
                return

            # Extract text from PDFs
            documents = {}
            for i, pdf_file in enumerate(pdf_files):
                try:
                    file_path = pdf_file
                    text = pdf_extractor.extract_text(file_path)
                    documents[file_path] = text
                    self.progress.emit(int((i + 1) / len(pdf_files) * 50))
                except Exception as e:
                    self.error.emit(f"Error processing {pdf_file}: {str(e)}")
                    return

            # Analyze for plagiarism
            plagiarism_results = plagiarism_analyzer.analyze_documents(documents)
            self.progress.emit(60)

            # Analyze for AI content
            ai_content_results = {}
            for file_path, text in documents.items():
                ai_content_results[file_path] = ai_analyzer.analyze_text(text)
            self.progress.emit(75)

            # AI Teacher evaluation
            ai_teacher_scores = ai_teacher_analyzer.compare_pdfs(documents)
            best_pdf, best_score = ai_teacher_analyzer.assign_best_qualification(ai_teacher_scores)
            self.progress.emit(100)

            # Generate report
            report = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_files_analyzed": len(pdf_files),
                "plagiarism_analysis": plagiarism_results,
                "ai_content_analysis": ai_content_results,
                "ai_teacher_evaluation": {
                    "best_pdf": best_pdf,
                    "best_score": best_score,
                    "scores": ai_teacher_scores
                },
                "summary": {
                    "total_files_analyzed": len(pdf_files),
                    "high_plagiarism_count": sum(1 for score in 
                        plagiarism_results["similarity_scores"].values() if score > 0.8),
                    "likely_ai_generated_count": sum(1 for score in 
                        ai_content_results.values() if score > 0.9)
                }
            }

            self.finished.emit(report)

        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")

    def _load_training_texts(self):
        doc_files = find_pdf_files(self.doc_path)
        pdf_extractor = PDFExtractor()
        training_texts = []
        for doc_file in doc_files:
            try:
                file_path = doc_file
                text = pdf_extractor.extract_text(file_path)
                training_texts.append(text)
            except Exception as e:
                print(f"Error extracting text from {doc_file}: {str(e)}")
        return training_texts

class PDFAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Analyzer")
        self.setMinimumSize(800, 600)
        self.current_results = None
        self.initUI()

    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Input section
        input_group = QGroupBox("Input Section")
        input_layout = QGridLayout(input_group)
        self.folder_label = QLabel("Drag & drop PDF folder or")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_folder)
        input_layout.addWidget(self.folder_label, 0, 0)
        input_layout.addWidget(self.browse_btn, 0, 1)
        layout.addWidget(input_group)

        # Documentation input section
        doc_input_group = QGroupBox("Documentation Input Section")
        doc_input_layout = QGridLayout(doc_input_group)
        self.doc_folder_label = QLabel("Drag & drop documentation folder or")
        self.doc_browse_btn = QPushButton("Browse")
        self.doc_browse_btn.clicked.connect(self.browse_doc_folder)
        doc_input_layout.addWidget(self.doc_folder_label, 0, 0)
        doc_input_layout.addWidget(self.doc_browse_btn, 0, 1)
        layout.addWidget(doc_input_group)

        # Model selection dropdown
        model_selection_group = QGroupBox("Model Selection")
        model_selection_layout = QGridLayout(model_selection_group)
        self.model_label = QLabel("Select AI Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["gpt2", "deepseek", "gemma2", "llama3", "mixtral"])
        model_selection_layout.addWidget(self.model_label, 0, 0)
        model_selection_layout.addWidget(self.model_dropdown, 0, 1)
        layout.addWidget(model_selection_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Results tabs
        self.tabs = QTabWidget()
        
        # Plagiarism results tab
        self.plagiarism_table = QTableWidget()
        self.plagiarism_table.setColumnCount(3)
        self.plagiarism_table.setHorizontalHeaderLabels(
            ["File 1", "File 2", "Similarity Score"])
        self.plagiarism_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.plagiarism_table, "Plagiarism Analysis")

        # AI content results tab
        self.ai_content_table = QTableWidget()
        self.ai_content_table.setColumnCount(2)
        self.ai_content_table.setHorizontalHeaderLabels(
            ["File", "AI Generation Probability"])
        self.ai_content_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.ai_content_table, "AI Content Analysis")

        # AI teacher evaluation results tab
        self.ai_teacher_table = QTableWidget()
        self.ai_teacher_table.setColumnCount(2)
        self.ai_teacher_table.setHorizontalHeaderLabels(
            ["File", "Evaluation Score"])
        self.ai_teacher_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.ai_teacher_table, "AI Teacher Evaluation")

        layout.addWidget(self.tabs)

        # Statistics section
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        self.total_files_label = QLabel("Total Files Analyzed: 0")
        self.high_plagiarism_label = QLabel("High Plagiarism Cases: 0")
        self.likely_ai_generated_label = QLabel("Likely AI-Generated Documents: 0")
        stats_layout.addWidget(self.total_files_label, 0, 0)
        stats_layout.addWidget(self.high_plagiarism_label, 1, 0)
        stats_layout.addWidget(self.likely_ai_generated_label, 2, 0)
        layout.addWidget(stats_group)

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        # Enable drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if (event.mimeData().hasUrls()):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            folder_path = files[0]
            if os.path.isdir(folder_path):
                self.analyze_folder(folder_path)
            else:
                QMessageBox.warning(self, "Error", "Please drop a folder")

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing PDFs")
        if folder_path:
            self.analyze_folder(folder_path)

    def browse_doc_folder(self):
        doc_folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Documentation")
        if doc_folder_path:
            self.doc_folder_path = doc_folder_path

    def analyze_folder(self, folder_path):
        if not hasattr(self, 'doc_folder_path'):
            QMessageBox.warning(self, "Error", "Please select a documentation folder first.")
            return

        model_name = self.model_dropdown.currentText()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(folder_path, self.doc_folder_path, model_name)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)

    def display_results(self, results):
        self.current_results = results
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)

        # Update plagiarism table
        self.plagiarism_table.setRowCount(0)
        sorted_plagiarism_results = sorted(results["plagiarism_analysis"]["similarity_scores"].items(), key=lambda item: item[1], reverse=True)
        for pair, score in sorted_plagiarism_results:
            row = self.plagiarism_table.rowCount()
            self.plagiarism_table.insertRow(row)
            file1, file2 = pair.split(" vs ")
            file1_dir = os.path.basename(os.path.dirname(file1))
            file2_dir = os.path.basename(os.path.dirname(file2))
            self.plagiarism_table.setItem(row, 0, QTableWidgetItem(file1_dir))
            self.plagiarism_table.setItem(row, 1, QTableWidgetItem(file2_dir))
            self.plagiarism_table.setItem(row, 2, QTableWidgetItem(f"{score:.2%}"))

        # Update AI content table
        self.ai_content_table.setRowCount(0)
        for file_path, score in results["ai_content_analysis"].items():
            row = self.ai_content_table.rowCount()
            self.ai_content_table.insertRow(row)
            file_dir = os.path.basename(os.path.dirname(file_path))
            self.ai_content_table.setItem(row, 0, QTableWidgetItem(file_dir))
            self.ai_content_table.setItem(row, 1, QTableWidgetItem(f"{score:.2%}"))

        # Update AI teacher evaluation table
        self.ai_teacher_table.setRowCount(0)
        for file_path, score in results["ai_teacher_evaluation"]["scores"].items():
            row = self.ai_teacher_table.rowCount()
            self.ai_teacher_table.insertRow(row)
            file_dir = os.path.basename(os.path.dirname(file_path))
            self.ai_teacher_table.setItem(row, 0, QTableWidgetItem(file_dir))
            self.ai_teacher_table.setItem(row, 1, QTableWidgetItem(f"{score:.2f}"))

        self.plagiarism_table.resizeColumnsToContents()
        self.ai_content_table.resizeColumnsToContents()
        self.ai_teacher_table.resizeColumnsToContents()

        # Update statistics
        self.total_files_label.setText(f"Total Files Analyzed: {results['summary']['total_files_analyzed']}")
        self.high_plagiarism_label.setText(f"High Plagiarism Cases: {results['summary']['high_plagiarism_count']}")
        self.likely_ai_generated_label.setText(f"Likely AI-Generated Documents: {results['summary']['likely_ai_generated_count']}")

    def export_results(self):
        if not self.current_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "JSON Files (*.json)")
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2)
                QMessageBox.information(
                    self, "Success", "Results exported successfully!")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to export results: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFAnalyzerGUI()
    window.show()
    sys.exit(app.exec())