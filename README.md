# Plagiarism & AI Content Detection Tool

## Overview
This tool is designed to assist educators in detecting plagiarism, identifying AI-generated content, and evaluating student submissions against predefined reference materials. It takes as input:
1. A directory containing reference PDFs (context for correction).
2. A selection of the detection model.
3. A directory containing student-submitted PDFs.

## Features
- **Plagiarism Detection**: Compares student submissions against reference materials to identify potential plagiarism.
- **AI Content Detection**: Analyzes text to determine the likelihood of AI-generated content.
- **AI Teacher Evaluation**: Assesses submissions based on selected models to provide feedback on quality and originality.

## Installation
Ensure you have Python installed, then clone the repository and install dependencies:
```bash
git clone https://github.com/dataduran/plagios-AIdetectorApp.git
cd plagios-AIdetectorApp
pip install -r requirements.txt
```

## Usage
Run the tool using the following command:
```bash
python detect.py --reference_path "path/to/reference_pdfs" --model "chosen_model" --student_path "path/to/student_pdfs"
```

### Arguments
- `--reference_path`: Directory containing PDFs used as reference material.
- `--model`: The detection model to be used (e.g., `GPT-4-detector`, `BERT-based-plagiarism-checker`).
- `--student_path`: Directory containing student-submitted PDFs.

## Output
The tool generates a report with:
- A plagiarism percentage for each document.
- AI-content probability scores.
- Teacher evaluation feedback based on the selected model.

## Contributing
Feel free to contribute by submitting issues or pull requests.

## Contact
For support or inquiries, please reach out to [pdurgo@gmail.com].

