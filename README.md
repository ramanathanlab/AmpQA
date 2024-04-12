# AmpQA
This script is designed to generate question-and-answer (QA) datasets for Antimicrobial Peptides (AMP), based on either interactions or function data downloaded from UniProt. It is capable of handling large datasets and utilizes powerful language models for generating function-based questions.

## Features

- **Protein Interaction QA**: Generate questions from a dataset describing protein interactions.
- **Protein Function QA**: Utilize a pretrained model to generate function-based questions.
- **Customizable Input/Output**: Specify input files and output directories through command-line arguments.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- torch
- transformers

This script is tested on Linux and Windows operating systems.

## Installation

1. **Clone the repository** (if the script is hosted on a Git repository):
   ```bash
   git clone https://your-repository-link.git
   cd your-repository-directory
   ```

2. **Set up a Python virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
   ```

3. **Install required packages**:
   ```bash
   pip install torch transformers
   ```
## Usage

To use the script, you will need to provide the type of questions, the path to the input JSON file, and the output directory where the results will be stored.

### Command-Line Arguments

- `-t`, `--type`: Specify the type of questions (`i` for interactions, `f` for functions)
- `-i`, `--input`: Path to the input JSON file containing protein data
- `-o`, `--output`: Path to the directory where the results will be saved

### Example Commands

- **Generating Interaction Questions**:
  ```bash
  python script_name.py -t i -i your/path/to/protein_interactions.json -o your/path/to/output
  ```

- **Generating Function Questions**:
  ```bash
  python script_name.py -t f -i your/path/to/protein_functions.json -o your/path/to/output
  ```

## Output

The output will be JSON files containing the QA datasets:
- `interaction_questions.json` for interaction-based questions.
- `function_questions.json` for function-based questions.

These files will be saved in the specified output directory.
