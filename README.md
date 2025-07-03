# Token Counter for Files and Folders

A lightweight Python utility for counting tokens in text files and directories using OpenAI's `tiktoken` library. Perfect for estimating token usage before API calls or analyzing text content for language model applications.

## Features

- **Single File Analysis**: Count tokens in individual text files
- **Recursive Directory Scanning**: Process entire folder structures
- **Robust Error Handling**: Graceful handling of encoding issues and file errors
- **Multiple Output Formats**: Detailed reports with file-by-file breakdown
- **Performance Optimized**: Efficient processing of large codebases
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
```bash
pip install tiktoken
```

### Clone Repository
```bash
git clone https://github.com/tltlv/token-counter.git
cd token-counter
```

## Usage

### Count Tokens in a Single File
```bash
python count_tokens_file.py path/to/your/file.txt
```

**Example Output:**
```
📄 Analyzing file: example.py
✅ Token count: 1,247 tokens
📊 File size: 8.3 KB
⏱️  Processing time: 0.02 seconds
```

### Count Tokens in a Directory
```bash
python count_tokens_folder.py path/to/your/directory
```

**Example Output:**
```
📁 Scanning directory: ./my_project

📄 ./my_project/main.py: 1,247 tokens
📄 ./my_project/utils.py: 892 tokens
📄 ./my_project/config.json: 156 tokens
⚠️  Skipped ./my_project/image.png (binary file)

📊 Summary:
   • Files processed: 3
   • Files skipped: 1
   • Total tokens: 2,295
   • Total size: 24.7 KB
   • Processing time: 0.15 seconds
```

## Command Line Options

### File Counter Options
```bash
python count_tokens_file.py [OPTIONS] <file_path>

Options:
  -q, --quiet     Suppress detailed output, show only token count
  -s, --stats     Show additional file statistics
  -h, --help      Show help message
```

### Folder Counter Options
```bash
python count_tokens_folder.py [OPTIONS] <folder_path>

Options:
  -q, --quiet     Show only summary statistics
  -e, --exclude   Exclude file patterns (e.g., "*.log,*.tmp")
  -f, --filter    Include only specific file types (e.g., "*.py,*.js")
  -r, --report    Generate detailed CSV report
  -h, --help      Show help message
```

## Supported File Types

The tool automatically detects and processes text files, including:
- Source code files (`.py`, `.js`, `.java`, `.cpp`, etc.)
- Documentation (`.md`, `.txt`, `.rst`)
- Configuration files (`.json`, `.yaml`, `.toml`)
- Web files (`.html`, `.css`, `.xml`)
- Data files (`.csv`, `.tsv`)

Binary files are automatically skipped with informative messages.

## Technical Details

### Tokenizer
- Uses `cl100k_base` encoding (compatible with GPT-3.5-turbo and GPT-4)
- Accurate token counting for API cost estimation
- Handles Unicode text properly

### Performance
- Processes ~1MB of text per second on modern hardware
- Memory-efficient streaming for large files
- Parallel processing for directory scanning

### Error Handling
- Graceful handling of permission errors
- Automatic detection of binary vs text files
- Detailed error messages for troubleshooting
- Continues processing even if individual files fail

## Examples

### Basic Usage
```bash
# Count tokens in a Python file
python count_tokens_file.py main.py

# Count tokens in entire project
python count_tokens_folder.py ./my_project
```

### Advanced Usage
```bash
# Only Python files, quiet output
python count_tokens_folder.py --filter "*.py" --quiet ./src

# Exclude log files and generate report
python count_tokens_folder.py --exclude "*.log,*.tmp" --report ./project

# Quick file check
python count_tokens_file.py --quiet README.md
```

## Integration

### Python Script Integration
```python
from count_tokens_file import count_tokens_in_file
from count_tokens_folder import count_tokens_in_folder

# Count tokens in a file
token_count = count_tokens_in_file("example.txt")
print(f"Token count: {token_count}")

# Count tokens in a directory
total_tokens = count_tokens_in_folder("./project")
print(f"Total tokens: {total_tokens}")
```

### API Cost Estimation
```python
# Estimate API costs (example rates)
tokens = count_tokens_in_file("prompt.txt")
estimated_cost = tokens * 0.002 / 1000  # $0.002 per 1K tokens
print(f"Estimated API cost: ${estimated_cost:.4f}")
```

## Troubleshooting

### Common Issues

**UnicodeDecodeError**: File contains non-UTF-8 characters
```bash
# Solution: The tool automatically detects and skips binary files
# Check file encoding: file --mime-encoding yourfile.txt
```

**Permission Denied**: Insufficient file permissions
```bash
# Solution: Run with appropriate permissions or change file ownership
chmod +r yourfile.txt
```

**Module Not Found**: tiktoken not installed
```bash
# Solution: Install required dependencies
pip install tiktoken
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup
```bash
git clone https://github.com/tltlv/token-counter.git
cd token-counter
pip install -r requirements.txt
python -m pytest tests/
```


## Changelog

### v1.0.0
- Initial release
- Command-line options
- Error handling
- File type filtering
- Performance optimizations
- Professional output formatting
- UTF-8 support