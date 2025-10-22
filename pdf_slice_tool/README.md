# PDF Slice Tool

A Python utility for slicing PDF files into individual pages, with built-in LangChain tool integration.

## Features

- Split PDF files into individual page files
- LangChain tool integration for AI workflows
- Customizable output directory and file naming
- Support for page range selection
- Command-line interface
- Python API for programmatic use

## Installation

```bash
pip install -r requirements.txt
```

### Minimal Installation (without LangChain)

If you only need the basic PDF slicing functionality:

```bash
pip install pypdf
```

## Usage

### 1. As a Python Module

```python
from pdf_slice_tool import slice_pdf

# Basic usage - slice all pages
output_files = slice_pdf("document.pdf")
# Creates: document_pages/page_1.pdf, page_2.pdf, ..., page_10.pdf

# Custom output directory
output_files = slice_pdf(
    "document.pdf",
    output_dir="./my_output",
    prefix="section"
)
# Creates: my_output/section_1.pdf, section_2.pdf, ...

# Slice specific page range
output_files = slice_pdf(
    "document.pdf",
    output_dir="./chapters",
    prefix="chapter",
    start_page=5,
    end_page=10
)
# Creates: chapters/chapter_5.pdf, ..., chapter_10.pdf
```

### 2. Command Line Interface

```bash
# Basic usage
python -m pdf_slice_tool.pdf_slicer document.pdf

# Custom output directory
python -m pdf_slice_tool.pdf_slicer document.pdf ./output

# Custom prefix
python -m pdf_slice_tool.pdf_slicer document.pdf ./output my_page
```

### 3. As a LangChain Tool

```python
from pdf_slice_tool import PDFSlicerTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI

# Create the tool
pdf_slicer = PDFSlicerTool()

# Use in a LangChain agent
llm = OpenAI(temperature=0)
tools = [pdf_slicer]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# The agent can now use the PDF slicer
result = agent.run(
    "Please slice the document.pdf file into individual pages"
)
```

### 4. With LangChain Chains

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pdf_slice_tool import PDFSlicerTool

# Create the tool
slicer = PDFSlicerTool()

# Use directly
result = slicer._run("document.pdf", output_dir="./sliced_pages")
print(result)
```

## API Reference

### `slice_pdf()`

```python
def slice_pdf(
    input_pdf: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "page",
    start_page: int = 1,
    end_page: Optional[int] = None
) -> List[str]
```

**Parameters:**
- `input_pdf`: Path to the input PDF file
- `output_dir`: Directory to save sliced PDFs (default: `{input_filename}_pages/`)
- `prefix`: Prefix for output filenames (default: `"page"`)
- `start_page`: Starting page number, 1-indexed (default: 1)
- `end_page`: Ending page number, 1-indexed (default: last page)

**Returns:**
- List of paths to created PDF files

**Raises:**
- `FileNotFoundError`: If input PDF doesn't exist
- `ValueError`: If page range is invalid

### `PDFSlicerTool` (LangChain Tool)

```python
class PDFSlicerTool(BaseTool):
    name: str = "pdf_slicer"
    description: str = "Slice a PDF file into individual page files..."
```

**Methods:**
- `_run(input_pdf, output_dir=None, prefix="page")`: Synchronous execution
- `_arun(input_pdf, output_dir=None, prefix="page")`: Async execution

## Examples

### Example 1: Processing a 10-page PDF

```python
from pdf_slice_tool import slice_pdf

# Input: 10-page PDF
result = slice_pdf("report.pdf")

# Output:
# report_pages/page_1.pdf
# report_pages/page_2.pdf
# report_pages/page_3.pdf
# ...
# report_pages/page_10.pdf
```

### Example 2: Custom Naming

```python
output_files = slice_pdf(
    "thesis.pdf",
    output_dir="./thesis_chapters",
    prefix="chapter"
)

# Output:
# thesis_chapters/chapter_1.pdf
# thesis_chapters/chapter_2.pdf
# ...
```

### Example 3: Page Range

```python
# Extract only pages 15-20
output_files = slice_pdf(
    "book.pdf",
    output_dir="./selected_pages",
    start_page=15,
    end_page=20
)
```

### Example 4: LangChain Agent Workflow

```python
from langchain.agents import Tool
from pdf_slice_tool import slice_pdf_cli

# Create a custom tool wrapper
pdf_tool = Tool(
    name="PDFSlicer",
    func=slice_pdf_cli,
    description="Splits a PDF into individual pages"
)

# Use in your agent workflow
tools = [pdf_tool, other_tools...]
```

## Error Handling

The tool includes comprehensive error handling:

```python
from pdf_slice_tool import slice_pdf

try:
    output_files = slice_pdf("nonexistent.pdf")
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    output_files = slice_pdf("doc.pdf", start_page=10, end_page=5)
except ValueError as e:
    print(f"Invalid page range: {e}")
```

## Output Structure

When you slice a PDF, the tool creates an output directory with the following structure:

```
output_directory/
├── page_1.pdf
├── page_2.pdf
├── page_3.pdf
├── ...
└── page_N.pdf
```

Each file contains exactly one page from the original PDF.

## Requirements

- Python 3.7+
- pypdf >= 3.17.0
- langchain >= 0.1.0 (optional, for LangChain integration)
- pydantic >= 2.0.0 (optional, for LangChain integration)

## License

This tool is part of the Vibe Coding Utilities collection.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## See Also

- [PyPDF Documentation](https://pypdf.readthedocs.io/)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
