"""PDF Slicer - Split PDF files into individual pages."""

import os
from pathlib import Path
from typing import Optional, Union, List
from pypdf import PdfReader, PdfWriter


def slice_pdf(
    input_pdf: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "page",
    start_page: int = 1,
    end_page: Optional[int] = None
) -> List[str]:
    """
    Slice a PDF file into individual page files.

    Args:
        input_pdf: Path to the input PDF file
        output_dir: Directory to save sliced PDFs (default: same as input PDF)
        prefix: Prefix for output filenames (default: "page")
        start_page: Starting page number (1-indexed, default: 1)
        end_page: Ending page number (1-indexed, default: last page)

    Returns:
        List of paths to created PDF files

    Example:
        >>> slice_pdf("document.pdf", output_dir="./output")
        ['output/page_1.pdf', 'output/page_2.pdf', 'output/page_3.pdf', ...]
    """
    input_path = Path(input_pdf)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    # Determine output directory
    if output_dir is None:
        output_path = input_path.parent / f"{input_path.stem}_pages"
    else:
        output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the PDF
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)

    # Validate page ranges
    if start_page < 1:
        start_page = 1
    if end_page is None or end_page > total_pages:
        end_page = total_pages

    if start_page > end_page:
        raise ValueError(f"start_page ({start_page}) cannot be greater than end_page ({end_page})")

    # Slice pages
    output_files = []

    for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])

        # Create output filename (1-indexed for user-friendly naming)
        output_file = output_path / f"{prefix}_{page_num + 1}.pdf"

        # Write the page to file
        with open(output_file, "wb") as output_stream:
            writer.write(output_stream)

        output_files.append(str(output_file))

    return output_files


def slice_pdf_cli(
    input_pdf: str,
    output_dir: Optional[str] = None,
    prefix: str = "page"
) -> str:
    """
    CLI-friendly version that returns a formatted string result.

    Args:
        input_pdf: Path to the input PDF file
        output_dir: Directory to save sliced PDFs
        prefix: Prefix for output filenames

    Returns:
        Formatted string with slice results
    """
    try:
        output_files = slice_pdf(input_pdf, output_dir, prefix)

        result = f"Successfully sliced PDF into {len(output_files)} pages:\n"
        result += f"Output directory: {Path(output_files[0]).parent}\n\n"
        result += "Created files:\n"
        for file_path in output_files:
            result += f"  - {Path(file_path).name}\n"

        return result

    except Exception as e:
        return f"Error slicing PDF: {str(e)}"


# LangChain Tool Integration
try:
    from langchain.tools import BaseTool
    from pydantic import Field

    class PDFSlicerTool(BaseTool):
        """LangChain tool for slicing PDF files into individual pages."""

        name: str = "pdf_slicer"
        description: str = (
            "Slice a PDF file into individual page files. "
            "Input should be a path to a PDF file. "
            "The tool will create separate PDF files for each page. "
            "Useful when you need to process or analyze individual pages of a PDF document."
        )

        def _run(
            self,
            input_pdf: str,
            output_dir: Optional[str] = None,
            prefix: str = "page"
        ) -> str:
            """Run the PDF slicing operation."""
            return slice_pdf_cli(input_pdf, output_dir, prefix)

        async def _arun(
            self,
            input_pdf: str,
            output_dir: Optional[str] = None,
            prefix: str = "page"
        ) -> str:
            """Async version of the PDF slicing operation."""
            # For now, just call the sync version
            # In a production environment, you might want to use asyncio
            return self._run(input_pdf, output_dir, prefix)

except ImportError:
    # LangChain not available, skip tool definition
    class PDFSlicerTool:
        """Placeholder when LangChain is not installed."""
        def __init__(self):
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_slicer.py <input_pdf> [output_dir] [prefix]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    prefix = sys.argv[3] if len(sys.argv) > 3 else "page"

    print(slice_pdf_cli(input_pdf, output_dir, prefix))
