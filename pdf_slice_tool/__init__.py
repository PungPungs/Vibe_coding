"""PDF Slice Tool - Split PDF files into individual pages for LangChain."""

from .pdf_slicer import PDFSlicerTool, slice_pdf

__version__ = "0.1.0"
__all__ = ["PDFSlicerTool", "slice_pdf"]
