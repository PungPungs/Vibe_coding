"""Example usage of the PDF Slice Tool."""

from pdf_slice_tool import slice_pdf, PDFSlicerTool


def example_basic_usage():
    """Example 1: Basic PDF slicing."""
    print("=" * 60)
    print("Example 1: Basic PDF Slicing")
    print("=" * 60)

    # This will slice a PDF into individual pages
    # If you have a test PDF, replace 'document.pdf' with its path
    try:
        output_files = slice_pdf("document.pdf")

        print(f"\nSuccessfully sliced PDF into {len(output_files)} pages")
        print("\nCreated files:")
        for file_path in output_files[:5]:  # Show first 5
            print(f"  - {file_path}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more")

    except FileNotFoundError:
        print("\nNote: Replace 'document.pdf' with an actual PDF file path")
        print("Example: slice_pdf('/path/to/your/file.pdf')")


def example_custom_output():
    """Example 2: Custom output directory and prefix."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Output and Naming")
    print("=" * 60)

    try:
        output_files = slice_pdf(
            "document.pdf",
            output_dir="./my_output",
            prefix="section"
        )

        print(f"\nSliced PDF with custom naming")
        print(f"Output directory: ./my_output")
        print(f"File prefix: section")
        print(f"\nTotal pages: {len(output_files)}")

    except FileNotFoundError:
        print("\nThis example requires a PDF file named 'document.pdf'")


def example_page_range():
    """Example 3: Slicing specific page range."""
    print("\n" + "=" * 60)
    print("Example 3: Slicing Specific Page Range")
    print("=" * 60)

    try:
        output_files = slice_pdf(
            "document.pdf",
            output_dir="./selected_pages",
            prefix="page",
            start_page=3,
            end_page=7
        )

        print(f"\nExtracted pages 3-7")
        print(f"Created {len(output_files)} files:")
        for file_path in output_files:
            print(f"  - {file_path}")

    except FileNotFoundError:
        print("\nThis example requires a PDF file named 'document.pdf'")


def example_langchain_tool():
    """Example 4: Using as a LangChain tool."""
    print("\n" + "=" * 60)
    print("Example 4: LangChain Tool Integration")
    print("=" * 60)

    try:
        # Create the LangChain tool
        pdf_slicer = PDFSlicerTool()

        print(f"\nTool Name: {pdf_slicer.name}")
        print(f"Tool Description: {pdf_slicer.description}")

        # Use the tool directly
        result = pdf_slicer._run("document.pdf", output_dir="./langchain_output")
        print(f"\nResult:\n{result}")

    except ImportError as e:
        print(f"\nLangChain not available: {e}")
        print("Install with: pip install langchain")
    except FileNotFoundError:
        print("\nThis example requires a PDF file named 'document.pdf'")


def example_langchain_agent():
    """Example 5: Using in a LangChain agent (conceptual)."""
    print("\n" + "=" * 60)
    print("Example 5: LangChain Agent Usage (Conceptual)")
    print("=" * 60)

    example_code = '''
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from pdf_slice_tool import PDFSlicerTool

# Create the tool
pdf_slicer = PDFSlicerTool()

# Initialize agent with the tool
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[pdf_slicer],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent can now slice PDFs
result = agent.run(
    "Please slice the report.pdf file into individual pages"
)
'''

    print("\nExample code for using with LangChain agent:")
    print(example_code)
    print("\nNote: This requires an OpenAI API key and the langchain package")


def create_sample_pdf():
    """Helper: Create a sample PDF for testing (requires reportlab)."""
    print("\n" + "=" * 60)
    print("Bonus: Creating Sample PDF for Testing")
    print("=" * 60)

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        output_file = "sample_document.pdf"
        c = canvas.Canvas(output_file, pagesize=letter)

        # Create a 10-page sample PDF
        for page_num in range(1, 11):
            c.drawString(100, 750, f"This is page {page_num}")
            c.drawString(100, 700, f"Sample content for page {page_num}")
            c.drawString(100, 650, "This PDF can be used to test the PDF Slice Tool")
            c.showPage()

        c.save()
        print(f"\nCreated sample PDF: {output_file}")
        print("You can now use this file with the examples above!")

        return output_file

    except ImportError:
        print("\nTo create a sample PDF, install reportlab:")
        print("  pip install reportlab")
        return None


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "PDF SLICE TOOL EXAMPLES" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")

    # Create a sample PDF if possible
    sample_pdf = create_sample_pdf()

    # Run examples
    example_basic_usage()
    example_custom_output()
    example_page_range()
    example_langchain_tool()
    example_langchain_agent()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)

    if sample_pdf:
        print(f"\nTry running with the sample PDF:")
        print(f"  python -m pdf_slice_tool.pdf_slicer {sample_pdf}")
    else:
        print("\nTo test with your own PDF:")
        print("  from pdf_slice_tool import slice_pdf")
        print('  slice_pdf("your_document.pdf")')

    print()


if __name__ == "__main__":
    main()
