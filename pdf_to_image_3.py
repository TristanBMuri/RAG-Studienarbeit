#!/usr/bin/env python3
"""
NOTE: If you encounter the warning
      "grpc_wait_for_shutdown_with_timeout() timed out."
Please downgrade grpcio by running:
    pip install grpcio==1.60.1
"""

import base64
import getpass
import os
import fitz  # PyMuPDF

from langchain_google_genai import ChatGoogleGenerativeAI

class PDFMarkdownConverter:
    def __init__(
        self,
        pdf_dir="Docs",
        model="gemini-2.0-flash-exp",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ):
        self.pdf_dir = pdf_dir
        # Initialize the multimodal LLM.
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        # Improved system message with clear separation between instructions.
        self.system_message = (
            "You are an expert assistant specialized in converting images of PDF pages into well-structured markdown. "
            "Analyze the image content (text, tables, and graphics) and return the relevant markdown. "
            "Do not summarize. Do not echo any base64 data or image metadata in your output."
        )

    def extract_pdf_images_base64(self, file_name):
        """
        Opens the specified PDF from the pdf_dir, converts each page to a PNG image,
        and encodes it as a base64 string. Returns a list of image attachment dictionaries.
        Also saves each page as a PNG file for verification.
        """
        path = os.path.join(self.pdf_dir, file_name)
        doc = fitz.open(path)
        attachments = []

        # Create output directory if it doesn't exist
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap()  # Render the page to an image
            img_bytes = pix.tobytes("png")  # Convert to PNG format

            # Save the image to a file for verification
            output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
            with open(output_path, "wb") as img_file:
                img_file.write(img_bytes)

            encoded_image = base64.b64encode(img_bytes).decode("utf-8")
            attachments.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
            })

        return attachments

    def convert_pdf_to_markdown(self, file_name):
        """
        For each page in the PDF, creates a message with:
         - A system message with clear instructions.
         - A user message containing a list with:
             1. A text part instructing the conversion.
             2. An image part containing the base64-encoded image.
        Invokes the LLM and returns a list of markdown results per page.
        """
        attachments = self.extract_pdf_images_base64(file_name)
        meta_data = []

        for i, attachment in enumerate(attachments):
            # Build messages according to the multimodal format.
            messages = [
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please convert the attached image to markdown. Analyze the image content carefully."
                        },
                        {
                            "type": "image_url",
                            "image_url": attachment["image_url"]
                        }
                    ]
                }
            ]
            # Invoke the LLM using the recommended method.
            response = self.llm.invoke(messages)

            # Extract markdown content from the response.
            markdown_result = getattr(response, "content", response)
            meta_data.append({
                "page": i + 1,
                "markdown": markdown_result
            })

        return meta_data

if __name__ == '__main__':
    # Request API key if not already set.
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    converter = PDFMarkdownConverter()
    pdf_file = input("Enter the PDF file name (located in the 'Docs' folder): ")
    results = converter.convert_pdf_to_markdown(pdf_file)

    # Print the markdown output for each page.
    for page in results:
        print(f"\n--- Page {page['page']} Markdown ---\n")
        print(page["markdown"])
