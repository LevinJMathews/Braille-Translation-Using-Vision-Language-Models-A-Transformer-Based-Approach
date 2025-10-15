import fitz
import os
from PIL import Image

def pdf_to_jpg(pdf_path, output_folder, target_size=(512, 512), zoom_x=2.0, zoom_y=2.0, rotation_angle=0):
    """Converts each page of a PDF to a JPG image and resizes it."""
    try:
        pdf_document = fitz.open(pdf_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)

            output_file = os.path.join(output_folder, f"{pdf_document.name.split('/')[-1].split('.')[0]}_page_{page_num + 1}.jpg")
            pix.save(output_file, "jpeg")

            # Resize the image to 512x512 using Pillow
            img = Image.open(output_file)
            if rotation_angle != 0:
                img = img.rotate(rotation_angle, expand=True) # Rotate with Pillow
            img = img.resize(target_size, Image.LANCZOS)
            img.save(output_file, "jpeg")

        pdf_document.close()
        print(f"PDF converted to JPG images (resized to {target_size}) in '{output_folder}'")
        return pdf_document.name.split('/')[-1].split('.')[0]

    except Exception as e:
        print(f"Error converting PDF to images: {e}")

def extract_text_from_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)
        text_pages = []

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text("text")
            text = text.replace('\n', '')
            text_pages.append(text)

        pdf_document.close()
        return text_pages

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

def pdf_to_images_and_annotations(pdf_files, img_folder="./data/img", annotations_file="./data/annotations.txt"):
    """Converts multiple PDFs to images and extracts text annotations."""

    if not os.path.exists("./data"):
        os.makedirs("./data")

    with open(annotations_file, "w", encoding="utf-8") as f:
        for pdf_path in pdf_files:
            file_name = pdf_to_jpg(pdf_path, img_folder)
            extracted_texts = extract_text_from_pdf(pdf_path)

            if extracted_texts:
                for page_num, text in enumerate(extracted_texts):
                    image_name = f"{file_name}_page_{page_num + 1}.jpg"
                    f.write(f"{image_name}\t{text}\n")
                print(f"Annotations for '{pdf_path}' saved.")
            else:
                print(f"Failed to extract text from '{pdf_path}'. Annotations not saved.")
    print(f"All annotations saved to '{annotations_file}'")

pdf_file_paths = ["./pre/set1.pdf", "./pre/set2.pdf"]
pdf_to_images_and_annotations(pdf_file_paths)
