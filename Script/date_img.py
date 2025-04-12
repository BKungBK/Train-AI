import gradio as gr
import easygui as eg
import os
import exifread
from PIL import Image
from PIL.ExifTags import TAGS
import pillow_heif
from datetime import datetime

# Function to open a folder selection dialog
def select_folder():
    folder = eg.diropenbox(title="เลือกระบบโฟลเดอร์", default="D:/")
    return folder or ""

# Function to read the date taken from EXIF metadata using exifread
def get_image_datetime(img_path):
    try:
        with open(img_path, 'rb') as image_file:
            tags = exifread.process_file(image_file)
            for tag in ["EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"]:
                date_taken = tags.get(tag)
                if date_taken:
                    return datetime.strptime(str(date_taken), "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"Error reading EXIF data from {img_path}: {e}")
        return None
    return None

# Function to convert images to JPG format and preserve EXIF if available
def convert_to_jpg(filepath, output_path):
    try:
        with Image.open(filepath) as img:
            # ดึงข้อมูล EXIF ถ้ามี
            exif_data = img.info.get('exif')
            rgb_image = img.convert('RGB')
            if exif_data:
                rgb_image.save(output_path, "JPEG", exif=exif_data)
            else:
                rgb_image.save(output_path, "JPEG")
        return True
    except Exception as e:
        print(f"Error converting {filepath} to JPG: {e}")
        return False

# Function to process and rename images
def rename_images(folder, name):
    if not folder or not os.path.isdir(folder):
        return "❌ กรุณาเลือกโฟลเดอร์ที่ถูกต้อง!"

    count = 0
    skipped = 0
    converted_files = []
    try:
        pillow_heif.register_heif_opener()  # Ensure HEIC support is registered

        # Step 1: Convert all non-JPG files to JPG while preserving EXIF
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext != '.jpg':
                temp_path = os.path.join(folder, f"temp_{os.path.splitext(filename)[0]}.jpg")
                if convert_to_jpg(filepath, temp_path):
                    converted_files.append(filepath)  # Mark original file for deletion
                    # Replace original file with converted file
                    os.rename(temp_path, os.path.join(folder, f"{os.path.splitext(filename)[0]}.jpg"))

        # Step 2: Rename all JPG files
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext != '.jpg':
                continue

            # Extract date information
            date_taken = get_image_datetime(filepath)
            if not date_taken:
                skipped += 1
                continue

            # Generate new name
            new_name = f"{name}_IMG_{date_taken.strftime('%Y-%m-%d_%H%M%S')}[{count+1}].jpg"
            new_path = os.path.join(folder, new_name)

            # Handle duplicate names
            index = 1
            while os.path.exists(new_path):
                new_name = f"{name}_IMG_{date_taken.strftime('%Y-%m-%d_%H%M%S')}_{index}[{count+1}].jpg"
                new_path = os.path.join(folder, new_name)
                index += 1

            os.rename(filepath, new_path)
            count += 1

        # Step 3: Remove original non-JPG files (ที่เหลืออยู่)
        for file in converted_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

        result_message = f"✅ เปลี่ยนชื่อสำเร็จ {count} ไฟล์!"
        if skipped > 0:
            result_message += f" ❌ ข้าม {skipped} ไฟล์ (ไม่มีข้อมูลวันที่ถ่าย)"
        return result_message
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

# Gradio GUI
with gr.Blocks(theme=gr.themes.Default()) as app:
    gr.Markdown("## 🖼️ ตัวเปลี่ยนชื่อและแปลงไฟล์ภาพถ่าย")
    folder_path = gr.Textbox(label="📂 โฟลเดอร์ที่เลือก", interactive=False)
    select_button = gr.Button("เลือกโฟลเดอร์")
    name_dropdown = gr.Dropdown(
        choices=["Assanee", "Narathip", "Natchanon", "Narinthorn", "Phummarat", "Woranat"],  # รายชื่อที่ต้องการให้เลือก
        label="👤 เลือกชื่อของผู้ถ่าย",
        value="Assanee"  # ค่าเริ่มต้น
    )
    run_button = gr.Button("🚀 เริ่มเปลี่ยนชื่อและแปลงไฟล์")
    output = gr.Textbox(label="ผลลัพธ์", interactive=False)
    console = gr.Textbox(label="📜 Console Log", interactive=False, lines=10)

    def update_folder():
        selected = select_folder()
        if selected and os.path.isdir(selected):
            return selected if selected else "❌ ไม่ได้เลือกโฟลเดอร์"
        return "❌ ไม่ได้เลือกโฟลเดอร์"

    def process_images(folder, name):
        result = rename_images(folder, name)
        return result, f"Processing completed for folder: {folder}"

    select_button.click(fn=update_folder, outputs=folder_path)
    run_button.click(fn=process_images, inputs=[folder_path, name_dropdown], outputs=[output, console])

app.launch()
