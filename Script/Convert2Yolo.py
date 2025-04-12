import cv2 as cv
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
from threading import Thread
from PIL import Image, ImageTk, ImageSequence  # Updated import for GIF handling

# ตั้งค่า extension ที่จะเซฟรูป
SAVE_IMAGE_EXTENSION = "png"

# Dict label ตามที่ไอโบให้ไว้
dictLabel = {
    'PlasticBottle': 0,
    'GlassBottle': 1,
    'PlasticCup': 2,
    'PlasticBottleCap': 3,
    'Straw': 4,
    'Can': 5,
    'MilkBox': 6,
    'PlasticBag': 7,
    'RopeNets': 8,
    'FireLighter': 9,
    'Foam': 10,
    'Fabric': 11,
    'Sponge': 12,
    'Paper': 13,
    'Metal': 14,
    'PlasticSpoon': 15,
    'Rubber': 16,
    'CigaretteButt': 17,
    'NaturalWood': 18,
    'Shrimp': 19,
    'Shell': 20,
    'Crab': 21,
    'Fish': 22,
    'Coral': 23,
    'Seaweed': 24,
    'Jellyfish': 25,
    'Other': 26,
    'OtherPlastic': 27
}

# Class สำหรับเก็บข้อมูล bounding box
class cvRect:
    def __init__(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xmin = self.x
        self.ymin = self.y
        self.xmax = self.x + self.w
        self.ymax = self.y + self.h
    def area(self):
        return self.w * self.h
    def center(self):
        return [self.x + self.w / 2, self.y + self.h / 2]

# ฟังก์ชันแปลงข้อมูลให้เป็น YOLO format
def makeLabelYOLO(xywh, nameClass, IMAGE_WIDTH, IMAGE_HEIGHT):
    Label_ID = dictLabel[nameClass]
    x_center_norm = xywh.center()[0] / IMAGE_WIDTH
    y_center_norm = xywh.center()[1] / IMAGE_HEIGHT
    width_norm = xywh.w / IMAGE_WIDTH
    height_norm = xywh.h / IMAGE_HEIGHT
    return f"{Label_ID} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

# ฟังก์ชันสำหรับแปลงไฟล์ .anno ทั้งหมดในโฟลเดอร์ input
def convert_folder(path_dir, test_mode, rename_files):
    numIMG = 0
    # สร้าง classes.txt ใน output folder (ถ้ายังไม่มี)
    path_classes_txt = os.path.join(path_dir, "classes.txt")
    if not os.path.exists(path_classes_txt):
        with open(path_classes_txt, "w") as classesFile:
            for key in dictLabel:
                classesFile.write(key + "\n")
    
    # เดินลูปทุกไฟล์ใน input_dir และโฟลเดอร์ย่อย
    for root_dir, dirs, files in os.walk(path_dir):
        for file in files:
            if file.lower().endswith('.anno'):
                # Logic for renaming files if enabled
                if rename_files and '.jpg' in file:
                    new_filename = file.replace('.jpg', '')
                    old_file_path = os.path.join(root_dir, file)
                    new_file_path = os.path.join(root_dir, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed file {file} to {new_filename}')
                    file = new_filename

                full_filename = os.path.join(root_dir, file)
                fname = os.path.splitext(file)[0]
                prefixImagePath = os.path.join(root_dir, fname)
                found_image = False
                aImagePath = ""
                # ลองหานามสกุลของรูปที่รองรับ
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
                    temp_path = prefixImagePath + ext
                    if os.path.exists(temp_path):
                        aImagePath = temp_path
                        found_image = True
                        break

                if found_image:
                    IMG = cv.imread(aImagePath)
                    if IMG is not None:
                        heightIMG, widthIMG = IMG.shape[:2]
                        abs_path_to_save = os.path.join(path_dir, fname)
                        if not test_mode:
                            cv.imwrite(abs_path_to_save + '.' + SAVE_IMAGE_EXTENSION, IMG)
                            # Delete the old image file
                            os.remove(aImagePath)
                        numIMG += 1

                        # อ่านไฟล์ .anno แล้วแปลงข้อมูล
                        write_text = ""
                        with open(full_filename, "r") as anno_file:
                            lines = anno_file.readlines()
                            countLine = 0
                            for line in lines:
                                countLine += 1
                                xywh_str = re.split(r'\t+', line.strip())
                                if len(xywh_str) == 5:
                                    if xywh_str[0] in dictLabel:
                                        try:
                                            xPos = int(xywh_str[1])
                                            yPos = int(xywh_str[2])
                                            wPos = int(xywh_str[3])
                                            hPos = int(xywh_str[4])
                                            xywh = cvRect([xPos, yPos, wPos, hPos])
                                            yolo_line = makeLabelYOLO(xywh, xywh_str[0], widthIMG, heightIMG)
                                            write_text += yolo_line + "\n"
                                        except Exception as e:
                                            print(f"Error converting line {countLine} in {full_filename}: {e}")
                                    else:
                                        print(f"Unknown Label {xywh_str[0]} in line {countLine} at {full_filename}")
                                else:
                                    print(f"Format error in {full_filename}")
                        if not test_mode:
                            with open(abs_path_to_save + ".txt", "w") as f:
                                f.write(write_text)
                            # Delete the old .anno file
                            os.remove(full_filename)
                    else:
                        print("Couldn't open image:", aImagePath)
                else:
                    print("Image not found for", full_filename)
    print("Finished processing. Total images processed:", numIMG)
    if test_mode:
        print("Test Mode enabled: No files were written to disk.")

# ฟังก์ชันสำหรับแสดงหน้า Loading
def show_loading_screen():
    global loading_root, progress_var, gif_label, frames, frame_count

    loading_root = tk.Toplevel(root)
    loading_root.title("Processing...")
    loading_root.geometry("400x200")
    loading_root.resizable(False, False)
    ttk.Label(loading_root, text="Processing files...", font=("Arial", 14)).pack(pady=20)
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(loading_root, variable=progress_var, maximum=100)
    progress_bar.pack(pady=10, fill="x", padx=20)

    # Load and resize GIF
    gif_path = "loading.gif"  # Replace with your GIF path
    gif = Image.open(gif_path)
    resized_width, resized_height = 200, 100  # Adjust to fit the program window
    frames = [
        ImageTk.PhotoImage(frame.copy().convert("RGBA").resize((resized_width, resized_height), Image.Resampling.LANCZOS))
        for frame in ImageSequence.Iterator(gif)
    ]
    frame_count = len(frames)

    gif_label = ttk.Label(loading_root)
    gif_label.pack(pady=10)

    def update(ind):
        frame = frames[ind]
        ind = (ind + 1) % frame_count
        gif_label.configure(image=frame)
        loading_root.after(100, update, ind)

    update(0)
    loading_root.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable closing

# ฟังก์ชันสำหรับซ่อนหน้า Loading
def hide_loading_screen():
    global loading_root
    loading_root.destroy()

# ปรับปรุงฟังก์ชัน convert_folder ให้รองรับการอัปเดต UI
def convert_folder_with_ui(path_dir, test_mode, rename_files):
    start_time = time.time()
    num_files = sum([len(files) for _, _, files in os.walk(path_dir) if any(f.lower().endswith('.anno') for f in files)])
    processed_files = 0

    def update_progress():
        nonlocal processed_files
        while processed_files < num_files:
            elapsed_time = time.time() - start_time
            remaining_files = num_files - processed_files
            estimated_time = (elapsed_time / processed_files) * remaining_files if processed_files > 0 else 0
            progress_var.set((processed_files / num_files) * 100)
            # Removed progress_label.config for estimated time
            time.sleep(0.5)

    # Start progress updater in a separate thread
    progress_thread = Thread(target=update_progress, daemon=True)
    progress_thread.start()

    # Modified convert_folder to increment processed_files
    numIMG = 0
    path_classes_txt = os.path.join(path_dir, "classes.txt")
    if not os.path.exists(path_classes_txt):
        with open(path_classes_txt, "w") as classesFile:
            for key in dictLabel:
                classesFile.write(key + "\n")
    
    for root_dir, dirs, files in os.walk(path_dir):
        for file in files:
            if file.lower().endswith('.anno'):
                # Logic for renaming files if enabled
                if rename_files and '.jpg' in file:
                    new_filename = file.replace('.jpg', '')
                    old_file_path = os.path.join(root_dir, file)
                    new_file_path = os.path.join(root_dir, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed file {file} to {new_filename}')
                    file = new_filename

                full_filename = os.path.join(root_dir, file)
                fname = os.path.splitext(file)[0]
                prefixImagePath = os.path.join(root_dir, fname)
                found_image = False
                aImagePath = ""
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
                    temp_path = prefixImagePath + ext
                    if os.path.exists(temp_path):
                        aImagePath = temp_path
                        found_image = True
                        break

                if found_image:
                    IMG = cv.imread(aImagePath)
                    if IMG is not None:
                        heightIMG, widthIMG = IMG.shape[:2]
                        abs_path_to_save = os.path.join(path_dir, fname)
                        if not test_mode:
                            cv.imwrite(abs_path_to_save + '.' + SAVE_IMAGE_EXTENSION, IMG)
                            os.remove(aImagePath)
                        numIMG += 1

                        write_text = ""
                        with open(full_filename, "r") as anno_file:
                            lines = anno_file.readlines()
                            countLine = 0
                            for line in lines:
                                countLine += 1
                                xywh_str = re.split(r'\t+', line.strip())
                                if len(xywh_str) == 5:
                                    if xywh_str[0] in dictLabel:
                                        try:
                                            xPos = int(xywh_str[1])
                                            yPos = int(xywh_str[2])
                                            wPos = int(xywh_str[3])
                                            hPos = int(xywh_str[4])
                                            xywh = cvRect([xPos, yPos, wPos, hPos])
                                            yolo_line = makeLabelYOLO(xywh, xywh_str[0], widthIMG, heightIMG)
                                            write_text += yolo_line + "\n"
                                        except Exception as e:
                                            print(f"Error converting line {countLine} in {full_filename}: {e}")
                                    else:
                                        print(f"Unknown Label {xywh_str[0]} in line {countLine} at {full_filename}")
                                else:
                                    print(f"Format error in {full_filename}")
                        if not test_mode:
                            with open(abs_path_to_save + ".txt", "w") as f:
                                f.write(write_text)
                            os.remove(full_filename)
                    else:
                        print("Couldn't open image:", aImagePath)
                else:
                    print("Image not found for", full_filename)
                processed_files += 1  # Increment processed_files here

    print("Finished processing. Total images processed:", numIMG)
    if test_mode:
        print("Test Mode enabled: No files were written to disk.")
    processed_files = num_files  # Ensure progress reaches 100% at the end

# ฟังก์ชันรันแปลงผ่าน UI
def run_conversion():
    path_dir = input_entry.get()
    test_mode = test_mode_var.get()
    rename_files = rename_files_var.get()
    
    if not os.path.isdir(path_dir):
        messagebox.showerror("Error", "Input folder is invalid!")
        return

    # ซ่อนหน้าหลักและแสดงหน้า Loading
    root.withdraw()
    show_loading_screen()

    # เรียกใช้ convert_folder_with_ui ใน Thread เพื่อไม่ให้ UI ค้าง
    def conversion_thread():
        try:
            convert_folder_with_ui(path_dir, test_mode, rename_files)
        finally:
            # ซ่อนหน้า Loading และแสดงหน้าหลัก
            hide_loading_screen()
            root.deiconify()
            messagebox.showinfo("Done", "Conversion completed!")

    Thread(target=conversion_thread, daemon=True).start()

# สร้าง UI ด้วย tkinter (แบบ modern ด้วย ttk)
root = tk.Tk()
root.title("YOLO Dataset Converter by ไอโบ")
root.geometry("600x350")  # Adjusted height for new checkbox
style = ttk.Style(root)
style.theme_use('clam')

frame = ttk.Frame(root, padding="20")
frame.pack(expand=True, fill="both")

# โซนเลือก Input Folder
ttk.Label(frame, text="Path Folder (contains .anno files):", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
input_entry = ttk.Entry(frame, width=50, font=("Arial", 12))
input_entry.grid(row=1, column=0, padx=(0, 10))
def select_input_folder():
    folder = filedialog.askdirectory(title="Select Input Folder")
    if folder:
        input_entry.delete(0, tk.END)
        input_entry.insert(0, folder)
ttk.Button(frame, text="Browse", command=select_input_folder).grid(row=1, column=1)

# Checkbox สำหรับ Test Mode
test_mode_var = tk.BooleanVar()
test_mode_check = ttk.Checkbutton(frame, text="Test Mode (do not write to disk)", variable=test_mode_var)
test_mode_check.grid(row=4, column=0, pady=(20,0), sticky="w")

# Checkbox สำหรับ Rename Files
rename_files_var = tk.BooleanVar()
rename_files_check = ttk.Checkbutton(frame, text="Rename .anno files (remove .jpg from name)", variable=rename_files_var)
rename_files_check.grid(row=5, column=0, pady=(10,0), sticky="w")

# ปุ่มเริ่มการแปลง
ttk.Button(frame, text="Start Conversion", command=run_conversion).grid(row=6, column=0, pady=(20,0))

root.mainloop()
