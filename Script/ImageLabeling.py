import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Class สำหรับการทำงานกับ annotation และการ crop รูป
class ImageLabeler:
    def __init__(self, master):
        # Load settings from setting.json
        self.load_settings()
        self.master = master
        self.master.title("Image Labeling Tool")
        self.master.configure(bg=self.settings.get("background_color", "#2e2e2e"))
        self.master.geometry("1200x800")
        
        # Update default font to support Thai
        self.settings["font"] = "Tahoma"

        # โหลด label จากไฟล์ label_list.json หรือสร้างใหม่ถ้าไม่มี
        self.label_file = "label_list.json"
        self.labels = self.load_labels()
        self.current_label = self.labels[0]["nameLabel"]

        # Load label colors from settings or assign default colors
        self.label_colors = self.settings.get("label_colors", {})
        for label in self.labels:
            if label["nameLabel"] not in self.label_colors:
                self.label_colors[label["nameLabel"]] = self.generate_random_color()
        # Save updated label colors back to settings
        self.settings["label_colors"] = self.label_colors
        self.save_settings()

        # Load keyboard shortcuts from settings
        self.shortcuts = self.settings.get("shortcuts", {
            "confirm_crop": ["Return", "space"],
            "cancel_crop": ["Escape"],
            "remove_all_labels": ["Delete"],
            "remove_oldest_label": ["p"],
            "remove_newest_label": ["n"],
            "cycle_label": ["w"],
            "next_image": ["Left", "Up", "a"],
            "prev_image": ["Right", "Down", "d"],
            "quit": ["q"]
        })
        # Save updated shortcuts back to settings if not present
        self.settings["shortcuts"] = self.shortcuts
        self.save_settings()

        # ข้อมูลสำหรับภาพและ annotation
        self.image_paths = []
        self.annotations = {}  # เก็บ annotation ของแต่ละไฟล์ในรูปแบบ {image_path: [ {label, rect}, ... ]}
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        self.scale = 1.0  # อัตราส่วนการ resize รูป
        self.rect_id = None
        self.start_x = None
        self.start_y = None

        # สร้าง top frame สำหรับปุ่มควบคุม
        self.top_frame = tk.Frame(master, bg=self.settings.get("background_color", "#2e2e2e"))
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        # ปุ่มเลือก label
        tk.Label(self.top_frame, text="Label:", bg=self.settings.get("background_color", "#2e2e2e"), fg="white", font=(self.settings.get("font", "Tahoma"), 12)).pack(side=tk.LEFT, padx=5)
        self.label_var = tk.StringVar(value=self.current_label)
        self.label_menu = tk.OptionMenu(self.top_frame, self.label_var, *[lbl["nameLabel"] for lbl in self.labels], command=self.change_label)
        self.label_menu.config(bg=self.settings.get("button_color", "#444444"), fg="white", activebackground=self.settings.get("active_button_color", "#666666"), activeforeground="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.label_menu["menu"].config(bg=self.settings.get("button_color", "#444444"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.label_menu.pack(side=tk.LEFT, padx=5)

        # ปุ่ม previous, next
        self.prev_btn = tk.Button(self.top_frame, text="Previous", command=self.prev_image, bg=self.settings.get("button_color", "#444444"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = tk.Button(self.top_frame, text="Next", command=self.next_image, bg=self.settings.get("button_color", "#444444"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # ปุ่มบันทึก annotation
        self.save_btn = tk.Button(self.top_frame, text="Save Annotation", command=self.save_annotation, bg=self.settings.get("button_color", "#444444"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # ปุ่มลบกล่องล่าสุด
        self.delete_btn = tk.Button(self.top_frame, text="Delete Last Box", command=self.delete_last_box, bg=self.settings.get("button_color", "#444444"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.delete_btn.pack(side=tk.LEFT, padx=5)

        # แสดงชื่อไฟล์ที่มุมบนซ้าย
        self.image_name_label = tk.Label(self.top_frame, text="", bg=self.settings.get("background_color", "#2e2e2e"), fg="white", font=(self.settings.get("font", "Tahoma"), 12))
        self.image_name_label.pack(side=tk.LEFT, padx=10)

        # Canvas สำหรับแสดงภาพ
        self.canvas = tk.Canvas(master, bg="#1e1e1e")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Explicitly force focus to the main window and bind key events globally
        master.focus_force()
        master.update()  # Ensure the window is properly focused
        master.bind_all("<Key>", self.on_key_press)

        # โหลดภาพจากโฟลเดอร์ที่เลือก
        self.load_images()
        if self.image_paths:
            self.load_current_image()
        else:
            messagebox.showerror("Error", "No images found in selected folder.")
            master.quit()

        # อัพเดท canvas เมื่อหน้าต่าง resize
        self.master.bind("<Configure>", lambda event: self.show_image())

    def load_settings(self):
        settings_path = "setting.json"
        default_settings = {
            "background_color": "#2e2e2e",
            "button_color": "#444444",
            "active_button_color": "#666666",
            "font": "Arial",
            "label_colors": {
                "default": "lime",
                "crop_confirm": "purple"
            },
            "shortcuts": {
                "confirm_crop": ["Return", "space"],
                "cancel_crop": ["Escape"],
                "remove_all_labels": ["Delete"],
                "remove_oldest_label": ["p"],
                "remove_newest_label": ["n"],
                "cycle_label": ["w"],
                "next_image": ["Left", "Up", "a"],
                "prev_image": ["Right", "Down", "d"],
                "quit": ["q"]
            }
        }
        if not os.path.exists(settings_path):
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(default_settings, f, ensure_ascii=False, indent=4)
            self.settings = default_settings
        else:
            with open(settings_path, "r", encoding="utf-8") as f:
                self.settings = json.load(f)

    def load_labels(self):
        if not os.path.exists(self.label_file):
            default = [{"nameLabel": "PlasticBottle", "description": "ขวดพลาสติก"}]
            with open(self.label_file, "w", encoding="utf-8") as f:
                json.dump(default, f, ensure_ascii=False, indent=4)
            return default
        else:
            with open(self.label_file, "r", encoding="utf-8") as f:
                return json.load(f)

    def load_images(self):
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีไฟล์ภาพ")
        if not folder:
            self.master.quit()
        valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                            if os.path.splitext(f)[1].lower() in valid_ext]
        self.image_paths.sort()

    def load_current_image(self):
        path = self.image_paths[self.current_index]
        self.image_name_label.config(text=os.path.basename(path))
        self.current_image = cv2.imread(path)
        if self.current_image is None:
            return
        # สร้าง entry ใน annotations ถ้ายังไม่มี
        self.annotations.setdefault(path, [])
        self.show_image()

    def show_image(self):
        if self.current_image is None:
            return
        # รับขนาด canvas ปัจจุบัน
        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 600
        h, w = self.current_image.shape[:2]
        self.scale = min(canvas_width/w, canvas_height/h)
        new_w, new_h = int(w*self.scale), int(h*self.scale)
        resized = cv2.resize(self.current_image, (new_w, new_h))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        self.canvas.delete("all")
        # แสดงภาพตรงกลาง canvas
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
        # วาด annotation ที่มีอยู่ (scale ปรับให้พอดี)
        for annot in self.annotations[self.image_paths[self.current_index]]:
            rx, ry, rw, rh = annot["rect"]
            rx, ry, rw, rh = int(rx*self.scale), int(ry*self.scale), int(rw*self.scale), int(rh*self.scale)
            label_color = self.label_colors.get(annot["label"], "lime")
            self.canvas.create_rectangle(rx, ry, rx+rw, ry+rh, outline=label_color, width=2)
            self.canvas.create_text(rx + rw // 2, ry - 10, anchor=tk.CENTER, text=annot["label"], fill=label_color, font=(self.settings.get("font", "Tahoma"), 12))
        # Display current label info at bottom-left: "namelabel : description" in purple
        label_text = f"{self.current_label} : {self.get_current_label_description()}"
        text_bbox = self.canvas.bbox(self.canvas.create_text(10, canvas_height-30, anchor="sw", text=label_text, fill=self.settings["label_colors"].get("crop_confirm", "red"), font=(self.settings.get("font", "Tahoma"), 8)))
        if text_bbox:
            self.canvas.create_rectangle(text_bbox, outline="")
            self.canvas.create_text(10, canvas_height-30, anchor="sw", text=label_text, fill=self.settings["label_colors"].get("crop_confirm", "red"), font=(self.settings.get("font", "Tahoma"), 8))
        # Display image name at bottom-left corner
        image_name = os.path.basename(self.image_paths[self.current_index])
        name_bbox = self.canvas.bbox(self.canvas.create_text(10, canvas_height - 10, anchor="sw", text=image_name, fill="white", font=(self.settings.get("font", "Tahoma"), 8)))
        if name_bbox:
            self.canvas.create_rectangle(name_bbox, outline="")
            self.canvas.create_text(10, canvas_height - 10, anchor="sw", text=image_name, fill="white", font=(self.settings.get("font", "Tahoma"), 8))

    def change_label(self, value):
        self.current_label = value

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        if self.rect_id:
            x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
            rect = [int(min(x1, x2)/self.scale), int(min(y1, y2)/self.scale),
                    int(abs(x2-x1)/self.scale), int(abs(y2-y1)/self.scale)]
            # ตรวจสอบพื้นที่เล็กเกินไป
            if rect[2]*rect[3] < 500:
                self.canvas.delete(self.rect_id)
            else:
                path = self.image_paths[self.current_index]
                self.annotations[path].append({"label": self.current_label, "rect": rect})
            self.rect_id = None
            self.show_image()

    def on_key_press(self, event):
        key = event.keysym
        c = event.char.lower() if event.char else ""
        # Check shortcuts and execute corresponding actions
        if key in self.shortcuts["confirm_crop"] or c in self.shortcuts["confirm_crop"]:
            self.confirm_crop()
        elif key in self.shortcuts["confirm_label"] or c in self.shortcuts["confirm_label"]:
            self.save_annotation()
        elif key in self.shortcuts["cancel_crop"] or c in self.shortcuts["cancel_crop"]:
            self.cancel_crop()
        elif key in self.shortcuts["remove_all_labels"] or c in self.shortcuts["remove_all_labels"]:
            self.remove_all_labels()
        elif key in self.shortcuts["remove_oldest_label"] or c in self.shortcuts["remove_oldest_label"]:
            self.remove_oldest_label()
        elif key in self.shortcuts["remove_newest_label"] or c in self.shortcuts["remove_newest_label"]:
            self.delete_last_box()
        elif key in self.shortcuts["cycle_label"] or c in self.shortcuts["cycle_label"]:
            self.cycle_label()
        elif key in self.shortcuts["next_image"] or c in self.shortcuts["next_image"]:
            self.next_image()
        elif key in self.shortcuts["prev_image"] or c in self.shortcuts["prev_image"]:
            self.prev_image()
        elif key in self.shortcuts["quit"] or c in self.shortcuts["quit"]:
            self.master.quit()
        elif key == "z" or c == "z":
            self.ai_generate_box()

    def confirm_crop(self):
        if self.rect_id:
            coords = self.canvas.coords(self.rect_id)
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                rect = [int(min(x1, x2)/self.scale), int(min(y1, y2)/self.scale),
                        int(abs(x2-x1)/self.scale), int(abs(y2-y1)/self.scale)]
                if rect[2]*rect[3] >= 500:
                    path = self.image_paths[self.current_index]
                    self.annotations[path].append({"label": self.current_label, "rect": rect})
                self.canvas.delete(self.rect_id)
                self.rect_id = None
                self.show_image()

    def cancel_crop(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def remove_all_labels(self):
        path = self.image_paths[self.current_index]
        self.annotations[path] = []
        self.show_image()

    def remove_oldest_label(self):
        path = self.image_paths[self.current_index]
        if self.annotations[path]:
            self.annotations[path].pop(0)
            self.show_image()

    def cycle_label(self):
        # Cycle to next label in self.labels list
        idx = next((i for i, lbl in enumerate(self.labels) if lbl["nameLabel"] == self.current_label), 0)
        next_idx = (idx + 1) % len(self.labels)
        self.current_label = self.labels[next_idx]["nameLabel"]
        self.label_var.set(self.current_label)

    def get_current_label_description(self):
        for lbl in self.labels:
            if lbl["nameLabel"] == self.current_label:
                return lbl.get("description", "")
        return ""

    def show_temp_message(self, message, duration=3000):
        """Display a temporary message at the bottom-right corner of the canvas with a background."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        text_id = self.canvas.create_text(canvas_width - 10, canvas_height - 10, anchor="se", 
                                          text=message, fill="#FFBEA3", font=(self.settings.get("font", "Tahoma"), 24))
        bbox = self.canvas.bbox(text_id)  # Get bounding box of the text
        if bbox:
            rect_id = self.canvas.create_rectangle(bbox, fill="#333333", outline="")  # Add background rectangle
            self.canvas.tag_lower(rect_id, text_id)  # Ensure the rectangle is behind the text
        self.master.after(duration, lambda: (self.canvas.delete(text_id), self.canvas.delete(rect_id)))

    def save_annotation(self):
        path = self.image_paths[self.current_index]
        anno_path = path + ".anno"
        try:
            with open(anno_path, "w", encoding="utf-8") as f:
                for annot in self.annotations[path]:
                    rx, ry, rw, rh = annot["rect"]
                    f.write(f"{annot['label']}\t{rx}\t{ry}\t{rw}\t{rh}\n")
            self.show_temp_message("Annotation saved successfully!")  # Show temporary message
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def delete_last_box(self):
        path = self.image_paths[self.current_index]
        if self.annotations[path]:
            self.annotations[path].pop()
            self.show_image()

    def next_image(self):
        if self.current_index < len(self.image_paths)-1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def generate_random_color(self):
        import random
        return f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"

    def save_settings(self):
        settings_path = "setting.json"
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)

    def ai_generate_box(self):
        """ใช้โมเดลจับขยะแบบ segmentation ที่แม่นยำ เพื่อสร้าง bounding box ที่ครอบคลุมวัตถุภายในกรอบ label"""
        if self.current_image is None:
            self.show_temp_message("No image loaded for AI processing.")
            return

        path = self.image_paths[self.current_index]
        if not self.annotations.get(path):
            self.show_temp_message("No annotations available to crop the image.")
            return

        # หา annotation ที่ตรงกับ label ปัจจุบัน
        target_annot = next((annot for annot in self.annotations[path] if annot["label"] == self.current_label), None)
        if target_annot is None:
            self.show_temp_message("No bounding box found for the current label.")
            return

        try:
            # โหลดโมเดลจับขยะที่เทรนเฉพาะ
            model = YOLO('yolov8m-seg.pt')
            
            # กำหนดกรอบ crop ตาม annotation และเก็บ offset ไว้
            x_offset, y_offset, w_rect, h_rect = target_annot["rect"]
            img_crop = self.current_image[y_offset:y_offset+h_rect, x_offset:x_offset+w_rect]
            
            # รัน detection บนภาพที่ crop มา โดยใช้ threshold ที่สูงขึ้นเพื่อความแม่นยำ
            results = model(img_crop, conf=0.7)

            # เคลียร์ bounding boxes เก่า
            self.annotations[path] = []

            # ประมวลผลผลลัพธ์ segmentation
            for result in results:
                for mask, conf in zip(result.masks.data, result.boxes.conf):
                    if conf < 0.7:
                        continue

                    # แปลง mask เป็น numpy array (binary mask) แล้วหา contour
                    mask_np = (mask.cpu().numpy().astype('uint8')) * 255
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue

                    # เลือก contour ที่มีพื้นที่มากที่สุด
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)

                    # คำนวณ bounding box แบบ relative ต่อ full image
                    x_full = x + x_offset
                    y_full = y + y_offset

                    # ปรับ bounding boxให้แน่ใจว่าอยู่ภายในกรอบ label เดิม
                    x_final = max(x_offset, x_full)
                    y_final = max(y_offset, y_full)
                    x_end = min(x_offset + w_rect, x_full + w)
                    y_end = min(y_offset + h_rect, y_full + h)
                    w_final = x_end - x_final
                    h_final = y_end - y_final

                    # ตรวจสอบให้แน่ใจว่า bounding box มีขนาดที่เหมาะสม
                    if w_final * h_final >= 500:
                        self.annotations[path].append({"label": self.current_label, "rect": [x_final, y_final, w_final, h_final]})
            
            self.show_image()
            self.show_temp_message("Bounding boxes updated with trash_detector_seg.")

        except Exception as e:
            messagebox.showerror("AI Error", f"Failed to adjust boxes: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()
