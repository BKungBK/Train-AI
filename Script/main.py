import os
import json
import random
import shutil
import threading
import numpy as np
from pathlib import Path
import customtkinter as ctk  # Modern GUI library
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import torch
import optuna  # newly added import
from sklearn.preprocessing import label_binarize

# ----------------- Pipeline Functions -----------------
def prepare_dataset(dataset_dir, output_dir):
    """
    รวม dataset จากโฟลเดอร์ที่ให้มา แล้วแบ่งเป็น train/val/test ด้วยสัดส่วน 80/10/10
    คัดลอกไฟล์จาก dataset_dir ไปไว้ใน output_dir ที่มีโครงสร้าง train/val/test พร้อม images และ labels
    """
    output_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    all_images = list(Path(dataset_dir).rglob("*.png"))
    if not all_images:
        raise Exception("ไม่พบไฟล์ .png ใน dataset ที่เลือก")
    random.shuffle(all_images)
    total = len(all_images)
    train_split = int(0.8 * total)
    val_split = int(0.1 * total)

    splits = {
        'train': all_images[:train_split],
        'val': all_images[train_split:train_split + val_split],
        'test': all_images[train_split + val_split:]
    }

    for split, images in splits.items():
        for img_path in images:
            label_path = str(img_path).replace('images', 'labels').replace('.png', '.txt')
            if not os.path.exists(label_path):
                continue  # ถ้าไม่เจอ label ก็ข้ามไป
            shutil.copy(img_path, output_dir / split / 'images' / img_path.name)
            shutil.copy(label_path, output_dir / split / 'labels' / Path(label_path).name)
    print("Dataset prepared successfully!")

def create_data_yaml(output_dir, class_txt, yaml_path):
    """
    สร้างไฟล์ data.yaml สำหรับ YOLO โดยอ่าน class จาก class.txt
    """
    with open(class_txt, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    # เขียน data.yaml ในรูปแบบ YAML พื้นฐาน
    content = (
        f"path: {output_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names: {names}\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(content)
    print("data.yaml created!")

def train_yolo(yaml_path, epochs=80, batch_size=13, lr0=0.00011952532989826581, weight_decay=0.0001471781529795871):
    """
    เทรน YOLO ด้วย transfer learning พร้อมใช้ hyperparameters ที่ดีที่สุดตามผล Optuna
    """
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = YOLO("best.pt")

    results = model.train(
        task="detect",
        mode="train",
        data=yaml_path,
        epochs=80,                        # เพิ่ม epochs เพราะใช้ early stopping อยู่แล้ว
        imgsz=512,                        # ขยับขึ้นมาหน่อยจาก 416 ให้ได้ detail มากขึ้น (ถ้า VRAM ไหว)
        batch=8,                          # ถ้า GPU ไหว เพิ่มเป็น 8 เพื่อเสถียรภาพการฝึก
        device=0,
        lr0=1e-4,                         # ปรับให้ดู standard ขึ้น แต่ใกล้ค่าที่คุณตั้งไว้
        lrf=0.1,
        momentum=0.937,
        weight_decay=1.5e-4,             # ค่าที่คุณใช้อยู่ดีอยู่แล้ว
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=1.0,                      # เพิ่มการหมุนนิดหน่อย เผื่อเพิ่ม variety
        translate=0.1,
        scale=0.5,
        shear=0.01,
        perspective=0.001,               # ขยับขึ้นเล็กน้อยเพื่อให้เกิด variety มากขึ้น
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.75,                     # ลดมาจาก 0.8 นิดนึงให้เนียนขึ้น
        mixup=0.05,                      # ลดลงอีก เพราะบางครั้ง mixup รบกวน learning
        copy_paste=0.1,
        patience=30,                     # ลด patience ลง เพื่อหยุดเร็วขึ้นถ้าระบบเริ่ม overfit
        amp=True,
        verbose=True
    )

    return model


def evaluate_model(model, yaml_path):
    """
    ประเมิน performance ของโมเดลโดยใช้ test set จาก data.yaml
    """
    metrics = model.val(data=yaml_path, split='test', verbose=True)
    print("Evaluation complete!")
    print(metrics)
    return metrics

# New objective function for hyperparameter tuning using Optuna
def objective(trial, yaml_path):
    # Suggest hyperparameters
    suggested_lr0 = trial.suggest_float("lr0", 0.0001, 0.01, log=True)
    suggested_weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.001, log=True)
    suggested_epochs = trial.suggest_int("epochs", 10, 50)
    suggested_batch_size = trial.suggest_int("batch_size", 4, 16)

    try:
        model = train_yolo(
            yaml_path,
            epochs=suggested_epochs,
            batch_size=suggested_batch_size,
            lr0=suggested_lr0,
            weight_decay=suggested_weight_decay,
        )

        metrics = evaluate_model(model, yaml_path)
        score = metrics.box.map  # mAP@0.5:0.95
        print(f"Trial score (mAP): {score}")
        return score

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Fallback score in case of failure


def plot_confusion_matrix(preds, labels, class_names):
    """
    วาด confusion matrix โดยใช้ seaborn
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def detect_and_analyze(model, test_images_dir, class_names):
    """
    ตรวจจับวัตถุใน test images แล้ววิเคราะห์ผล โดยคำนวณ mAP (Mean Average Precision),
    พร้อมทั้งแสดงค่า Precision, Recall (และ F1-score) ใน Classification Report และวาด Confusion Matrix แบบ Multi-class

    ฟังก์ชันนี้สมมติว่าแต่ละภาพมี label เดียวในไฟล์ .txt
    """
    test_images_dir = Path(test_images_dir)
    test_images = list(test_images_dir.glob("*.png"))
    
    preds = []
    trues = []
    confidences = []  # เก็บค่า confidence ของ prediction

    for img_path in test_images:
        # ดำเนินการตรวจจับ
        results = model(img_path, verbose=False)
        label_file = str(img_path).replace('images', 'labels').replace('.png', '.txt')
        
        true_cls = None
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    # สมมติว่า label อยู่บรรทัดแรกและเป็นตัวเลข
                    true_cls = int(f.readline().strip().split()[0])
            except Exception as e:
                print(f"Error reading label for {img_path}: {e}")
        
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # เลือก box ที่มี confidence สูงสุด (สำหรับ demo แบบ single prediction ต่อภาพ)
            boxes = results[0].boxes
            # sort by confidence (high to low)
            boxes = sorted(boxes, key=lambda box: float(box.conf), reverse=True)
            best_box = boxes[0]
            preds.append(int(best_box.cls))
            confidences.append(float(best_box.conf))
        else:
            preds.append(-1)  # หากไม่ตรวจจับได้ ให้ใช้ -1 แทน
            confidences.append(0.0)
        
        if true_cls is not None:
            trues.append(true_cls)
    
    # แปลงข้อมูลเป็น numpy arrays
    preds = np.array(preds)
    trues = np.array(trues)

    # คำนวณ mAP โดยใช้ average_precision_score จาก sklearn
    # จำเป็นต้องใช้ one-hot encoding สำหรับ true labels
    classes = list(range(len(class_names)))
    y_true = label_binarize(trues, classes=classes)
    
    # สำหรับ score ของแต่ละ class เราสร้าง matrix ที่มี confidence ในตำแหน่งที่ prediction นั้นออกมา
    y_score = np.zeros((len(preds), len(class_names)))
    for i, (pred, conf) in enumerate(zip(preds, confidences)):
        if pred in classes:
            y_score[i, pred] = conf
    # คำนวณ mAP แบบ macro averaging
    mAP = average_precision_score(y_true, y_score, average="macro")
    print(f"\nMean Average Precision (mAP): {mAP:.4f}\n")
    
    # พิมพ์ Classification Report
    print("Classification Report:")
    print(classification_report(trues, preds, target_names=class_names, zero_division=0))
    
    # วาด Confusion Matrix
    plot_confusion_matrix(preds, trues, class_names)

# ----------------- GUI Part ด้วย CustomTkinter -----------------
class YOLOTrainerGUI(ctk.CTk):
    SETTINGS_FILE = "settings.json"

    def __init__(self):
        super().__init__()
        self.title("AI Trainer GUI By BKung")
        self.geometry("700x400")
        self.resizable(False, False)
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color="#6C567B")

        self.settings = self.load_settings()

        # ตัวแปรเก็บ paths และสถานะ
        self.dataset_path = ctk.StringVar(value=self.settings.get("dataset_path", ""))
        self.class_file = ctk.StringVar(value=self.settings.get("class_file", ""))
        self.log_text = ctk.StringVar()
        self.pipeline_running = False
        self.fast_mode = ctk.BooleanVar(value=self.settings.get("fast_mode", False))
        self.prepare_dataset = ctk.BooleanVar(value=True)  # New variable

        self.create_widgets()

    def create_widgets(self):
        font = tuple(self.settings.get("font", ["Arial", 12]))
        button_settings = {
            "fg_color": self.settings.get("button_fg_color", "#3B8ED0"),
            "hover_color": self.settings.get("button_hover_color", "#36719F"),
            "border_color": self.settings.get("button_border_color", "#2C6E91"),
            "text_color": self.settings.get("button_text_color", "#FFFFFF"),
            "font": font,
        }
        label_settings = {
            "text_color": self.settings.get("label_text_color", "#FFFFFF"),
            "font": font,
        }
        entry_settings = {
            "fg_color": self.settings.get("entry_fg_color", "#C06C84"),
            "text_color": self.settings.get("entry_text_color", "#FFFFFF"),
            "state": "disabled",
            "font": font,
        }
        progress_bar_settings = {
            "fg_color": self.settings.get("progress_fg_color", "#2B2B2B"),
            "border_color": self.settings.get("progress_border_color", "#FFFFFF"),
            "progress_color": self.settings.get("progress_progress_color", "#3B8ED0"),
        }

        dataset_frame = ctk.CTkFrame(self)
        dataset_frame.configure(fg_color="#C06C84", border_color="#F9F9F9")
        dataset_frame.pack(pady=(15, 0))
        ctk.CTkLabel(dataset_frame, text="Dataset Folder:", **label_settings).pack(side="left", padx=5)
        ctk.CTkEntry(dataset_frame, textvariable=self.dataset_path, width=300, **entry_settings).pack(side="left", padx=5)
        ctk.CTkButton(dataset_frame, text="Browse", command=self.browse_dataset, **button_settings).pack(side="left", padx=5)

        class_file_frame = ctk.CTkFrame(self)
        class_file_frame.configure(fg_color="#C06C84", border_color="#F9F9F9")
        class_file_frame.pack(pady=(15, 0))
        ctk.CTkLabel(class_file_frame, text="Class File (class.txt):", **label_settings).pack(side="left", padx=5)
        ctk.CTkEntry(class_file_frame, textvariable=self.class_file, width=300, **entry_settings).pack(side="left", padx=5)
        ctk.CTkButton(class_file_frame, text="Browse", command=self.browse_class_file, **button_settings).pack(side="left", padx=5)

        ctk.CTkCheckBox(self, text="Enable Fast Training Mode (Quick Test)", fg_color="#C7DCA7", border_color="#F9F9F9", variable=self.fast_mode, font=font).pack(pady=10)
        ctk.CTkCheckBox(self, text="Prepare dataset", variable=self.prepare_dataset, font=font).pack(pady=5)  # New checkbox
        ctk.CTkButton(self, text="Start Training", command=self.start_training_thread, width=200, **button_settings).pack(pady=10)
        ctk.CTkButton(self, text="Auto Tune", command=self.start_auto_tuning_thread, width=200, **button_settings).pack(pady=5)
        ctk.CTkButton(self, text="Settings", command=self.open_settings_window, **button_settings).pack(pady=5)
        ctk.CTkLabel(self, textvariable=self.log_text, wraplength=600, justify="left", **label_settings).pack(pady=5)
        

        self.progress_bar = ctk.CTkProgressBar(self, mode="indeterminate", width=400, **progress_bar_settings)
        self.progress_bar.pack(pady=5)
        

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.dataset_path.set(path)

    def browse_class_file(self):
        path = filedialog.askopenfilename(title="Select class.txt", filetypes=[("Text Files", "*.txt")])
        if path:
            self.class_file.set(path)

    def start_training_thread(self):
        if not self.dataset_path.get() or not self.class_file.get():
            messagebox.showerror("Error", "กรุณาเลือกทั้ง Dataset Folder และ class.txt ให้ครบ")
            return
        self.log_text.set("Starting training pipeline...")
        self.pipeline_running = True
        self.progress_bar.start()
        threading.Thread(target=self.training_pipeline, daemon=True).start()

    def start_auto_tuning_thread(self):
        if not self.dataset_path.get() or not self.class_file.get():
            messagebox.showerror("Error", "กรุณาเลือกทั้ง Dataset Folder และ class.txt ให้ครบ")
            return
        self.log_text.set("Starting hyperparameter tuning...")
        self.pipeline_running = True
        self.progress_bar.start()
        threading.Thread(target=self.auto_tune_pipeline, daemon=True).start()

    def stop_loading_effect(self):
        self.pipeline_running = False
        self.progress_bar.stop()

    def training_pipeline(self):
        try:
            # Save settings ก่อนเริ่ม
            self.save_settings()

            # Determine output_dir based on prepare_dataset option
            dataset_dir = self.dataset_path.get()
            if self.prepare_dataset.get():
                self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ขั้นตอนที่ 1: เตรียมชุดข้อมูล...\"")
                output_dir = os.path.join(os.getcwd(), "dataset_split")
                prepare_dataset(dataset_dir, output_dir)
            else:
                self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ใช้ dataset ที่มีโครงสร้างครบแล้ว\"")
                output_dir = dataset_dir

            # Step 2: Create data.yaml
            self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ขั้นตอนที่ 2: สร้างไฟล์ data.yaml...\"")
            yaml_path = os.path.join(output_dir, "data.yaml")
            create_data_yaml(output_dir, self.class_file.get(), yaml_path)

            # Step 3: Train YOLO
            self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ขั้นตอนที่ 3: ฝึกโมเดล YOLO...\"")
            if self.fast_mode.get():
                model = train_yolo(yaml_path, epochs=1, batch_size=1)
                self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"โหมดฝึกแบบเร็ว: ฝึกเสร็จแล้ว!\"")
            else:
                model = train_yolo(yaml_path, epochs=500, batch_size=4)  # ปรับ batch_size ให้เหมาะสมสำหรับ VRAM 8GB
                self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ฝึกเสร็จแล้ว!\"")

            # Step 4: Evaluate model
            self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ขั้นตอนที่ 4: ประเมินผลโมเดล...\"")
            evaluate_model(model, yaml_path)

            # Step 5: Detect and Analyze test set
            self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"ขั้นตอนที่ 5: ตรวจจับและวิเคราะห์ชุดทดสอบ...\"")
            with open(self.class_file.get(), 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
            detect_and_analyze(model, os.path.join(output_dir, "test", "images"), class_names)
            self.log_text.set("(\ _ /) \n(˶• ༝ •˶)  \n( > \"การตรวจจับและการวิเคราะห์เสร็จสิ้น!\"")

            self.stop_loading_effect()
            self.log_text.set("สำเร็จ กระบวนการเสร็จสมบูรณ์!\n𝙔𝘼𝙔!ーーーーー\n    ☆  *    .      ☆\n        . ∧＿∧    ∩    * ☆\n*  ☆ ( ・∀・)/ .\n    .  ⊂         ノ* ☆\n☆ * (つ ノ  .☆\n         (ノ")
            messagebox.showinfo("สำเร็จ", "กระบวนการเสร็จสมบูรณ์!")
        except Exception as e:
            self.stop_loading_effect()
            self.log_text.set(f"Error occurred: {str(e)}")
            messagebox.showerror("Error", str(e))

    def auto_tune_pipeline(self):
        try:
            # Save settings
            self.save_settings()
            
            # Step 1: เตรียมชุดข้อมูล
            dataset_dir = self.dataset_path.get()
            if self.prepare_dataset.get():
                self.log_text.set("ขั้นตอนที่ 1: เตรียมชุดข้อมูลสำหรับ Auto Tune...")
                output_dir = os.path.join(os.getcwd(), "dataset_split")
                prepare_dataset(dataset_dir, output_dir)
            else:
                self.log_text.set("ขั้นตอนที่ 1: ใช้ dataset ที่มีโครงสร้างครบแล้วสำหรับ Auto Tune...")
                output_dir = dataset_dir
            
            # Step 2: สร้างไฟล์ data.yaml สำหรับ Auto Tune
            self.log_text.set("ขั้นตอนที่ 2: สร้างไฟล์ data.yaml สำหรับ Auto Tune...")
            yaml_path = os.path.join(output_dir, "data.yaml")
            create_data_yaml(output_dir, self.class_file.get(), yaml_path)
            
            # Step 3: รัน hyperparameter tuning ด้วย Optuna
            self.log_text.set("ขั้นตอนที่ 3: รัน Auto Tune ด้วย Optuna...")
            # สร้าง study โดยเพิ่ม pruner เพื่อลดเวลาใน trial ที่ผลลัพธ์ไม่ดี
            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
            # เพิ่มจำนวน n_trials ให้มากขึ้น (ปรับค่าได้ตามเวลาที่มี)
            study.optimize(lambda trial: objective(trial, yaml_path), n_trials=10)
            
            best_params = study.best_params
            best_value = study.best_value
            self.log_text.set(f"Best tuning result: mAP={best_value}\nBest Params: {best_params}")
            messagebox.showinfo("Auto Tune Complete", f"Best mAP: {best_value}\nParameters: {best_params}")
            
            self.progress_bar.stop()
            self.pipeline_running = False

        except Exception as e:
            self.progress_bar.stop()
            self.pipeline_running = False
            self.log_text.set(f"Error occurred during tuning: {str(e)}")
            messagebox.showerror("Error", str(e))


    def save_settings(self):
        """ Save settings to settings.json in a structured format """
        settings = {
            "dataset_path": self.dataset_path.get(),
            "class_file": self.class_file.get(),
            "fast_mode": self.fast_mode.get(),
            "appearance_mode": ctk.get_appearance_mode(),
            "font": self.settings.get("font", ["Arial", 12]),
            "button_fg_color": self.settings.get("button_fg_color", "#3B8ED0"),
            "button_hover_color": self.settings.get("button_hover_color", "#36719F"),
            "button_border_color": self.settings.get("button_border_color", "#2C6E91"),
            "button_text_color": self.settings.get("button_text_color", "#FFFFFF"),
            "label_text_color": self.settings.get("label_text_color", "#FFFFFF"),
            "entry_fg_color": self.settings.get("entry_fg_color", "#2B2B2B"),
            "entry_text_color": self.settings.get("entry_text_color", "#FFFFFF"),
            "progress_fg_color": self.settings.get("progress_fg_color", "#2B2B2B"),
            "progress_border_color": self.settings.get("progress_border_color", "#FFFFFF"),
            "progress_progress_color": self.settings.get("progress_progress_color", "#3B8ED0"),
            "task": self.settings.get("task", "detect"),
            "mode": self.settings.get("mode", "train"),
            "epochs": self.settings.get("epochs", 50),
            "imgsz": self.settings.get("imgsz", 640),
            "batch_size": self.settings.get("batch_size", 16),
            "verbose": self.settings.get("verbose", True),
            "device": self.settings.get("device", "cpu"),
            "lr0": self.settings.get("lr0", 0.01),
            "lrf": self.settings.get("lrf", 0.1),
            "lr_scheduler": self.settings.get("lr_scheduler", "cosine")
        }
        with open(self.SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)  # Add indentation for readability

    def load_settings(self):
        if os.path.exists(self.SETTINGS_FILE):
            with open(self.SETTINGS_FILE, "r") as f:
                return json.load(f)
        return {}

    def open_settings_window(self):
        """ Open the settings window """
        settings_window = SettingsWindow(self)
        settings_window.grab_set()

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("500x650")
        self.resizable(False, False)

        # Center the window on the screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        # Load settings from parent
        self.settings = parent.settings

        # Variables for settings
        self.appearance_mode_var = ctk.StringVar(value=self.settings.get("appearance_mode", "System"))
        self.font_var = ctk.StringVar(value=self.settings.get("font", ["Arial", 12])[0])
        self.font_size_var = ctk.StringVar(value=str(self.settings.get("font", ["Arial", 12])[1]))
        self.button_fg_color_var = ctk.StringVar(value=self.settings.get("button_fg_color", "#3B8ED0"))
        self.button_hover_color_var = ctk.StringVar(value=self.settings.get("button_hover_color", "#36719F"))
        self.button_border_color_var = ctk.StringVar(value=self.settings.get("button_border_color", "#2C6E91"))
        self.button_text_color_var = ctk.StringVar(value=self.settings.get("button_text_color", "#FFFFFF"))
        self.label_text_color_var = ctk.StringVar(value=self.settings.get("label_text_color", "#FFFFFF"))
        self.entry_fg_color_var = ctk.StringVar(value=self.settings.get("entry_fg_color", "#2B2B2B"))
        self.entry_text_color_var = ctk.StringVar(value=self.settings.get("entry_text_color", "#FFFFFF"))
        self.progress_fg_color_var = ctk.StringVar(value=self.settings.get("progress_fg_color", "#2B2B2B"))
        self.progress_border_color_var = ctk.StringVar(value=self.settings.get("progress_border_color", "#FFFFFF"))
        self.progress_progress_color_var = ctk.StringVar(value=self.settings.get("progress_progress_color", "#3B8ED0"))

        # Variables for model settings
        self.task_var = ctk.StringVar(value=self.settings.get("task", "detect"))
        self.mode_var = ctk.StringVar(value=self.settings.get("mode", "train"))
        self.epochs_var = ctk.StringVar(value=str(self.settings.get("epochs", 50)))
        self.imgsz_var = ctk.StringVar(value=str(self.settings.get("imgsz", 640)))
        self.batch_size_var = ctk.StringVar(value=str(self.settings.get("batch_size", 16)))
        self.verbose_var = ctk.BooleanVar(value=self.settings.get("verbose", True))
        self.device_var = ctk.StringVar(value=self.settings.get("device", "cpu"))
        self.lr0_var = ctk.StringVar(value=str(self.settings.get("lr0", 0.01)))
        self.lrf_var = ctk.StringVar(value=str(self.settings.get("lrf", 0.1)))
        self.lr_scheduler_var = ctk.StringVar(value=self.settings.get("lr_scheduler", "cosine"))

        self.create_widgets()

    def create_widgets(self):
        # Scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(self, width=480, height=500)
        scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)

        label_settings = {
            "text_color": "#FFFFFF",
            "font": ("Arial", 12),
        }
        entry_settings = {
            "fg_color": "#2B2B2B",
            "text_color": "#FFFFFF",
            "font": ("Arial", 12),
        }

        # Appearance mode dropdown
        ctk.CTkLabel(scrollable_frame, text="Appearance Mode:", **label_settings).pack(anchor="w", pady=5)
        ctk.CTkOptionMenu(scrollable_frame, values=["System", "Light", "Dark"], variable=self.appearance_mode_var).pack(anchor="w", pady=5)

        # Font and size input
        ctk.CTkLabel(scrollable_frame, text="Font:", **label_settings).pack(anchor="w", pady=5)
        font_frame = ctk.CTkFrame(scrollable_frame)
        font_frame.pack(anchor="w", pady=5, fill="x")
        ctk.CTkEntry(font_frame, textvariable=self.font_var, width=150, **entry_settings).pack(side="left", padx=5)
        ctk.CTkLabel(font_frame, text="Size:", **label_settings).pack(side="left", padx=5)
        ctk.CTkEntry(font_frame, textvariable=self.font_size_var, width=50, **entry_settings).pack(side="left", padx=5)

        # Add color pickers with preview
        self.add_color_picker(scrollable_frame, "Button Foreground Color:", self.button_fg_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Button Hover Color:", self.button_hover_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Button Border Color:", self.button_border_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Button Text Color:", self.button_text_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Label Text Color:", self.label_text_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Entry Foreground Color:", self.entry_fg_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Entry Text Color:", self.entry_text_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Progress Foreground Color:", self.progress_fg_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Progress Border Color:", self.progress_border_color_var, label_settings, entry_settings)
        self.add_color_picker(scrollable_frame, "Progress Progress Color:", self.progress_progress_color_var, label_settings, entry_settings)

        # Add Model Settings Tab
        self.add_model_settings(scrollable_frame)

        # Save button
        ctk.CTkLabel(self, text="อย่าลืมปิด - เปิดโปรแกรมใหม่หลักจาก Save Settings",font=("Tahoma", 18), text_color="#FF8A8A").pack(pady=10)
        ctk.CTkButton(self, text="Save Settings", command=self.save_settings).pack(pady=10)

    def add_color_picker(self, parent, label_text, variable, label_settings, entry_settings):
        """ Helper method to add a color picker row with a preview box """
        frame = ctk.CTkFrame(parent)
        frame.pack(anchor="w", pady=5, fill="x")

        ctk.CTkLabel(frame, text=label_text, **label_settings).pack(side="left", padx=5)
        color_entry = ctk.CTkEntry(frame, textvariable=variable, width=150, **entry_settings)
        color_entry.pack(side="left", padx=5)

        # Color preview box
        color_preview = ctk.CTkLabel(frame, width=30, height=20, text="", fg_color=variable.get())
        color_preview.pack(side="left", padx=5)

        # Update preview dynamically
        def update_preview(*args):
            color_value = variable.get()
            try:
                # Validate the color value
                self.winfo_toplevel().tk.call("winfo", "rgb", color_value)
                color_preview.configure(fg_color=color_value)
            except Exception:
                # If invalid, set to a default color (e.g., white)
                color_preview.configure(fg_color="#FFFFFF")

        variable.trace_add("write", update_preview)

    def add_model_settings(self, parent):
        """ Add model settings section """
        label_settings = {
            "text_color": "#FFFFFF",
            "font": ("Arial", 12),
        }
        entry_settings = {
            "fg_color": "#2B2B2B",
            "text_color": "#FFFFFF",
            "font": ("Arial", 12),
        }

        ctk.CTkLabel(parent, text="Model Settings", font=("Arial", 16), text_color="#FF8A8A").pack(anchor="w", pady=10)

        self.add_setting_row(parent, "Task:", self.task_var, ["detect", "classify", "segment"], label_settings)
        self.add_setting_row(parent, "Mode:", self.mode_var, ["train", "val"], label_settings)
        self.add_setting_row(parent, "Epochs:", self.epochs_var, None, label_settings, entry_settings)
        self.add_setting_row(parent, "Image Size:", self.imgsz_var, None, label_settings, entry_settings)
        self.add_setting_row(parent, "Batch Size:", self.batch_size_var, None, label_settings, entry_settings)
        self.add_setting_row(parent, "Verbose:", self.verbose_var, None, label_settings, entry_settings, is_checkbox=True)
        self.add_setting_row(parent, "Device:", self.device_var, ["cpu", "cuda"], label_settings)
        self.add_setting_row(parent, "Learning Rate (Start):", self.lr0_var, None, label_settings, entry_settings)
        self.add_setting_row(parent, "Learning Rate (End):", self.lrf_var, None, label_settings, entry_settings)
        self.add_setting_row(parent, "LR Scheduler:", self.lr_scheduler_var, ["cosine", "linear", "step"], label_settings)

    def add_setting_row(self, parent, label_text, variable, options, label_settings, entry_settings=None, is_checkbox=False):
        """ Helper method to add a setting row """
        frame = ctk.CTkFrame(parent)
        frame.pack(anchor="w", pady=5, fill="x")

        ctk.CTkLabel(frame, text=label_text, **label_settings).pack(side="left", padx=5)

        if is_checkbox:
            ctk.CTkCheckBox(frame, variable=variable).pack(side="left", padx=5)
        elif options:
            ctk.CTkOptionMenu(frame, values=options, variable=variable).pack(side="left", padx=5)
        else:
            ctk.CTkEntry(frame, textvariable=variable, width=150, **entry_settings).pack(side="left", padx=5)

    def save_settings(self):
        """ Save updated settings """
        self.settings["appearance_mode"] = self.appearance_mode_var.get()
        self.settings["font"] = [self.font_var.get(), int(self.font_size_var.get())]
        self.settings["button_fg_color"] = self.button_fg_color_var.get()
        self.settings["button_hover_color"] = self.button_hover_color_var.get()
        self.settings["button_border_color"] = self.button_border_color_var.get()
        self.settings["button_text_color"] = self.button_text_color_var.get()
        self.settings["label_text_color"] = self.label_text_color_var.get()
        self.settings["entry_fg_color"] = self.entry_fg_color_var.get()
        self.settings["entry_text_color"] = self.entry_text_color_var.get()
        self.settings["progress_fg_color"] = self.progress_fg_color_var.get()
        self.settings["progress_border_color"] = self.progress_border_color_var.get()
        self.settings["progress_progress_color"] = self.progress_progress_color_var.get()
        self.settings["task"] = self.task_var.get()
        self.settings["mode"] = self.mode_var.get()
        self.settings["epochs"] = int(self.epochs_var.get())
        self.settings["imgsz"] = int(self.imgsz_var.get())
        self.settings["batch_size"] = int(self.batch_size_var.get())
        self.settings["verbose"] = self.verbose_var.get()
        self.settings["device"] = self.device_var.get()
        self.settings["lr0"] = float(self.lr0_var.get())
        self.settings["lrf"] = float(self.lrf_var.get())
        self.settings["lr_scheduler"] = self.lr_scheduler_var.get()

        # Save settings to file
        self.master.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = YOLOTrainerGUI()
    app.mainloop()
