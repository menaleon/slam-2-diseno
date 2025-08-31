import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from calc_metrics import calc_metrics

SLAM_FOLDER = "SLAM"

def get_SLAM_types():
    return [
        name for name in os.listdir(SLAM_FOLDER)
        if os.path.isdir(os.path.join(SLAM_FOLDER, name)) and name.startswith("SLAM_")
    ] if os.path.exists(SLAM_FOLDER) else []

def select_video():
    return filedialog.askopenfilename(
        title="Select .mp4 video",
        filetypes=[("Video files", "*.mp4")]
    )

def run_SLAM(SLAM_name):
    video_path = select_video()
    if not video_path:
        messagebox.showwarning("Selection canceled", "No video was selected.")
        return

    messagebox.showinfo("SLAM selected", f"Running: {SLAM_name}\nWith video:\n{video_path}")
    script_path = os.path.join(SLAM_FOLDER, SLAM_name, "main.py")
    command = ["python3", script_path, video_path]

    results = calc_metrics(command, SLAM_name)

    filtered_output = {}
    for key, value in results.items():
        if isinstance(value, str):
            if not value.strip():
                continue
            if len(value) > 500:
                filtered_output[key] = value[:500] + "... (truncated)"
                continue
        filtered_output[key] = value

    message = json.dumps(filtered_output, indent=4, ensure_ascii=False)
    messagebox.showinfo("Execution results", message)

def create_interface():
    window = tk.Tk()
    window.title("Multi-SLAM Platform")
    window.geometry("800x500")
    window.resizable(False, False)

    # === Gradient canvas ===
    canvas = tk.Canvas(window, width=800, height=500)
    canvas.pack(fill="both", expand=True)

    # Gradient from #99c2ff to #f0f4f8
    for i in range(500):
        r = int(153 + (240 - 153) * (i / 500))  # Red: 153 → 240
        g = int(194 + (244 - 194) * (i / 500))  # Green: 194 → 244
        b = int(255 + (248 - 255) * (i / 500))  # Blue: 255 → 248
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, 800, i, fill=color)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton",
                     font=("Segoe UI", 11, "bold"),
                     padding=10,
                     background="#388e3c",
                     foreground="white",
                     relief="flat")
    style.map("TButton", background=[("active", "#2e7d32")])
    style.configure("TLabel", background="#ffffff", foreground="#333")

    frame = tk.Frame(canvas, bg="white", bd=2, relief="ridge")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=350)

    tk.Label(frame, text="Multi-SLAM Platform", font=("Helvetica Neue", 20, "bold"), bg="white", fg="#0d47a1").pack(pady=(20, 10))
    tk.Label(frame, text="Select an SLAM and an .mp4 video to generate a 2D trajectory", font=("Segoe UI", 11), bg="white").pack()

    # === Buttons ===
    button_frame = tk.Frame(frame, bg="white")
    button_frame.pack(pady=20)

    SLAM_types = get_SLAM_types()
    for i, SLAM in enumerate(SLAM_types):
        display_name = SLAM.replace("SLAM_", "SLAM type ")
        button = ttk.Button(button_frame, text=display_name, command=lambda SLAM=SLAM: run_SLAM(SLAM))
        button.grid(row=i // 2, column=i % 2, padx=15, pady=10)

    tk.Label(frame, text="© TEC 2025 | SLAM Project", font=("Segoe UI", 11), bg="white", fg="black").pack(side="bottom", pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_interface()
