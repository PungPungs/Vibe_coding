
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from PIL import Image

class FileRenamerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("파일 이름 병합 프로그램")

        self.properties = ["영수증", "거래명세서", "전자세금계산서", "전표", "캡처", "사업자등록증_통장사본", "사진"]
        self.file_groups = {}

        # Base Name
        tk.Label(root, text="파일 이름:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.base_name_entry = tk.Entry(root, width=50)
        self.base_name_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=5)

        # File List Header
        tk.Label(root, text="파일 추가").grid(row=1, column=0, padx=10, pady=5)
        tk.Label(root, text="속성").grid(row=1, column=1, padx=10, pady=5)

        # File List Frame
        self.file_frame = tk.Frame(root)
        self.file_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        # Add File Button
        self.add_file_button = tk.Button(root, text="파일 추가", command=self.add_files_for_property)
        self.add_file_button.grid(row=3, column=0, padx=10, pady=10)

        # Rename Button
        self.rename_button = tk.Button(root, text="이름 및 포멧 변경", command=self.rename_and_merge)
        self.rename_button.grid(row=3, column=1, padx=10, pady=10)

    def add_files_for_property(self):
        prop_var = tk.StringVar()
        prop_var.set(self.properties[0])
        
        # Simple dialog to select property first
        top = tk.Toplevel(self.root)
        top.title("파일 종류")
        tk.Label(top, text="파일의 종류를 선택해주세요:").pack(padx=10, pady=10)
        prop_menu = ttk.Combobox(top, textvariable=prop_var, values=self.properties, width=20)
        prop_menu.pack(padx=10, pady=5)
        
        def on_select():
            prop = prop_var.get()
            top.destroy()
            files = filedialog.askopenfilenames()
            if files:
                self.add_file_entry(prop, list(files))

        select_button = tk.Button(top, text="파일 선택", command=on_select)
        select_button.pack(padx=10, pady=10)

    def add_file_entry(self, prop, file_paths):
        if prop not in self.file_groups:
            self.file_groups[prop] = []
        self.file_groups[prop].extend(file_paths)

        frame = tk.Frame(self.file_frame)
        frame.pack(fill="x", padx=5, pady=2)

        file_names = ", ".join([os.path.basename(p) for p in file_paths])
        if len(file_names) > 50:
            file_names = file_names[:47] + "..."
            
        name_label = tk.Label(frame, text=file_names, width=40, anchor="w")
        name_label.pack(side="left")

        prop_label = tk.Label(frame, text=prop, width=15)
        prop_label.pack(side="left", padx=5)

    def rename_and_merge(self):
        base_name = self.base_name_entry.get()
        if not base_name:
            messagebox.showerror("Error", "Please enter a standard file name.")
            return

        if not self.file_groups:
            messagebox.showerror("Error", "Please add files for at least one property.")
            return

        for prop, file_paths in self.file_groups.items():
            if not file_paths:
                continue

            image_files = [f for f in file_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            if image_files:
                # If there are any image files, merge them all into a PDF
                if len(image_files) != len(file_paths):
                    messagebox.showwarning("Warning", f"For property '{prop}', non-image files were ignored during PDF conversion.")

                try:
                    images = [Image.open(f).convert("RGB") for f in image_files]
                    if images:
                        first_image = images[0]
                        dir_name = os.path.dirname(file_paths[0])
                        new_name = f"{base_name}_{prop}.pdf"
                        new_path = os.path.join(dir_name, new_name)
                        
                        first_image.save(new_path, save_all=True, append_images=images[1:])
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred during PDF conversion for property '{prop}': {e}")
                    continue
            elif len(file_paths) == 1:
                # If no images and only one file, just rename it
                file_path = file_paths[0]
                dir_name = os.path.dirname(file_path)
                ext = os.path.splitext(file_path)[1]
                new_name = f"{base_name}_{prop}{ext}"
                new_path = os.path.join(dir_name, new_name)
                try:
                    os.rename(file_path, new_path)
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred while renaming {os.path.basename(file_path)}: {e}")
                    continue
            else:
                # If multiple non-image files are selected for a property
                messagebox.showerror("Error", f"Cannot merge multiple non-image files for property '{prop}'.")
                continue

        messagebox.showinfo("Success", "파일이 성공적으로 변경되었습니다.")
        # Clear the list and frames
        self.file_groups = {}
        for widget in self.file_frame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        messagebox.showerror("Error", "Pillow library not found. Please install it using: pip install Pillow")
        exit()
        
    root = tk.Tk()
    app = FileRenamerApp(root)
    root.mainloop()
