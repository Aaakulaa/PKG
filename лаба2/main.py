import os
from tkinter import Tk, filedialog, Button, ttk
from PIL import Image

def analyze_images(folder_path):
    data = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                try:
                    file_path = os.path.join(root, file)
                    with Image.open(file_path) as img:
                        file_name = file
                        size = img.size
                        dpi = img.info.get('dpi', (72, 72))
                        color_mode = img.mode
                        if file.lower().endswith(('jpg', 'jpeg')):
                            compression = "Lossy"
                        elif file.lower().endswith('png'):
                            compression = "Lossless"
                        elif file.lower().endswith(('tiff', 'bmp')):
                            compression = "None"
                        else:
                            compression = "Unknown"
                        data.append((file_name,
                                     f"{size[0]} x {size[1]} px",
                                     f"{dpi[0]} x {dpi[1]} dpi",
                                     color_mode,
                                     compression))
                except Exception:
                    pass
    return data

def load_folder():
    folder_path = filedialog.askdirectory(title="Выберите папку с изображениями")
    if not folder_path:
        return
    data = analyze_images(folder_path)
    for row in tree.get_children():
        tree.delete(row)
    for row in data:
        tree.insert('', 'end', values=row)

root = Tk()
root.title("Информация об изображениях")
root.geometry("800x400")

btn_load = Button(root, text="Загрузить папку с изображениями", command=load_folder)
btn_load.pack(pady=10)

columns = ("Имя файла", "Размер (пиксели)", "Разрешение (dpi)", "Глубина цвета", "Сжатие")
tree = ttk.Treeview(root, columns=columns, show='headings')
tree.pack(fill="both", expand=True)

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=150)

root.mainloop()
