import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class HistogramWidget:
    def __init__(self, figure):
        self.figure = figure
        self.ax_hist = self.figure.add_subplot(121)
        self.ax_image = self.figure.add_subplot(122)

        # Настройка фона и сетки
        self.ax_hist.set_facecolor('#2b2b2b')
        self.ax_image.set_facecolor('#2b2b2b')
        self.figure.patch.set_facecolor('#2b2b2b')

        self.ax_hist.grid(True, color='gray', alpha=0.3)
        self.ax_image.grid(False)  # Отключаем сетку для изображения
        self.ax_hist.tick_params(colors='white')
        self.ax_image.tick_params(colors='white')

        # Изначально скрыть изображение (пустой график)
        self.ax_image.axis('off')

    def update_histogram(self, image):
        self.ax_hist.clear()
        if len(image.shape) == 3:  # Цветное изображение
            colors = ('b', 'g', 'r')
            labels = ('Blue', 'Green', 'Red')
            for i, (color, label) in enumerate(zip(colors, labels)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.ax_hist.plot(hist, color=color, label=label, linewidth=2)
            self.ax_hist.legend()
        else:  # Черно-белое изображение
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.ax_hist.plot(hist, color='white', linewidth=2)

        self.ax_hist.set_xlim([0, 256])
        self.ax_hist.set_ylim(bottom=0)  # Начинаем с нуля
        self.ax_hist.grid(True, color='gray', alpha=0.3)
        self.ax_hist.tick_params(colors='white')

        self.figure.canvas.draw()  # Обновляем отображение

    def update_image(self, image):
        self.ax_image.clear()  # Очищаем ось
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.ax_image.imshow(image_rgb)
        self.ax_image.axis('off')  # Убираем оси для чистого изображения

        self.figure.canvas.draw()  # Обновляем отображение


class ImageProcessor:
    @staticmethod
    def linear_contrast(image, alpha, beta):
        """Применение линейного контрастирования."""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def histogram_equalization_rgb(image):
        """Эквализация гистограммы для цветного изображения."""
        b, g, r = cv2.split(image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge((b_eq, g_eq, r_eq))

    @staticmethod
    def histogram_equalization_grayscale(image):
        """Эквализация гистограммы для изображения в градациях серого."""
        return cv2.equalizeHist(image)

    @staticmethod
    def histogram_equalization_hsv(image):
        """Эквализация гистограммы для компоненты яркости в пространстве HSV."""
        # Преобразуем изображение в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Экстрагируем компонент яркости (V)
        h, s, v = cv2.split(hsv)

        # Выполним эквализацию только для компоненты яркости (V)
        v_eq = cv2.equalizeHist(v)

        # Собираем изображение обратно
        hsv_eq = cv2.merge([h, s, v_eq])

        # Преобразуем изображение обратно в BGR
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


class ImageProcessingApp:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.root = Tk()
        self.root.title("Image Processing Application")

        self.histogram_widget = None
        self.status_label = Label(self.root, text="No image loaded", fg="white", bg="#2b2b2b")
        self.status_label.pack()

        self.create_widgets()

    def create_widgets(self):
        self.load_button = Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.process_button = Button(self.root, text="Process Image", command=self.process_image)
        self.process_button.pack()

        self.equalize_button_rgb = Button(self.root, text="Equalize Histogram (RGB)", command=self.equalize_histogram_rgb)
        self.equalize_button_rgb.pack()

        self.equalize_button_hsv = Button(self.root, text="Equalize Histogram (HSV)", command=self.equalize_histogram_hsv)
        self.equalize_button_hsv.pack()

        self.figure = Figure(figsize=(10, 4))  # Увеличиваем ширину фигуры
        self.histogram_widget = HistogramWidget(self.figure)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image)
            self.histogram_widget.update_histogram(self.original_image)
            self.status_label.config(text="Image loaded")

    def process_image(self):
        if self.original_image is None:
            return

        # Применение линейного контрастирования
        self.processed_image = ImageProcessor.linear_contrast(self.original_image, 1.5, 0)

        # Добавление текста на изображение
        cv2.putText(self.processed_image, "Linear Contrast Applied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.display_image(self.processed_image)
        self.histogram_widget.update_histogram(self.processed_image)

        # Обновление статуса
        self.status_label.config(text="Linear contrast applied")

    def equalize_histogram_rgb(self):
        if self.original_image is None:
            return

        # Эквализация гистограммы для цветного изображения
        self.processed_image = ImageProcessor.histogram_equalization_rgb(self.original_image)

        # Добавление текста на изображение
        cv2.putText(self.processed_image, "Histogram Equalized (RGB)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.display_image(self.processed_image)
        self.histogram_widget.update_histogram(self.processed_image)

        # Обновление статуса
        self.status_label.config(text="Histogram equalized (RGB)")

    def equalize_histogram_hsv(self):
        if self.original_image is None:
            return

        # Эквализация гистограммы для компоненты яркости в пространстве HSV
        self.processed_image = ImageProcessor.histogram_equalization_hsv(self.original_image)

        # Добавление текста на изображение
        cv2.putText(self.processed_image, "Histogram Equalized (HSV)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.display_image(self.processed_image)
        self.histogram_widget.update_histogram(self.processed_image)

        # Обновление статуса
        self.status_label.config(text="Histogram equalized (HSV)")

    def display_image(self, image):
        # Уменьшаем изображение
        image = imutils.resize(image, width=500)
        self.histogram_widget.update_image(image)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ImageProcessingApp()
    app.run()
