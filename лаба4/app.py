import tkinter as tk
import time
from tkinter import messagebox


class CoordinateGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Координатная сетка")

        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()

        self.execution_time_label = tk.Label(self.root, text="Время выполнения: -")
        self.execution_time_label.pack()

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.grid_step = 20
        self.width = 500
        self.height = 500
        self.points = []
        self.draw_grid()

        self.root.bind("<KeyPress-+>", self.zoom_in)
        self.root.bind("<KeyPress-minus>", self.zoom_out)

        self.create_input_fields()
        self.create_menu()

    def draw_grid(self):
        self.canvas.delete("all")
        scaled_step = int(self.grid_step * self.scale)

        # Рисование вертикальных линий сетки
        for i in range(0, self.width + scaled_step, scaled_step):
            self.canvas.create_line(i, 0, i, self.height, fill="lightgray")

        # Рисование горизонтальных линий сетки
        for i in range(0, self.height + scaled_step, scaled_step):
            self.canvas.create_line(0, i, self.width, i, fill="lightgray")

        # Координатные оси
        center_x = (self.width // 2) // scaled_step * scaled_step
        center_y = (self.height // 2) // scaled_step * scaled_step

        self.canvas.create_line(0, center_y, self.width, center_y, arrow=tk.LAST)
        self.canvas.create_line(center_x, 0, center_x, self.height, arrow=tk.LAST)
        self.canvas.create_text(self.width - 10, center_y - 10, text="x", fill="black", anchor=tk.SE)
        self.canvas.create_text(center_x + 10, 10, text="y", fill="black", anchor=tk.NW)
        # Метки на осях и обозначения начала координат
        for i in range(0, self.width + scaled_step, scaled_step):
            self.canvas.create_line(i, center_y - 5, i, center_y + 5, fill="black")
            self.canvas.create_line(center_x - 5, i, center_x + 5, i, fill="black")
        self.canvas.create_text(center_x + 10, center_y + 10, text="(0,0)", anchor=tk.NW)

    def create_input_fields(self):
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack()

        tk.Label(self.input_frame, text="x0:").grid(row=0, column=0)
        tk.Label(self.input_frame, text="y0:").grid(row=1, column=0)
        tk.Label(self.input_frame, text="x1:").grid(row=2, column=0)
        tk.Label(self.input_frame, text="y1:").grid(row=3, column=0)

        self.x0_entry = tk.Entry(self.input_frame)
        self.y0_entry = tk.Entry(self.input_frame)
        self.x1_entry = tk.Entry(self.input_frame)
        self.y1_entry = tk.Entry(self.input_frame)


        self.x0_entry.grid(row=0, column=1)
        self.y0_entry.grid(row=1, column=1)
        self.x1_entry.grid(row=2, column=1)
        self.y1_entry.grid(row=3, column=1)

        self.x0_entry.insert(0, "0")
        self.y0_entry.insert(0, "0")
        self.x1_entry.insert(0, "10")
        self.y1_entry.insert(0, "20")

        self.radius_label = tk.Label(self.input_frame, text="Радиус:")
        self.radius_entry = tk.Entry(self.input_frame)
        self.radius_entry.insert(0, "10")

        self.draw_button = tk.Button(self.input_frame, text="Нарисовать", command=self.draw_line)
        self.draw_button.grid(row=4, columnspan=2)
        self.clear_button = tk.Button(self.input_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=4, column=3)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.points.clear()

    def create_menu(self):
        self.algorithm_var = tk.StringVar(self.root)
        self.algorithm_var.set("Пошаговый алгоритм")  # Установить значение по умолчанию

        self.menu = tk.OptionMenu(self.root, self.algorithm_var, "Пошаговый алгоритм", "Брезенхема (линия)", "ЦДА",
                                  "Брезенхема (круг)", command=self.toggle_radius_entry)
        self.menu.pack()

    def toggle_radius_entry(self, algorithm):
        if algorithm == "Брезенхема (круг)":
            self.x1_entry.grid_remove()
            self.y1_entry.grid_remove()
            self.radius_label.grid(row=2, column=0)
            self.radius_entry.grid(row=2, column=1)
        else:
            self.radius_label.grid_remove()
            self.radius_entry.grid_remove()
            self.x1_entry.grid(row=2, column=1)
            self.y1_entry.grid(row=3, column=1)

    def draw_line(self):
        if not self.x0_entry.get() or not self.y0_entry.get() or not self.x1_entry.get() or not self.y1_entry.get():
            messagebox.showwarning("Input Error", "Please enter all coordinate values.")
            return
        try:
            x0 = int(self.x0_entry.get())
            y0 = int(self.y0_entry.get())
            x1 = int(self.x1_entry.get())
            y1 = int(self.y1_entry.get())
            print(f"Рисуем линию от ({x0}, {y0}) до ({x1}, {y1})")

            center_x = (self.width // 2) // int(self.grid_step * self.scale) * int(self.grid_step * self.scale)
            center_y = (self.height // 2) // int(self.grid_step * self.scale) * int(self.grid_step * self.scale)

            print(f"Центр координат: ({center_x}, {center_y})")

            algorithm = self.algorithm_var.get()
            start_time = time.time()
            if algorithm == "Пошаговый алгоритм":
                self.step_by_step_line(x0, y0, x1, y1)
            elif algorithm == "Брезенхема (линия)":
                self.bresenham_line(x0, y0, x1, y1)
            elif algorithm == "ЦДА":
                self.dda_line(x0, y0, x1, y1)
            elif algorithm == "Брезенхема (круг)":
                radius = int(self.radius_entry.get())
                self.bresenham_circle(x0, y0, radius)

            execution_time = time.time() - start_time
            self.execution_time_label.config(text=f"Время выполнения: {execution_time:.6f} секунд")
            self.canvas.focus_set()
        except ValueError as e:
            messagebox.showwarning("Input Error", "Please enter valid integer values for coordinates.")

    def plot_pixel(self, x, y):
        if (x, y) not in self.points:
            self.points.append((x, y))  # Сохраняем координаты точки
        self.redraw()

    def step_by_step_line(self, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        if dx > dy:  # Если линия более горизонтальная
            err = dx / 2.0
            while x1 != x2:
                self.plot_pixel(x1, y1)
                err -= dy
                if err < 0:
                    y1 += sy
                    err += dx
                x1 += sx
            self.plot_pixel(x1, y1)  # Закрасим последний пиксель
        else:  # Если линия более вертикальная
            err = dy / 2.0
            while y1 != y2:
                self.plot_pixel(x1, y1)
                err -= dx
                if err < 0:
                    x1 += sx
                    err += dy
                y1 += sy
            self.plot_pixel(x1, y1)  # Закрасим последний пиксель


    def dda_line(self, x0, y0, x1, y1):
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        x_inc = dx / steps
        y_inc = dy / steps
        x, y = x0, y0

        for _ in range(int(steps) + 1):
            self.plot_pixel(round(x), round(y))  # Ensure pixel coordinates are integers
            x += x_inc
            y += y_inc

    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.plot_pixel(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def bresenham_circle(self, xc, yc, r):
        x = 0
        y = r
        d = 3 - 2 * r
        self.plot_circle_points(xc, yc, x, y)

        while y >= x:
            x += 1
            if d > 0:
                y -= 1
                d += 4 * (x - y) + 10
            else:
                d += 4 * x + 6
            self.plot_circle_points(xc, yc, x, y)

    def plot_circle_points(self, xc, yc, x, y):
        self.plot_pixel(xc + x, yc + y)
        self.plot_pixel(xc - x, yc + y)
        self.plot_pixel(xc + x, yc - y)
        self.plot_pixel(xc - x, yc - y)
        self.plot_pixel(xc + y, yc + x)
        self.plot_pixel(xc - y, yc + x)
        self.plot_pixel(xc + y, yc - x)
        self.plot_pixel(xc - y, yc - x)

    def zoom_in(self, event):
        self.scale *= 1.1
        self.redraw()

    def zoom_out(self, event):
        self.scale /= 1.1
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.draw_grid()
        pixel_size = int(self.grid_step * self.scale)
        center_x = (self.width // 2) // pixel_size * pixel_size
        center_y = (self.height // 2) // pixel_size * pixel_size
        for x, y in self.points:
            self.canvas.create_rectangle(center_x + x * pixel_size, center_y - (y + 1) * pixel_size,
                                         center_x + (x + 1) * pixel_size, center_y - y * pixel_size,
                                         fill="green", outline="")


if __name__ == "__main__":
    root = tk.Tk()
    app = CoordinateGridApp(root)
    root.mainloop()