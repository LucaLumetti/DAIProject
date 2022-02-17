import numpy as np
import cv2
import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading
import pickle

from video import Video
from asynclenia import Lenia

class App:
    def __init__(self, root, lenia):
        self.lenia = lenia
        self.running = False
        #setting title
        root.title("undefined")
        #setting window size
        width=1200
        height=1000
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        # root.resizable(width=False, height=False)

        # Split the main windows in to two frames full width
        left_frame = tk.Frame(root, bg="purple")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)


        kernel_frame = tk.Frame(root, bg="orange")
        kernel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        mid_frame_a = tk.Frame(root, bg="green")
        mid_frame_a.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        mid_frame_b = tk.Frame(root, bg="orange")
        mid_frame_b.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.kernel_canvas = tk.Canvas(kernel_frame)
        self.kernel_canvas.pack(fill=tk.X, expand=False)
        self.kernel_canvas.update()

        # Add canvas to the left frame
        self.canvas = tk.Canvas(left_frame, bg="black", width=screenheight, height=screenheight)
        self.canvas.bind('<Button-1>', self.paint)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.update()

        # Add button and slider to the right frame
        self.run_button = tk.Button(right_frame, text="Run", command=self.run)
        self.run_button.pack()

        self.stop_button = tk.Button(right_frame, text="Stop", command=self.stop)
        self.stop_button.pack()

        self.noise_button = tk.Button(right_frame, text="Add noise", command=self.add_noise)
        self.noise_button.pack()

        self.red_noise_button = tk.Button(right_frame, text="Add red noise", command=self.add_red_noise)
        self.red_noise_button.pack()

        self.green_noise_button = tk.Button(right_frame, text="Add green noise", command=self.add_green_noise)
        self.green_noise_button.pack()

        self.blue_noise_button = tk.Button(right_frame, text="Add blue noise", command=self.add_blue_noise)
        self.blue_noise_button.pack()

        self.clear_button = tk.Button(right_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.random_param_1D_button = tk.Button(right_frame, text="Randomize 1D", command=self.randomize_and_update)
        self.random_param_1D_button.pack()

        self.fps_slider = tk.Scale(right_frame, from_=1, to=100, orient=tk.HORIZONTAL)
        self.fps_slider.pack()

        self.noise_strong = tk.Scale(right_frame, from_=1, to=100, orient=tk.HORIZONTAL)
        self.noise_strong.pack()

        self.save_map = tk.Button(right_frame, text="Save map", command=self.save_map)
        self.save_map.pack()

        self.load_map = tk.Button(right_frame, text="Load map", command=self.load_map)
        self.load_map.pack()

        self.save_img_button = tk.Button(right_frame, text="Save image", command=self.save_img)
        self.save_img_button.pack()

        # params controls
        self.R_slider = tk.Scale(mid_frame_a, label="R", from_=5, to=20, orient=tk.HORIZONTAL)
        self.R_slider.pack()

        self.k_mean_slider = tk.Scale(mid_frame_a, label="k_mean", from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.k_mean_slider.pack()

        self.k_std_slider = tk.Scale(mid_frame_a, label="k_std", from_=0.01, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.k_std_slider.pack()

        self.g_mean_slider = tk.Scale(mid_frame_a, label="g_mean", from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.g_mean_slider.pack()

        self.g_std_slider = tk.Scale(mid_frame_a, label="g_std", from_=0.001, to=0.1, resolution=0.001, orient=tk.HORIZONTAL)
        self.g_std_slider.pack()

        self.T_slider = tk.Scale(mid_frame_a, label="T", from_=1, to=25, resolution=1, orient=tk.HORIZONTAL)
        self.T_slider.pack()

        self.b_text = tk.Label(mid_frame_a, text="b")
        self.b_text.pack()
        self.b_input = tk.Text(mid_frame_a, height=1, width=10)
        self.b_input.pack()

        self.save_creature_button = tk.Button(mid_frame_a, text="Save creature", command=self.save_creature)
        self.save_creature_button.pack()

        self.load_creature_button = tk.Button(mid_frame_a, text="Load creature", command=self.load_creature)
        self.load_creature_button.pack()

        self.creature_name_input = tk.Text(mid_frame_a, height=1, width=10)
        self.creature_name_input.pack()

        self.update_param_list()

    def save_creature(self):
        creature_name = self.creature_name_input.get("1.0", "end-1c")
        print(f"saving creature {creature_name}")

        with open(f'creatures/{creature_name}', 'wb') as lenia_dump:
            pickle.dump(self.lenia, lenia_dump)

    def load_creature(self):
        creature_name = self.creature_name_input.get("1.0", "end-1c")
        with open(f'creatures/{creature_name}', 'rb') as lenia_dump:
            self.lenia = pickle.load(lenia_dump)
        self.update_param_list()

    def update_params(self):
        R = self.R_slider.get()
        k_mean = self.k_mean_slider.get()
        k_std = self.k_std_slider.get()
        g_mean = self.g_mean_slider.get()
        g_std = self.g_std_slider.get()
        T = self.T_slider.get()

        self.lenia.set_R(R)
        self.lenia.set_k_mean(k_mean)
        self.lenia.set_k_std(k_std)
        self.lenia.set_g_mean(g_mean)
        self.lenia.set_g_std(g_std)
        self.lenia.set_T(T)
        self.lenia.set_b(eval(self.b_input.get("1.0", "end-1c")))

        self.kernel = self.lenia.get_kernel_img().copy()
        kernel_canvas_width = self.kernel_canvas.winfo_width()
        kernel_canvas_height = self.kernel_canvas.winfo_height()
        self.kernel = self.numpy_to_tk(self.kernel, kernel_canvas_width, kernel_canvas_height)
        self.kernel_canvas.create_image(0, 0, image=self.kernel, anchor=tk.NW)
        self.kernel_canvas.update()

    def save_img(self):
        img = self.lenia.world.copy()
        img = img.swapaxes(0,2).swapaxes(0,1)
        img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("saved.jpg", img*255)

    def randomize_and_update(self):
        self.lenia.randomize_1D_param()
        self.update_param_list()

    def update_param_list(self):
        for param_i,param in enumerate(self.lenia.params):
            plist = param.get_params()
            self.R_slider.set(plist['R'])
            self.k_mean_slider.set(plist['k_mean'])
            self.k_std_slider.set(plist['k_std'])
            self.g_mean_slider.set(plist['g_mean'])
            self.g_std_slider.set(plist['g_std'])
            self.T_slider.set(plist['T'])
            self.b_input.delete("1.0", "end")
            self.b_input.insert("1.0", str(list(plist['b'])))

    def paint(self, event):
        print(event)
        x1, y1 = (event.x), (event.y)
        x1, y1 = x1 / self.scale - 5, y1 / self.scale - 5
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = (x1 + 10), (y1 + 10)
        x2, y2 = np.min([x2, self.lenia.world.shape[2]]), np.min([y2, self.lenia.world.shape[1] ])

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)

        self.lenia.world[:, y1:y2, x1:x2] = 0

    def save_map(self):
        self.saved = self.lenia.world.copy()

    def load_map(self):
        self.lenia.world = self.saved


    def add_red_noise(self):
        self.lenia.add_gaussian_noise(channels=[0], strong=self.noise_strong.get())

    def add_green_noise(self):
        self.lenia.add_gaussian_noise(channels=[1], strong=self.noise_strong.get())

    def add_blue_noise(self):
        self.lenia.add_gaussian_noise(channels=[2], strong=self.noise_strong.get())

    def add_noise(self):
        self.lenia.add_gaussian_noise(strong=self.noise_strong.get())

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.update()
        self.lenia.reset()

    def numpy_to_tk(self, img, canvas_width, canvas_height):
        scale_x = canvas_width / img.shape[1]
        scale_y = canvas_height / img.shape[0]
        self.scale = min(scale_x, scale_y)
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        return ImageTk.PhotoImage(image=Image.fromarray(img))

    # change the image showed in the canvas
    def step(self):
        if self.running == False:
            return


        self.lenia.step()
        self.img = self.lenia.get_world_img()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.tkimg = self.numpy_to_tk(self.img, canvas_width, canvas_height)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=tk.NW)
        self.canvas.update()
        self.update_params()
        self.canvas.after(1000//self.fps_slider.get(), self.step)

    def run(self):
        self.running = True
        self.canvas.after(0, self.step)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    lenia = Lenia(64, 1)
    root = tk.Tk()
    app = App(root, lenia)
    root.mainloop()

