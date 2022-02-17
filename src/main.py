import numpy as np                     # pip3 install numpy
import PIL.Image, PIL.ImageTk          # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
from tkinter import *

if __name__ == "__main__":
    root = Tk()
    root.title("Tkinter")
    root.update()

    width = root.winfo_width()
    height = root.winfo_height()
    print(width, height)

    dim = max(width, height)

    # create a canvas and draw a numpy array on it
    canvas = Canvas(root, width=dim, height=dim, bg="red")

    # # create a numpy array
    arr = np.random.randint(0, 255, (dim, dim, 3), dtype=np.uint8)
    print(arr.shape)
    print(arr.dtype)

    # write it on the canvas
    img = PIL.Image.frombuffer('P', (dim,dim), arr, 'raw', 'P', 0, 1)
    img_tk = PIL.ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img_tk)
    canvas.pack(expand=False, fill=None)
    root.mainloop()

