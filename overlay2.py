import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.MtrLoad = False
        self.rsLoad = False
        self.imgLoad = False

        # window
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.title('Overlay')
        self.master.geometry(f"{screen_width}x{screen_height}+0+0")
        self.mainframe = tk.Frame(self.master)
        self.mainframe.pack(fill=tk.BOTH, expand=True)

        # widget setting
        pad_x = 10
        pad_y = 10
        self.crrX = tk.DoubleVar()
        self.crrY = tk.DoubleVar()

        self.waveBtn = tk.Button(self.mainframe, text="Load RawData", command=self.loadMtr)
        self.waveBtn.grid(column=0, row=0, columnspan=2, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.valX = tk.IntVar(value=20)
        self.valY = tk.IntVar(value=20)

        self.Xlabel = tk.Label(self.mainframe, text="num X step")
        self.Xlabel.grid(column=0, row=2, sticky=tk.EW, padx=pad_x, pady=pad_y)
        self.Ylabel = tk.Label(self.mainframe, text="num Y step")
        self.Ylabel.grid(column=0, row=3, sticky=tk.EW, padx=pad_x, pady=pad_y)
        self.Xspnbox = tk.Spinbox(self.mainframe, textvariable=self.valX, from_=0, to=100, increment=1)
        self.Xspnbox.grid(column=1, row=2, sticky=tk.EW, padx=pad_x, pady=pad_y)
        self.Yspnbox = tk.Spinbox(self.mainframe, textvariable=self.valY, from_=0, to=100, increment=1)
        self.Yspnbox.grid(column=1, row=3, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.imgBtn = tk.Button(self.mainframe, text="Load img", command=self.loadImg)
        self.imgBtn.grid(column=0, row=4, columnspan=2, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.setBtn = tk.Button(self.mainframe, text="Initialize", command=self.init)
        self.setBtn.grid(column=0, row=5, columnspan=2, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.rotBtn = tk.Button(self.mainframe, text="Rotate clockwise", command=self.rotate_intensity_data)
        self.rotBtn.grid(column=0, row=6, columnspan=1, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.rotBtn = tk.Button(self.mainframe, text="Rotate anticlockwise", command=self.antrotate_intensity_data)
        self.rotBtn.grid(column=1, row=6, columnspan=1, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.mirror_button = tk.Button(self.mainframe, text="Mirror Horizontally", command=self.mirror_intensity_data)
        self.mirror_button.grid(column=0, row=7, columnspan=2, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.transparency_label = tk.Label(self.mainframe, text="Adjust Transparency")
        self.transparency_label.grid(column=0, row=8, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.transparency_scale = tk.Scale(self.mainframe, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_transparency)
        self.transparency_scale.set(40)  # Default transparency
        self.transparency_scale.grid(column=1, row=8, sticky=tk.EW, padx=pad_x, pady=pad_y)

        self.buffCanvas = tk.Canvas(self.mainframe, width=100, height=100)
        self.buffCanvas.grid(column=0, row=9, columnspan=2, sticky=tk.EW, padx=pad_x, pady=pad_y, ipady=60)

        # Image related
        self.canvasSize = 500
        self.imgCanvas = tk.Canvas(self.mainframe, width=self.canvasSize, height=self.canvasSize, highlightthickness=0)
        self.imgCanvas.create_rectangle(0, 0, self.canvasSize, self.canvasSize, fill='#DDDDDD')
        self.imgCanvas.grid(column=2, row=1, rowspan=10, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S, padx=pad_x * 2, pady=pad_y * 2, ipady=30)
        self.imgCanvas.bind('<B1-Motion>', self.pickPos)

        self.valCrrX = tk.IntVar()
        self.valCrrY = tk.IntVar()
        self.crrXValue = tk.Label(self.mainframe, textvariable=self.valCrrX)
        self.crrXValue.grid(column=3, row=7, sticky=tk.SW, padx=1, pady=pad_y / 4)
        self.crrYValue = tk.Label(self.mainframe, textvariable=self.valCrrY)
        self.crrYValue.grid(column=3, row=8, sticky=tk.NW, padx=1, pady=pad_y / 4)
        
        # Transparency variable
        self.transparency = self.transparency_scale.get() / 100.0

    def update_transparency(self, value):
        self.transparency = int(value) / 100.0
        self.overlay_intensity_data()

    # Add the overlay_intensity_data method
    def overlay_intensity_data(self):

        # Convert image to array
        image_array = np.array(self.img_)

        # Assuming the matrix data should be resized to match the image dimensions
        rows, cols = self.Mtr.shape
        resized_matrix = cv2.resize(self.Mtr, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize the resized matrix to the range [0, 255] for overlay
        normalized_matrix = cv2.normalize(resized_matrix, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized_matrix.astype(np.uint8), cv2.COLORMAP_JET)

        # Blend the heatmap with the original image
        overlay_image = cv2.addWeighted(image_array, 1-self.transparency, heatmap, self.transparency, 0)

        # Convert the overlay image back to PIL format for display in Tkinter
        overlay_image_pil = Image.fromarray(overlay_image)
        self.imgTk = ImageTk.PhotoImage(overlay_image_pil)
        self.imgCanvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        self.imgCanvas.config(scrollregion=self.imgCanvas.bbox(tk.ALL))

    # Load the image
    def loadImg(self):
        self.imgPath = filedialog.askopenfilename(filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.emf;*.tiff;*.tif')], title="Choose Image file")
        if self.imgPath == '':
            return 0
        self.img_array = np.fromfile(self.imgPath, dtype=np.uint8)
        self.img_ = cv2.imdecode(self.img_array, cv2.IMREAD_COLOR)
        self.imgLoad = True
        self.display_image()

    # Load the heatmap data
    def loadMtr(self):
        self.mtrPath = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')], title="Choose heatmap")
        if self.mtrPath == '':
            return 0

        # Read the file content
        with open(self.mtrPath, 'r') as file:
            lines = file.readlines()


        # Get the desired number of rows and columns from the spinboxes
        num_rows = self.valY.get()
        num_cols = self.valX.get()

        # Initialize an empty array with the specified shape
        self.Mtr = np.zeros((num_rows, num_cols))

        # Fill the array with values from the file
        current_row = 0
        current_col = 0
        for line in lines:
            numbers = line.split()
            for number in numbers:
                if current_col < num_cols:
                    self.Mtr[current_row, current_col] = float(number)
                    current_col += 1
                if current_col >= num_cols:
                    current_col = 0
                    current_row += 1
                if current_row >= num_rows:
                    break
            if current_row >= num_rows:
                break
        self.MtrLoad = True

    # Initialize
    def init(self):
        self.getXYGrid()
        if self.checkValid_onInit() == -1:
            return -1

        self.imgSet()
        self.ax.plot(self.rs, self.Mtr[:, 0], linewidth=0.5)
        self.figCanvas.draw()
        self.overlay_intensity_data()  # Call the overlay method
        
    def getXYGrid(self):
        self.xGrid = int(self.Xspnbox.get())
        self.yGrid = int(self.Yspnbox.get())

    def checkValid_onInit(self):  # confirm data validity
        if self.MtrLoad is False:
            print("Spectral data is insufficient ...")
            return -1

        if self.imgLoad is False:
            print("Image data is not loaded ...")
            return -1
        # size consistency

    def imgResize(self):  # img :image attribute
        self.imgResize = self.img_.copy()
        self.crrImgX = self.img_.shape[0]
        self.crrImgY = self.img_.shape[1]

        if self.crrImgX <= self.crrImgY:
            self.imgResize = cv2.resize(self.imgResize, dsize=None, fx=self.canvasSize / self.crrImgY, fy=self.canvasSize / self.crrImgY)
        else:
            self.imgResize = cv2.resize(self.imgResize, dsize=None, fx=self.canvasSize / self.crrImgX, fy=self.canvasSize / self.crrImgX)

        self.ResImgSizeY, self.ResImgSizeX = self.imgResize.shape[:2]

        self.img = cv2.cvtColor(self.imgResize, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(self.img)
        self.img_tk = ImageTk.PhotoImage(self.img_pil)

        return 0


    def imgSet(self):
        self.imgResize()
        self.imgCanvas.create_image(0, 0, image=self.img_tk, anchor='nw')
        
    def pickPos(self, event):
        self.prevX = self.crrX
        self.prevY = self.crrY

        self.crrX = event.x
        self.crrY = event.y

        if self.crrX >= self.ResImgSizeX:
            self.crrX = self.ResImgSizeX - 0.01
        if self.crrY >= self.ResImgSizeY:
            self.crrY = self.ResImgSizeY - 0.01
        if self.crrX < 0:
            self.crrX = 0
        if self.crrY < 0:
            self.crrY = 0

        self.calcGridPos()
        self.updateGraph()
        self.moveCursor()

    def moveCursor(self):
        if not hasattr(self, "lineX") or not hasattr(self, "lineY"):
            self.lineX = self.imgCanvas.create_line(0, self.crrY, self.ResImgSizeX, self.crrY, tag="lineX", fill="#3BAF75")
            self.lineY = self.imgCanvas.create_line(self.crrX, 0, self.crrX, self.ResImgSizeX, tag="lineY", fill="#3BAF75")
        else:
            self.imgCanvas.move(self.lineX, 0, self.crrY - self.prevY)
            self.imgCanvas.move(self.lineY, self.crrX - self.prevX, 0)

    def rotate_intensity_data(self):
        if hasattr(self, 'Mtr') and self.Mtr is not None:
            self.Mtr = np.rot90(self.Mtr)
            self.overlay_intensity_data()

    def antrotate_intensity_data(self):
        if hasattr(self, 'Mtr') and self.Mtr is not None:
            self.Mtr = np.rot90(self.Mtr,3)
            self.overlay_intensity_data()

    def mirror_intensity_data(self):
        if hasattr(self, 'Mtr') and self.Mtr is not None:
            self.Mtr = np.fliplr(self.Mtr)
            self.overlay_intensity_data()

if __name__ == "__main__":
    overlay_img = tk.Tk()
    Application = App(master=overlay_img)
    Application.mainloop()