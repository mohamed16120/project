import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog

window = Tk()
window.title("Image Processing")
window.geometry("600x500")

global file_path 
file_path = None
def upload_image():
    global file_path 
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.resize(image, (300, 400))
        cv2.imshow('image', image)

def display_image(title,image):
        image = cv2.resize(image, (300, 400))
        cv2.imshow(title, image)

slider_var = IntVar()
def add_slider( label_text, min_val, max_val, initial_val, callback):
    slider_frame = Frame(window)
    slider_frame.place(x=0, y=180, width=400, height=50) 

    slider_label = Label(slider_frame, text=label_text)
    slider_label.grid(row=0, column=0, padx=5)
    
    slider = Scale(slider_frame, from_=min_val, to=max_val, orient=HORIZONTAL, variable=slider_var)
    slider.set(initial_val)
    slider.bind("<ButtonRelease-1>", callback)
    slider.grid(row=0, column=1, padx=5)
    
def update_lpf():
    if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5, apply_lpf)

def apply_lpf(event):
    kernel_size = int(event.widget.get())
    if kernel_size % 2 == 0:
        kernel_size += 1
    image = cv2.imread(file_path)
    lpf_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    display_image('LPF Result', lpf_image)

def update_hpf():
    if file_path is not None:
        add_slider("Kernel Size", 1, 20, 5, apply_hpf)
def apply_hpf(event):
    kernel_size = int(event.widget.get())
    if kernel_size % 2 == 0:
        kernel_size += 1
    image = cv2.imread(file_path)
    hpf_image = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    display_image('HPF Result', hpf_image)

def update_mean():
        if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5,apply_mean)

def apply_mean(event):
    kernel_size = int(event.widget.get())
    image = cv2.imread(file_path)
    mean_image = cv2.blur(image, (kernel_size, kernel_size))
    display_image('Mean Result', mean_image)

def update_median():
        if file_path is not None:
            add_slider("Kernel Size (Odd)", 3, 21, 3,apply_median)

def apply_median(event):
    kernel_size = int(event.widget.get())
    if kernel_size % 2 == 0:
        kernel_size += 1
    image = cv2.imread(file_path)
    median_image = cv2.medianBlur(image,  kernel_size)
    display_image('Median Result', median_image)


def apply_roberts():
    if file_path is not None:
        image = cv2.imread(file_path)
        roberts_image = cv2.filter2D(image, -1, np.array([[1, 0], [0, -1]]))
        display_image('Roberts Result', roberts_image)

def apply_prewitt():
    if file_path is not None:
        image = cv2.imread(file_path)
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_image_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)/255
        prewitt_image_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)/255
        prewitt_image = cv2.magnitude(prewitt_image_x, prewitt_image_y)
        display_image('Prewitt Result', prewitt_image)

def apply_sobel():
    if file_path is not None:
        image = cv2.imread(file_path)
        sobel_image = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
        display_image('Sobel Result', sobel_image)

def update_erosion():
        if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5, apply_erosion)

def apply_erosion(event):
    kernel_size = int(event.widget.get())
    image = cv2.imread(file_path)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    display_image('Erosion Result', eroded_image)

def update_dilation():
        if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5, apply_dilation)

def apply_dilation(event):
        kernel_size = int(event.widget.get())
        image = cv2.imread(file_path)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        display_image('Dilation Result', dilated_image)

def update_opening():
        if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5, apply_opening)

def apply_opening(event):
        kernel_size = int(event.widget.get())
        image = cv2.imread(file_path)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        display_image('Opening Result', opened_image)

def update_close():
        if file_path is not None:
            add_slider("Kernel Size", 1, 20, 5, apply_closing)

def apply_closing(event):
    if file_path is not None:
        kernel_size = int(event.widget.get())
        image = cv2.imread(file_path)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        display_image('Closing Result', closed_image)

def apply_hough_circle():
    if file_path is not None:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        display_image('Hough Circle Result', image)

def apply_segmentation_thresholding():
    if file_path is not None:
        image = cv2.imread(file_path)
        grayim= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, thresh_image = cv2.threshold(grayim, 127, 255, cv2.THRESH_BINARY)
        display_image('Segmented Image', thresh_image)

upload_button = Button(window, text="Upload Image", bg="green", fg='white', command= upload_image)
upload_button.place(x=0, y=140, width=100, height=20)

b1 = Button(window, text="LPF Result", bg="green", fg='white', command=update_lpf)
b1.place(x=0, y=0, width=100, height=20)

b2 = Button(window, text="HPF Result", bg="green", fg='white', command=update_hpf)
b2.place(x=120, y=0, width=100, height=20)

b3 = Button(window, text="Mean Result", bg="green", fg='white', command= update_mean)
b3.place(x=240, y=0, width=100, height=20)

b4 = Button(window, text="Median Result", bg="green", fg='white', command=update_median)
b4.place(x=0, y=30, width=100, height=20)

b5 = Button(window, text="Roberts Edge", bg="green", fg='white', command= apply_roberts)
b5.place(x=120, y=30, width=100, height=20)

b6 = Button(window, text="Prewitt Edge", bg="green", fg='white', command= apply_prewitt)
b6.place(x=240, y=30, width=100, height=20)

b7 = Button(window, text="Sobel Edge", bg="green", fg='white', command= apply_sobel)
b7.place(x=0, y=60, width=100, height=20)

b8 = Button(window, text="Erosion", bg="green", fg='white', command= update_erosion)
b8.place(x=120, y=60, width=100, height=20)

b9 = Button(window, text="Dilation", bg="green", fg='white', command= update_dilation)
b9.place(x=240, y=60, width=100, height=20)

b10 = Button(window, text="Opening", bg="green", fg='white', command= update_opening)
b10.place(x=0, y=90, width=100, height=20)

b11 = Button(window, text="Closing", bg="green", fg='white', command= update_close)
b11.place(x=120, y=90, width=100, height=20)

b12 = Button(window, text="Hough Circle", bg="green", fg='white', command= apply_hough_circle)
b12.place(x=240, y=90, width=100, height=20)

b13 = Button(window, text="Segmentation using thresholding", bg="green", fg='white', command= apply_segmentation_thresholding)
b13.place(x=0, y=120, width=220, height=20)

window.mainloop()
