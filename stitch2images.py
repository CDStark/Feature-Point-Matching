"""
@author: Kartik Saini

"""

#%%

# importing tkinter gui 
import tkinter as tk 
import cv2
import numpy as np
from PIL import ImageTk, Image
from functools import partial
from pathlib import Path

#%%
canvasG = None
window_width = None
window_height = None
drawLine = False
img_w = None
img_h = None

x1 = 0
y1 = 0
x2 = 0
y2 = 0

ix1 = 0
iy1 = 0
ix2 = 0
iy2 = 0

inA = False
inB = False

imageB = None
imageA = None

matchesA = []
matchesB = []

imageBLoaded = False
imageALoaded = False

def selectGlobalCanvas(event):
    
    if imageBLoaded == False or imageALoaded == False:
        return
    
    global x1,y1,x2,y2, ix1, iy1, ix2, iy2, matchesA, matchesB
    global canvasG, drawLine, img_w, img_h, inA, inB
    x = event.x
    y = event.y
    
    if x < img_w:
        x2 = x
        y2 = y
        
        ix2 = x
        iy2 = y
        inB = True
        
        canvasG.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, outline="blue",
                         width=2)
        
    elif x > img_w+10:
        x1 = x
        y1 = y
        
        ix1 = x-img_w-10
        iy1 = y
        inA = True
        canvasG.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, outline="yellow",
                         width=2)
    
    
    
    if inA == True and inB == True:
        canvasG.create_line(x1, y1, x2, y2, fill="red", width=3)
        matchesA.append([ix1,iy1])
        matchesB.append([ix2,iy2])
        inA = False
        inB = False
        

        
def loadImages(image_root_path, image_id):
    global canvasG, img_w, img_h, imageB, imageA, imageBLoaded, imageALoaded

    root = Path(image_root_path)
    path1 = root / (image_id.get() + '_001.tiff')
    path2 = root / (image_id.get() + '_070.tiff')

    # Load an image using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path1)), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    imageA = cv_img

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    # Add a PhotoImage to the Canvas
    canvasG.create_image(0, 0, image=photo, anchor=tk.NW, tags="imageA")

    if imageALoaded == True:
        clearSelection()
    imageALoaded = True

    # Load an image using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path2)), cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    imageB = cv_img

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    # Add a PhotoImage to the Canvas
    canvasG.create_image(img_w+10, 0, image=photo, anchor=tk.NW, tags="imageB")

    if imageBLoaded == True:
        clearSelection()
    imageBLoaded = True

    tk.mainloop()

# def loadImage2(image_path):
#     global canvasG, img_w, img_h, imageA, imageB, imageALoaded, imageBLoaded
#
#     #print(image_path.get())
#     # Load an image using OpenCV
#     cv_img = cv2.cvtColor(cv2.imread(image_path.get()), cv2.COLOR_BGR2RGB)
#     # cv_img = cv2.cvtColor(cv2.imread("input_1_2.jpg"), cv2.COLOR_BGR2RGB)
#     # cv_img = cv2.imread(image_path.get())
#
#     cv_img = cv2.resize(cv_img, (img_w, img_h))
#     imageA= cv_img
#
#     # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
#     photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
#
#     # Add a PhotoImage to the Canvas
#     canvasG.create_image(img_w+10, 0, image=photo, anchor=tk.NW, tags="imageB")
#
#     if imageALoaded == True:
#         clearSelection()
#
#     imageALoaded = True
#
#     tk.mainloop()

def clearSelection():
    global canvasG, img_w, img_h, imageB, imageA, matchesA, matchesB
    global inA,inB
    
    canvasG.delete("all")
    matchesA.clear()
    matchesB.clear()
    
    inA = inB = False
    
    photoA = ImageTk.PhotoImage(image = Image.fromarray(imageA))
    canvasG.create_image(0, 0, image=photoA, anchor=tk.NW, tags="imageA")
    
    photoB = ImageTk.PhotoImage(image = Image.fromarray(imageB))
    canvasG.create_image(img_w+10, 0, image=photoB, anchor=tk.NW, tags="imageB")
    
    tk.mainloop()

def errorPopup(msg):
    popup = tk.Tk()
    popup.wm_title("Error!")
    label = tk.Label(popup, text=msg)
    label.pack(fill='x', padx=50, pady=5)
    B1 = tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    
def matchKeyPoints():
    global matchesA, matchesB, imageB, imageA
    
    if imageBLoaded == False or imageALoaded == False:
        errorPopup("First select two images!")
        return
    
    if len(matchesA) < 4:
        errorPopup("Not Sufficient Points selected\n Select atleast 4 correspondances!")
        return
    
    ptsA = np.float32(matchesA)
    ptsB = np.float32(matchesB)
    np.savetxt('matchesA.csv', delimiter=',')

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

#creating main window and widgets
window=tk.Tk() 
  
#getting screen width and height of display 
window_width= window.winfo_screenwidth()  
window_height= window.winfo_screenheight() 

img_w = int((window_width-40)/2) -10
img_h = window_height - 300

#setting tkinter window size 
window.geometry("%dx%d" % (window_width, window_height)) 
window.title("Image Stitching - Kartik Saini") 

# image path input
data_root_path = Path.cwd() / 'Data'
tk.Label(window, text='Path: '+str(data_root_path)).grid(row=0)
# image_root = tk.StringVar()
# e1 = tk.Entry(window, textvariable=image_root, width=50)
# e1.grid(row=0, column=1)

# tk.Label(window, text='Image File base').grid(row=0, column=1)
image_id = tk.StringVar()
e2 = tk.Entry(window, textvariable=image_id, width=50)
e2.grid(row=0, column=1)

tk.Label(window, text='_001.tiff bzw. _070.tiff').grid(row=0, column=2)

loadImages = partial(loadImages, data_root_path, image_id)
b1 = tk.Button(window, text='Load', width=10, command=loadImages)
b1.grid(row=0, column=3)

# window.grid_columnconfigure(3, minsize=200)


# loadImage2 = partial(loadImage2, img2)
# b2 = tk.Button(window, text='Load', width=10, command=loadImage2)
# b2.grid(row=0, column=6)


canvasG = tk.Canvas(window, width=2*img_w+10, height=img_h, bg="grey") 
canvasG.place(x = 10, y=50) 
canvasG.bind("<Button 1>",selectGlobalCanvas)

b3 = tk.Button(window, text='Undo Selection', width=30, command=clearSelection)
b3.place(x=300, y=img_h+100)

b4 = tk.Button(window, text='Save feature points', width=30, command=matchKeyPoints)
b4.place(x=window_width - 500, y=img_h+100)


window.mainloop() 
