"""
@author: Kartik Saini
Modified by CDS
"""

#%%

# importing tkinter gui 
import tkinter as tk 
import cv2
import numpy as np
from PIL import ImageTk, Image
from functools import partial
from pathlib import Path
import json

#%%
# Todo: Don't use global variables but encapsulate in objects
canvasG = None
window_width = None
window_height = None
drawLine = False
img_w = None
img_h = None
original_image_dimA = None
original_image_dimB = None

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


imagesLoaded = False

images_root = None
images_id = None

def selectGlobalCanvas(event):
    
    if not imagesLoaded:
        return
    
    global x1,y1,x2,y2, ix1, iy1, ix2, iy2, matchesA, matchesB
    global canvasG, drawLine, img_w, img_h, inA, inB
    x = event.x
    y = event.y
    
    if x < img_w:
        x1 = x
        y1 = y
        
        ix1 = x
        iy1 = y
        inA = True
        
        canvasG.create_line(event.x-5, event.y, event.x+5, event.y, fill="blue", width=1)
        canvasG.create_line(event.x, event.y-5, event.x, event.y+5, fill="blue", width=1)

    elif x > img_w+10:
        x2 = x
        y2 = y
        
        ix2 = x-img_w-10
        iy2 = y
        inB = True
        canvasG.create_line(event.x+5, event.y, event.x-5, event.y, fill="yellow", width=1)
        canvasG.create_line(event.x, event.y+5, event.x, event.y-5, fill="yellow", width=1)



    if inA == True and inB == True:
        canvasG.create_line(x1, y1, x2, y2, fill="red", width=1)
        matchesA.append([ix1,iy1])
        matchesB.append([ix2,iy2])
        inA = False
        inB = False




def loadImages(imgs_root_path, imgs_id):
    global canvasG, img_w, img_h, imageB, imageA, imagesLoaded, images_root,\
        images_id, original_image_dimA, original_image_dimB

    images_root = Path(imgs_root_path.get())
    images_id = imgs_id.get()
    path1 = images_root / (images_id + 'A.tiff')
    path2 = images_root / (images_id + 'B.tiff')

    # Load an image using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path1)), cv2.COLOR_BGR2RGB)
    original_image_dimA = cv_img.shape
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    imageA = cv_img

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photoA = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    # Add a PhotoImage to the Canvas
    canvasG.create_image(0, 0, image=photoA, anchor=tk.NW, tags="imageA")


    # Load an image using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path2)), cv2.COLOR_BGR2RGB)
    original_image_dimB = cv_img.shape
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    imageB = cv_img

    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photoB = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    # Add a PhotoImage to the Canvas
    canvasG.create_image(img_w+10, 0, image=photoB, anchor=tk.NW, tags="imageB")

    if imagesLoaded == True:
        clearSelection()
    imagesLoaded = True

    tk.mainloop()


def clearSelection():
    global canvasG, img_w, img_h, imageB, imageA, matchesA, matchesB, images_root, images_id, bbox1, bbox2
    global inA, inB
    
    canvasG.delete("all")
    matchesA.clear()
    matchesB.clear()

    
    inA = inB = False
    
    photoA = ImageTk.PhotoImage(image=Image.fromarray(imageA))
    canvasG.create_image(0, 0, image=photoA, anchor=tk.NW, tags="imageA")
    
    photoB = ImageTk.PhotoImage(image=Image.fromarray(imageB))
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
    
def exportMatchData():
    # global matchesA, matchesB, imageB, imageA
    
    if not imagesLoaded:
        errorPopup("First select two images!")
        return
    
    if len(matchesA) < 4:
        errorPopup("Not Sufficient Points selected\n Select atleast 4 correspondances!")
        return

    # undo resize to achive positions on original images
    kxa = original_image_dimA[0]/img_w
    kya = original_image_dimA[1]/img_h
    kxb = original_image_dimB[0]/img_w
    kyb = original_image_dimB[1]/img_h
    matchesA_resized = [[x*kxa, y*kya] for x, y in matchesA]
    matchesB_resized = [[x*kxb, y*kyb] for x, y in matchesB]

    # pack data and save to json file
    save_data = {'pointsA': matchesA_resized,
                 'pointsB': matchesB_resized,
                 'imagesID': images_id,
                 }
    path = images_root / (images_id + '_match_data.json')
    if path.exists():
        errorPopup('! Warning - file already exists - won\'t be saved!')
        return
    with open(path, 'w') as file:
        json.dump(save_data, file)

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

# load images bar
tk.Label(window, text='Root dir: ').grid(row=0)

images_root = tk.StringVar()
images_root.set('/home/cstaerk/Documents/Uni/Studienarbeit/Feature-Point-Matching/Data/')
e1 = tk.Entry(window, textvariable=images_root, width=50)
e1.grid(row=0, column=1)

tk.Label(window, text='Image ID: ').grid(row=0, column=2)

images_id = tk.StringVar()
e2 = tk.Entry(window, textvariable=images_id, width=50)
e2.grid(row=0, column=3)

tk.Label(window, text='A.tiff bzw. B.tiff').grid(row=0, column=4)

loadImages = partial(loadImages, images_root, images_id)
b1 = tk.Button(window, text='Load', width=10, command=loadImages)
b1.grid(row=0, column=5)


canvasG = tk.Canvas(window, width=2*img_w+10, height=img_h, bg="grey") 
canvasG.place(x = 10, y=50) 
canvasG.bind("<Button-1>", selectGlobalCanvas)

b3 = tk.Button(window, text='Undo Selection', width=30, command=clearSelection)
b3.place(x=300, y=img_h+100)

b4 = tk.Button(window, text='Save feature points', width=30, command=exportMatchData)
b4.place(x=window_width - 500, y=img_h+100)


window.mainloop() 
