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

imageB = None
imageA = None
depth_mapA = None
depth_mapB = None

bbox1 = {'p1': None, 'p2': None, 'box': None}
bbox2 = {'p1': None, 'p2': None, 'box': None}

imagesLoaded = False

images_root = None
image_idA = None
image_idB = None

# def selectGlobalCanvas(event):
#
#     if not imagesLoaded:
#         return
#
#     global x1,y1,x2,y2, ix1, iy1, ix2, iy2, matchesA, matchesB
#     global canvasG, drawLine, img_w, img_h, inA, inB
#     x = event.x
#     y = event.y
#
#     if x < img_w:
#         x1 = x
#         y1 = y
#
#         ix1 = x
#         iy1 = y
#         inA = True
#
#         canvasG.create_line(event.x-5, event.y, event.x+5, event.y, fill="blue", width=1)
#         canvasG.create_line(event.x, event.y-5, event.x, event.y+5, fill="blue", width=1)
#
#     elif x > img_w+10:
#         x2 = x
#         y2 = y
#
#         ix2 = x-img_w-10
#         iy2 = y
#         inB = True
#         canvasG.create_line(event.x+5, event.y, event.x-5, event.y, fill="yellow", width=1)
#         canvasG.create_line(event.x, event.y+5, event.x, event.y-5, fill="yellow", width=1)
#
#
#
#     if inA == True and inB == True:
#         canvasG.create_line(x1, y1, x2, y2, fill="red", width=1)
#         matchesA.append([ix1,iy1])
#         matchesB.append([ix2,iy2])
#         inA = False
#         inB = False


def selectGlobalCanvas(event):
    if not imagesLoaded:
        return

    global bbox1, bbox2, canvasG
    x = event.x
    y = event.y

    # def square_bbox(bbox):
    #     kxa = original_image_dimA[0] / img_w
    #     kya = original_image_dimA[1] / img_h
    #     bbox['p1']
    #     # To be implemented


    if x < img_w:
        if bbox1['p1'] is not None:
            bbox1['p2'] = bbox1['p1']
        bbox1['p1'] = np.array((x, y))

        bbox2['p1'] = bbox1['p1'] + [(img_w + 10), 0]
        bbox2['p2'] = bbox1['p2'] + [(img_w + 10), 0] if bbox1['p2'] is not None else None

    elif x > img_w + 10:
        if bbox2['p1'] is not None:
            bbox2['p2'] = bbox2['p1']
        bbox2['p1'] = np.array((x, y))

        bbox1['p1'] = bbox2['p1'] - [(img_w + 10), 0]
        bbox1['p2'] = bbox2['p2'] - [(img_w + 10), 0] if bbox2['p2'] is not None else None

    def make_box(bbox):
        if bbox['box'] is not None:
            for line in bbox['box']:
                canvasG.delete(line)

        lines = []
        lines.append(canvasG.create_line(bbox['p1'][0], bbox['p1'][1], bbox['p2'][0], bbox['p1'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p2'][0], bbox['p1'][1], bbox['p2'][0], bbox['p2'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p2'][0], bbox['p2'][1], bbox['p1'][0], bbox['p2'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p1'][0], bbox['p2'][1], bbox['p1'][0], bbox['p1'][1], fill="green", width=3))
        bbox['box'] = lines
        canvasG.update()
        return bbox

    if bbox1['p1'] is not None and bbox1['p2'] is not None:
        bbox1 = make_box(bbox1)

    if bbox2['p1'] is not None and bbox2['p2'] is not None:
        bbox2 = make_box(bbox2)


def loadImages(imgs_root_path, imgs_idA, imgs_idB):
    global canvasG, img_w, img_h, imageB, imageA, depth_mapA, depth_mapB, imagesLoaded, images_root,\
        image_idA, image_idB, original_image_dimA, original_image_dimB

    images_root = Path(imgs_root_path.get())
    image_idA = imgs_idA.get()
    image_idB = imgs_idB.get()
    path1 = images_root / (image_idA + '.tiff')
    path2 = images_root / (image_idA + '_depth.csv')
    path3 = images_root / (image_idB + '.tiff')
    path4 = images_root / (image_idB + '_depth.csv')

    # Load an image A using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path1)), cv2.COLOR_BGR2RGB)
    original_image_dimA = cv_img.shape
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo1 = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    imageA = cv_img

    # Load depth map from csv using numpy
    depth_mapA = np.genfromtxt(path2)
    # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
    photo2 = ImageTk.PhotoImage(image=Image.fromarray(cv_img))

    # Load an image B using OpenCV
    cv_img = cv2.cvtColor(cv2.imread(str(path3)), cv2.COLOR_BGR2RGB)
    original_image_dimB = cv_img.shape
    cv_img = cv2.resize(cv_img, (img_w, img_h))
    imageB = cv_img

    # Load depth map from csv using numpy
    depth_mapB = np.genfromtxt(path4)

    # Add a PhotoImage to the Canvas
    canvasG.create_image(0, 0, image=photo1, anchor=tk.NW, tags="image1")

    # Add a PhotoImage to the Canvas
    canvasG.create_image(img_w+10, 0, image=photo2, anchor=tk.NW, tags="image2")


    if imagesLoaded == True:
        clearSelection()
    imagesLoaded = True

    tk.mainloop()


def clearSelection():
    global canvasG, img_w, img_h, imageB, imageA, matchesA, matchesB, images_root, images_id, bbox1, bbox2
    global inA, inB
    
    canvasG.delete("all")
    bbox1 = {'p1': None, 'p2': None, 'box': None}
    bbox2 = {'p1': None, 'p2': None, 'box': None}

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

    if not imagesLoaded:
        errorPopup("First select two images!")
        return
    
    # if len(matchesA) < 4:
    #     errorPopup("Not Sufficient Points selected\n Select atleast 4 correspondances!")
    #     return

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

image_idA = tk.StringVar()
e2 = tk.Entry(window, textvariable=image_idA, width=50)
e2.grid(row=0, column=3)

image_idB = tk.StringVar()
e2 = tk.Entry(window, textvariable=image_idB, width=50)
e2.grid(row=0, column=4)

tk.Label(window, text='.tiff bzw. _depth.csv').grid(row=0, column=5)

loadImages = partial(loadImages, images_root, image_idA, image_idB)
b1 = tk.Button(window, text='Load', width=10, command=loadImages)
b1.grid(row=0, column=5)


canvasG = tk.Canvas(window, width=2*img_w+10, height=img_h, bg="grey") 
canvasG.place(x = 10, y=50) 
canvasG.bind("<Button-1>", selectGlobalCanvas)

b3 = tk.Button(window, text='Undo Selection', width=30, command=clearSelection)
b3.place(x=300, y=img_h+100)

b4 = tk.Button(window, text='Crop images', width=30, command=exportMatchData)
b4.place(x=window_width - 500, y=img_h+100)


window.mainloop() 
