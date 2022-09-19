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
import os.path

#%%
# Todo: Don't use global variables but encapsulate in objects
canvasG = None
window_width = None
window_height = None
drawLine = False
img_w = None
img_h = None

sizing = None

imageB = None
imageA = None
depth_imageA = None
depth_mapA = None
depth_mapB = None

bbox1 = {'p1': None, 'p2': None, 'box': None, 'buffer': None}
bbox2 = {'p1': None, 'p2': None, 'box': None, 'buffer': None}

imagesLoaded = False

images_root = None
image_idA = None
image_idB = None
images_id = None


def selectGlobalCanvas(event):
    if not imagesLoaded:
        return

    global bbox1, bbox2, canvasG

    x = event.x
    y = event.y

    if x < img_w:
        if bbox1['p1'] is not None:
            bbox1['p2'] = bbox1['buffer']
        bbox1['p1'] = np.array((x, y))
        bbox1['buffer'] = np.array((x, y))

        bbox2['p1'] = bbox1['buffer'] + [(img_w + 10), 0]
        bbox2['buffer'] = bbox1['buffer'] + [(img_w + 10), 0]
        bbox2['p2'] = bbox1['p2'] + [(img_w + 10), 0] if bbox1['p2'] is not None else None

    elif x > img_w + 10:
        if bbox2['p1'] is not None:
            bbox2['p2'] = bbox2['buffer']
        bbox2['p1'] = np.array((x, y))
        bbox2['buffer'] = np.array((x, y))

        bbox1['p1'] = bbox2['buffer'] - [(img_w + 10), 0]
        bbox1['buffer'] = bbox2['buffer'] - [(img_w + 10), 0]
        bbox1['p2'] = bbox2['p2'] - [(img_w + 10), 0] if bbox2['p2'] is not None else None

    def square_bbox(bbox):
        # change either x or y lenght of box to be square in original image
        p1 = bbox['p1']
        p2 = bbox['p2']
        if abs(p1[0]-p2[0]) < abs(p1[1]-p2[1]):
            p1[0] = p2[0] + abs(p1[1]-p2[1])*np.sign(p1[0]-p2[0])
        else:
            p1[1] = p2[1] + abs(p1[0]-p2[0])*np.sign(p1[1]-p2[1])
        bbox['p1'] = p1
        bbox['p2'] = p2
        return bbox

    def make_box(bbox):
        if bbox['box'] is not None:
            for line in bbox['box']:
                canvasG.delete(line)

        lines = []
        lines.append(canvasG.create_line(bbox['p1'][0], bbox['p1'][1], bbox['p2'][0], bbox['p1'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p2'][0], bbox['p1'][1], bbox['p2'][0], bbox['p2'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p2'][0], bbox['p2'][1], bbox['p1'][0], bbox['p2'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['p1'][0], bbox['p2'][1], bbox['p1'][0], bbox['p1'][1], fill="green", width=3))
        lines.append(canvasG.create_line(bbox['buffer'][0]-1, bbox['buffer'][1], bbox['buffer'][0]+1, bbox['buffer'][1],
                                         fill="red", width=2))
        lines.append(canvasG.create_line(bbox['buffer'][0], bbox['buffer'][1]-1, bbox['buffer'][0], bbox['buffer'][1]+1,
                                         fill="red", width=2))
        bbox['box'] = lines
        canvasG.update()
        return bbox

    if bbox1['p1'] is not None and bbox1['p2'] is not None:
        bbox1 = square_bbox(bbox1)
        bbox1 = make_box(bbox1)

    if bbox2['p1'] is not None and bbox2['p2'] is not None:
        bbox2 = square_bbox(bbox2)
        bbox2 = make_box(bbox2)


def loadImages(imgs_root_path, imgs_idA, imgs_idB):
    global canvasG, img_w, img_h, imagesLoaded, images_root, image_idA, image_idB, images_id, sizing,\
        orig_imageA, orig_imageB, imageA, imageB, depth_mapA, depth_mapB, depth_imageA, depth_imageB


    images_root = Path(imgs_root_path.get())
    image_idA = imgs_idA.get()
    image_idB = imgs_idB.get()
    images_id = os.path.commonprefix([image_idA, image_idB])
    path1 = images_root / (image_idA + '.tiff')
    path2 = images_root / (image_idA + '_depth.csv')
    path3 = images_root / (image_idB + '.tiff')
    path4 = images_root / (image_idB + '_depth.csv')

    def get_sizing_parameter(image_shape):
        if img_w/image_shape[1] < img_h/image_shape[0]:
            return img_w/(image_shape[1])
        else:
            return img_h/(image_shape[0])

    # Load an image A using OpenCV
    orig_imageA = cv2.cvtColor(cv2.imread(str(path1)), cv2.COLOR_BGR2RGB)
    sizing = get_sizing_parameter(orig_imageA.shape)
    shape = (int(orig_imageA.shape[1]*sizing), int(orig_imageA.shape[0]*sizing))
    imageA = cv2.resize(orig_imageA, shape)
    photo1 = ImageTk.PhotoImage(image=Image.fromarray(imageA))

    # Load depth map from csv using numpy
    depth_mapA = np.genfromtxt(path2)
    shape = (int(depth_mapA.shape[1]*sizing), int(depth_mapA.shape[0]*sizing))
    depth_imageA = cv2.resize(depth_mapA, shape)
    min_val, max_val = depth_imageA.min(), depth_imageA.max()
    depth_imageA = (depth_imageA - min_val)/(max_val-min_val)*128
    photo2 = ImageTk.PhotoImage(image=Image.fromarray(depth_imageA))

    # Load an image B using OpenCV
    orig_imageB = cv2.cvtColor(cv2.imread(str(path3)), cv2.COLOR_BGR2RGB)
    shape = (int(orig_imageB.shape[1]*sizing), int(orig_imageB.shape[0]*sizing))
    imageB = cv2.resize(orig_imageB, shape)
    photo3 = ImageTk.PhotoImage(image=Image.fromarray(imageB))

    # Load depth map from csv using numpy
    depth_mapB = np.genfromtxt(path4)
    shape = (int(depth_mapB.shape[1]*sizing), int(depth_mapB.shape[0]*sizing))
    depth_imageB = cv2.resize(depth_mapB, shape)
    min_val, max_val = depth_imageB.min(), depth_imageB.max()
    depth_imageB = (depth_imageB - min_val)/(max_val-min_val)*128
    photo4 = ImageTk.PhotoImage(image=Image.fromarray(depth_imageB))

    # Add a PhotoImage to the Canvas
    canvasG.create_image(0, 0, image=photo2, anchor=tk.NW, tags="image2")
    canvasG.create_image(img_w+10, 0, image=photo4, anchor=tk.NW, tags="image4")
    canvasG.create_image(0, img_h+10, image=photo1, anchor=tk.NW, tags="image1")
    canvasG.create_image(img_w+10, img_h+10, image=photo3, anchor=tk.NW, tags="image3")


    if imagesLoaded == True:
        clearSelection()
    imagesLoaded = True

    tk.mainloop()


def clearSelection():
    global canvasG, bbox1, bbox2

    canvasG.delete("all")
    bbox1 = {'p1': None, 'p2': None, 'box': None, 'buffer': None}
    bbox2 = {'p1': None, 'p2': None, 'box': None, 'buffer': None}

    photo1 = ImageTk.PhotoImage(image=Image.fromarray(imageA))
    photo2 = ImageTk.PhotoImage(image=Image.fromarray(depth_imageA))
    photo3 = ImageTk.PhotoImage(image=Image.fromarray(imageB))
    photo4 = ImageTk.PhotoImage(image=Image.fromarray(depth_imageB))

    # Add a PhotoImage to the Canvas
    canvasG.create_image(0, 0, image=photo2, anchor=tk.NW, tags="image2")
    canvasG.create_image(img_w+10, 0, image=photo4, anchor=tk.NW, tags="image4")
    canvasG.create_image(0, img_h+10, image=photo1, anchor=tk.NW, tags="image1")
    canvasG.create_image(img_w+10, img_h+10, image=photo3, anchor=tk.NW, tags="image3")

    tk.mainloop()

def errorPopup(msg):
    popup = tk.Tk()
    popup.wm_title("Error!")
    label = tk.Label(popup, text=msg)
    label.pack(fill='x', padx=50, pady=5)
    B1 = tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    
def process_and_export_data():

    if not imagesLoaded:
        errorPopup("First select two images!")
        return

    # prepare cropping
    cropbox = [None, None, None, None]
    cropbox[0], cropbox[2] = (bbox1['p1'][0], bbox1['p2'][0]) \
        if bbox1['p1'][0] < bbox1['p2'][0] else (bbox1['p2'][0], bbox1['p1'][0])
    cropbox[1], cropbox[3] = (bbox1['p1'][1], bbox1['p2'][1]) \
        if bbox1['p1'][1] < bbox1['p2'][1] else (bbox1['p2'][1], bbox1['p1'][1])
    def imcrop(img, bbox):
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            errorPopup("Can't crop outside of image")
        return img[y1:y2, x1:x2]


    for image, extention in zip([imageA, imageB], ['A', 'B']):
        path = images_root / (images_id + extention +'.tiff')
        if path.exists():
            errorPopup('! Warning - file already exists - {} and following '
                       'won\'t be saved!'.format(str(path)))
            return
        image_cropped = imcrop(image, cropbox)
        image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_cropped)

    for depth_map, extention in zip([depth_mapA, depth_mapB], ['A', 'B']):
        path = images_root / (images_id + extention + '.csv')
        if path.exists():
            errorPopup('! Warning - file already exists - {} and following '
                       'won\'t be saved!'.format(str(path)))
            return
        depth_map_cropped = imcrop(depth_map, cropbox)
        np.savetxt(str(path), depth_map_cropped)
# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

#creating main window and widgets
window=tk.Tk() 
  
#getting screen width and height of display 
window_width= window.winfo_screenwidth()  
window_height= window.winfo_screenheight() 

img_w = int((window_width-50)/2) -10
img_h = int(window_height/2) - 120

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
b1.grid(row=0, column=6)


canvasG = tk.Canvas(window, width=2*img_w+10, height=2*img_h+10, bg="grey")
canvasG.place(x = 10, y=50) 
canvasG.bind("<Button-1>", selectGlobalCanvas)

b3 = tk.Button(window, text='Undo Selection', width=30, command=clearSelection)
b3.place(x=300, y=2*img_h+80)

b4 = tk.Button(window, text='Crop images', width=30, command=process_and_export_data)
b4.place(x=window_width - 500, y=2*img_h+80)


window.mainloop() 
