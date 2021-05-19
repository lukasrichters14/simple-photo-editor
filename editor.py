# Simple Photo Editor
# The simple photo editor is exactly as it sounds, it is designed to be a
# hands-off programatic bulk photo editor.

import cv2
import sys
import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from tqdm import tqdm

CASCADE_FILE = 'assets/face_cascade.xml'


def do_cascade(image, cascade, scale_factor):
    faces = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=5,
            minSize=(30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
    return faces


def main():
    # Load the cascade
    cascade = cv2.CascadeClassifier(CASCADE_FILE)

    # Get images from the user
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames()

    # Get height and width of the crop
    width = simpledialog.askinteger('Input Width', 'Width:')
    height = simpledialog.askinteger('Input Height', 'Height:')
    
    # Keep track of any files that fail to save
    failed_saves = []

    # Load the image and do an initial resize
    for filename in tqdm(files):
        img = cv2.imread(filename)
        path = os.path.split(filename)

        th = int(height + (height / 2))
        tw = int(img.shape[1] * th / img.shape[0])
        img = cv2.resize(img, (tw,th), interpolation=cv2.INTER_AREA)
     
        # Convert to grayscale for the cascade  
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Search for faces
        sf = 1.1
        faces = do_cascade(gray_img, cascade, sf)

        # For these specific photos, there should only be one face
        while len(faces) != 1 and sf < 2:
            faces = do_cascade(gray_img, cascade, sf)
            sf += 0.1

        # Crop centered on each face
        if len(faces) == 1:
            # Get values for face
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
            # Add a border around the face
            border = 250
            w += x + border
            h += y + border
            y -= border
            x -= border
             
            # Ensure user-supplied width and height doesn't exceed the image size
            if y < 0:
                y = 0
            if h >= img.shape[0]:
                h = img.shape[0] - 1
            if x < 0:
                x = 0
            if w >= img.shape[1]:
                w = img.shape[1] - 1

            cropped = img[y:h, x:w]
            final = cv2.resize(cropped, (width,height), interpolation=cv2.INTER_AREA)
            
            # Make save folder if it doesn't already exist
            if not os.path.exists(os.path.join(path[0], 'edited')):
                os.mkdir(os.path.join(path[0], 'edited'))
            # Save the image
            if not cv2.imwrite(os.path.join(path[0], 'edited', path[1]), final):
                failed_saves.append(path[1])
    
    print('Finished.')
    if len(failed_saves) > 0:
        print('These files failed to save.')
        for i in failed_saves:
            print('\t' + i)

    user = input('Press any key to continue.')


if __name__ == '__main__':
    main()

