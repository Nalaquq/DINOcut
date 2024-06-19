import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from colorfilters import HSVFilter #<= See the maskFinder function for info

#The HanCo .zip files are far too big to push via a branch, so I kept one of the folders of each rgb, mask_hand, and mask_fg for you to check out & the .zip folders if you want any more hands pictures
def importData(file):
  try: 
    home='./noise'
    os.chdir(home)
    file_name = file
    with ZipFile(file_name, 'r') as zip:
      zip.extractall()
      print(f'Your files in {file} have been unzipped. Here are the new files in {home}:')
      print(os.listdir())
  except: 
    print('zipped directory could not be found. Did you upload the file to /content?. If so, make sure it has loaded completely')
    pass
  
##Makeshift folder function
#Will most likely be replaced by the argument parser setup you made in crop.py, but I didn't want to fool w/ it
def enterFolder():
    #Input statement may be a bit clunky
    folder = input("Enter name of parent image and mask folder (EX: for 'noise/test/images', enter 'noise/test'):")
    dir = os.chdir(folder)
    return(dir)

'''Figure out the min and max H, S, & V values with sliders to create a mask for any color
    - Specific to OpenCV b/c of they keep bit sizes small, their hue slider is 180°, not 360°
    - Just a tool to find values & not necessary for creating masks, so the colorfilters library shouldn't be necessary
    - This is the install link if you want to check it out => $ pip install git+https://github.com/alkasm/colorfilters'''
def maskFinder():
    img = cv2.imread('noise/test/images/hat001.jpg')
    window = HSVFilter(img)
    window.show() 
    print(f"Image filtered in HSV between {window.lowerb} and {window.upperb}.")


def maskMaker():
    #Getting into the image folder
    enterFolder()
    cwd = os.getcwd()
    os.chdir('./images')
    imgList = os.listdir()

    #Iterating through the images in the image folder
    for x in imgList:
        img = cv2.imread(x)
        imgCopy = np.copy(img)
        imgCopy = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)

        #Switching each image to HSV (as I find the color space more intuitive for masking)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        '''ANY DIFFERENCE IN FILTERS (green stays the same for all)
            GENERAL - green[50, 45, 100][72, 169, 188], teal[72, 8, 120][107, 90, 205]
            CARD - teal[72, 43, 85][107, 73, 205]
            TAPE - teal[72, 8, 120][141*, 90, 205]
            *Revert to general value (107) to inclue label in the object'''

        #Mask for the background green screen
        lowerGreen = np.array([50, 45, 100])    
        upperGreen = np.array([72, 169, 188])

        #I needed a green shirt so the podium the items sat on was easier to mask out, but the closest I had was a green shirt that turned out more tealish gray
        lowerTeal = np.array([72, 8, 120])
        upperTeal = np.array([107, 90, 205])

        #Creating the two masks, combining them, & applying them to the OG image
        greenMask = cv2.inRange(hsv, lowerGreen, upperGreen)
        tealMask = cv2.inRange(hsv, lowerTeal, upperTeal)
        mask = cv2.bitwise_or(greenMask, tealMask)
        maskedImg = cv2.bitwise_and(imgCopy, imgCopy, mask=mask)

        #Grayscale funtimes
        maskedImg = cv2.cvtColor(maskedImg, cv2.COLOR_RGB2GRAY)

        #Saving the masked image in the corresponding mask folder & returning to the image folder to start the whole process over again with the next image
        os.chdir(cwd)
        os.chdir('./masks')
        imgName = os.path.splitext(x)
        cv2.imwrite(f'{imgName[0]}.jpg', maskedImg)
        os.chdir(cwd)
        os.chdir('./images')
    print("All masks have been made and are stored in the class's mask folder")

#Yay! Functional programming
maskMaker()

'''Thoughts and Recap: The actual programming & masking, while somewhat tedious, wasn't too terrible. 
The hardest part and the thing I would redo if I had the time would be the images. 
I didn't realize (even after a second photoshoot) how important color differentiation and proper lighting truly is. 
If done properly, the right photo makes the masking much easier.
Idk if the ring masks can be used because of how similar the shirt and ring colors are.'''

'''P.S. The test folder was a mismatch of cropped and normal items used to create a general mask. 
It can be deleted, but I kept it in case you would like to take a look at some of my process.'''
'''P.P.S. The people behind HanCo have been grateful enough to provide us with masks for each one of their hand images. 
Idk if the foreground or just the hand masks are necessary for our purposes (however, I don't think we need both).'''
