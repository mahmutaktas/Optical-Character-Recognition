import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import copy

def CropTheWhiteBGForBWImages(image):
    th, threshed = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    x, y, w, h = cv2.boundingRect(cnt)
    dst = image[y:y + h, x:x + w]
    return dst

def DivideLettersForAlphabetImages(image):

    height, width = image.shape# Getting the input image's sizes

    prev_i = 0 #Set the inital letter bound the zero

    #Create a "sentence" string with the same exact order of the alphabet image
    sentence = "thequickbrownfoxjumpsoverthelazydogTHEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
    alphabet_ctr  = 0

    # #Iterating theese loops to extract each letter on the line
    for i in range(width):

        if image[0, i] == 255:
            for j in range(height):

                sum = image[j, i]
                if sum != 255: #if the current current pixel is black iterate the next column
                    break

                # If there is no black pixel until the line border of the line then set the letter border
                elif j == height-1:

                    #Extract the current letter
                    imm = image[0:j, prev_i:i]
                    hei2, wid2 = imm.shape
                    if hei2 != 0 and wid2 > 2:
                        imm_bw = CropTheWhiteBGForBWImages(imm) #Crop the unnecessary white spaces around the letter
                        #cv2.imshow('black', imm_bw)
                        alphabet_dict.update({sentence[alphabet_ctr]: imm_bw}) #Assign each extracted letter with the corresponding character
                        alphabet_ctr = alphabet_ctr + 1
                        cv2.waitKey(0)

                    prev_i = i
                    i = i+3

def ExtractLetterFromTheImage(image):
    height, width = image.shape # Getting the input image's sizes

    #Setting borders initally zero
    line_bound = 0
    letter_bound = 0

    #Setting dictionary counter
    letter_counter = 0

    #Finding the first black pixels location in order to understand where the text begins
    x, y = FindFirstBlackPixel(image)

    #Iterating first two loop to find line borders
    for i in range(x-1, height):

        for k in range(y-1, width):
            if image[i, k] == 0: #if the current current pixel is black iterate the next line
                break

            # If there is no black pixel until the horizontal end of the image then set the line border
            elif k == width - 1:
                prev_line_bound = line_bound
                line_bound = i

                #Iterating theese loops to extract each letter on the current line
                for a in range(width):

                    for b in range(prev_line_bound, line_bound):

                        if image[b, a] != 255:  #if the current current pixel is black iterate the next column
                            break

                        # If there is no black pixel until the line border of the current line then set the letter border
                        elif b == line_bound - 1:

                            imm = image[prev_line_bound-1:b+1, letter_bound:a] #Extract the letters depending on the borders

                            # If the current cropped image's width is greater than 1 and the height is more than 0 then put the image
                            # inside the input letters dictionary
                            hei2, wid2 = imm.shape
                            if hei2 !=0 and wid2 > 1:

                                imm_bw = CropTheWhiteBGForBWImages(imm)#crop the unnecessary white spaces around the letter
                                #cv2.imshow('black', imm_bw)
                                letters_dict.update({letter_counter: imm_bw})#put the cropped letter inside the dict
                                letter_counter += 1
                                cv2.waitKey(0)

                            letter_bound = a #set the letter bound to current column
                            a += 3 #iterate the column

def FindFirstBlackPixel(image):
    height, width = image.shape #Getting the sizes of the image

    #Iterating left to right until find the first black pixel
    for i in range(height):
        for k in range(width):
            if image[i,k] == 0:
                first_pixel_loc = (i,k)
                return first_pixel_loc

def ResizeAlphabetLetters(alph_dict, height, width):

    for k in alph_dict.keys():
        alph_dict[k] = CropTheWhiteBGForBWImages(alph_dict[k]) #Crop the white area around the letter for one more time
        unchanged_dict[k] = cv2.resize(alph_dict[k], dsize=(width, height), interpolation=cv2.INTER_CUBIC) #Resize the alphabet letter to the size of the input letter

def FindMatchingLetter(letters_dict, alph_dict):

    count = 0
    full_sent = ""
    for i in range(len(letters_dict)): #Getting the input letter
        A = np.empty([52]) #Creating an empty array to fill all the ssim values for one input letter
        for k in alph_dict.keys():

            #Resize the alphabet letters with size of the current letter
            height = len(letters_dict[i])
            width = len(letters_dict[i][0])
            ResizeAlphabetLetters(alph_dict, height, width)

            #Calculating ssim values and putting them inside the array
            score = ssim(unchanged_dict[k], letters_dict[i], multichannel=True)
            A[count] = score
            count = count+1


        count = 0

        #Getting the most similar letter and its index
        max_val = max(A)
        max_idx = np.argmax(A)

        print(f"letter is: {list(alph_dict.keys())[max_idx]} \nscore is: {max_val} ") #Printing the most similar letter and its ssim value
        full_sent += list(alph_dict.keys())[max_idx] #Putting found letter inside the string
        unchanged_dict.clear() #clearing the resized alphabet dict

    print(f"FULL SENTENCE IS: {full_sent}") #printing the result
    return list(alph_dict.keys())[max_idx]


#Create empty dictionaries
alphabet_dict = dict()
unchanged_dict = dict()
letters_dict = dict()

# Reading the images
img = cv2.imread('test_c.png')
alphabet = cv2.imread('calibri_20.png')


# Getting the sizes of the input image
height, width, channels = img.shape

# Converting both to grayscale to convert them fully black and white later
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayAlphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY)

# Converting the images to black and white
(thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
(thresh, bwAlphabet) = cv2.threshold(grayAlphabet, 127, 255, cv2.THRESH_BINARY)

# Extract the letters from the input image and put them in a dict
ExtractLetterFromTheImage(blackAndWhiteImage)

# Extract the letters from the alphabet image and put them in a dict with their corresponding letter as key
DivideLettersForAlphabetImages(bwAlphabet)

# Find and print the matching letters with their SSIMs
FindMatchingLetter(letters_dict, alphabet_dict)

cv2.waitKey(0)




