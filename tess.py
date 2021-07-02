import cv2
import pytesseract
import numpy as np
from textblob import TextBlob
import jamspell


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


img = cv2.imread("memes/5tsg7a.jpg")

def compose(a, b):
    return lambda x: b(a(x))

fn = lambda x: remove_noise(get_grayscale(x))

custom_config = r'--oem 1 --psm 6'
res = pytesseract.image_to_string(fn(img), config=custom_config)
print(res)
txtblob = TextBlob(res).correct()
print("---")
print("textblob")
print(txtblob)

corrector = jamspell.TSpellCorrector()
c = corrector.LoadLangModel('en.bin')
print(c)
print("---")
print("jamspell")
jspl = corrector.FixFragment(res)
rsp = res.replace('\n', ' ').replace('  ', ' ').split()
print(rsp)
for i in range(len(rsp)):
    print(corrector.GetCandidates(rsp, i))
print(jspl)
print("---")
print("textblob + jamspell")
jspl = TextBlob(jspl)
print(jspl.correct())

import contextualSpellCheck
import spacy
nlp = spacy.load("en_core_web_trf")
contextualSpellCheck.add_to_pipe(nlp)

lines = res.split("\n")
for l in lines:
    doc = nlp(l)
    print(doc._.outcome_spellCheck)
