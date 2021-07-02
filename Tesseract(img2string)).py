import cv2
import pytesseract

# for linux systems/Ubuntu
# sudo apt update
# sudo apt install tesseract-ocr
# sudo apt install libtesseract-dev

#for WINDOWS
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

path = 'images/can-somebody-text-me-im-bored.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#read characters from image
text = pytesseract.image_to_string(img)

print (text)


cv2.imshow('Result',img)
cv2.waitKey(0)