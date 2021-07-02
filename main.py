import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

im1 = 'images/can-somebody-text-me-im-bored.jpg'
im2 = 'images/distractedbf.jpg'
im3 = 'images/flextape.jpg'
im4 = 'images/expandingbrain.jpg'
im5 = 'images/inhaling.jpg'
im6 = 'images/twogender.png'


def recognize_text(img_path):
    #load and recognize text
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

result = recognize_text(im6)

print (result)

#test contextual spell check

# import contextualSpellCheck
# import spacy
# nlp = spacy.load("en_core_web_trf")
# contextualSpellCheck.add_to_pipe(nlp)

#result will be a list of words as follows:
#(coordinates = [(top_left), (top_right), (bottom_right), (bottom_left)])
#(word)
#(confidence)
