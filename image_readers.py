import pytesseract
import cv2
from pdf2image import convert_from_path,convert_from_bytes

def get_basic_tocr(image,filename):
    if filename[:-4].lower() == '.pdf':
        image = convert_from_path(image)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.color_BGR2RGB)

    text = pytesseract.image_to_string(img)
    print(text)
    return text

