# libraries
from calendar import c
from flask import Flask, jsonify
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd
from flask import request
from pdfminer import high_level
from flask_cors import CORS, cross_origin

# declaring application
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# method to preprocess image
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
def Opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def Canny(image):
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

import urllib

@app.route("/timetable", methods=['GET', 'POST'])
def transformImage():
    print(request.json['imageURL'])
    url=request.json['imageURL']
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, -1)
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
    slots={}
    for i in range(0,len(d['text'])):
        if d['text'][i] =='MON':
            break
        else:
            if d['text'][i]=='-':
                temp=d['text'][i-2].split(" ")
                for time in temp:
                    slots[time]=time
    clean=[]
    for i in d['text']:
        if i:
            clean.append(i)
    df=pd.read_excel('Book1.xlsx')
    df = df[df['Slot 1'].notna()]
    for i in df['Slot 1']:
        if i[5:] in slots.keys():
            print('found')
    cS=df['Slot 1'] +'---' + df['Code']
    codes=[]
    for i in cS:
        if i[5:10] in slots.keys() and int(i[5:6]) + 120 not in slots.keys():
            print('found codes are',i[13:])
            codes.append(i[13:])
        else:
            continue
    df['Code'].apply(lambda x: x not in codes)
    temp=df['Code'].apply(lambda x: x not in codes)
    toDrop=temp.loc[lambda x: x==False].index
    clashFree=df.drop(toDrop,axis=0)
    clashFree.to_csv('Book2.csv',index=False)
    data=clashFree.to_dict()
    return data['Course Title']
 

from pdfminer.high_level import extract_text
from io import StringIO, BytesIO
import urllib.request

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf_url(url):
    user_agent=None
    resource_manager = PDFResourceManager()
    fake_file_handle = StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)    

    if user_agent == None:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'

    headers = {'User-Agent': user_agent}
    request = urllib.request.Request(url, data=None, headers=headers)

    response = urllib.request.urlopen(request).read()
    fb = BytesIO(response)

    print(fb)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    for page in PDFPage.get_pages(fb,
                                caching=True,
                                check_extractable=True):
        page_interpreter.process_page(page)


    text = extract_text(fb)

    # close open handles
    fb.close()
    converter.close()   
    fake_file_handle.close()
    return text

@app.route("/transcript", methods=['GET', 'POST'])
def transcriptClash():
    print(request.json['pdfURL'])
    url=request.json['pdfURL']
    stdName=request.json['studentName']
    stdRegNo=request.json['studentRegNo']
    extracted_text=extract_text_from_pdf_url(url)
    data=extracted_text.splitlines()
    if stdRegNo.lower() not in data[2][-9:].lower():
        print(stdRegNo)
        print('reg not matched',data[2][-9:])
        return 'please enter your own transcript'
    else:
        temp=data[34:]
        courses=[]
        for i in range(0,len(temp)):
            if 'SEMESTER' not in temp[i] and temp[i] != 'Credit Hours' and len(temp[i]) > 8:
                print('found',temp[i])
                courses.append(temp[i])
        temp=data[34:]
        grades=[]
    for i in range(0,len(temp)):
        if temp[i]  == 'A' or temp[i]  == 'A-' or temp[i]  == 'B+' or temp[i]  == 'B' or temp[i]  == 'B-' or temp[i]  == 'C+' or temp[i]  == 'C' or temp[i]  == 'C-' or temp[i]  == 'D+' or temp[i]  == 'D'  or temp[i]  == 'F': 
            print('found',temp[i])
            grades.append(temp[i])
    df=pd.DataFrame()
    if len(grades) != len(courses):
        courses.pop()
        courses.pop()
        grades.append('B')
        df.insert(0,'Courses',courses)
        df.insert(1,'Grades',grades)   
        labs=df[df['Courses'].str.contains('Lab')]
        canTake=labs[labs['Grades'].isin(['A','A-','B+'])]
        unidf=pd.read_csv('Book2.csv')
        unidf = unidf[unidf['Slot 1'].notna()]
        tempDF=unidf.iloc[:,[1,2]]
        tempDF=unidf
        allowed=[]
        for i in canTake['Courses']:
            allowed.append(i)
        final=tempDF[tempDF['Course Title'].isin(allowed)]
        final=final.to_dict()
        toReturn=final['Course Title']
        listing=[]  
        for k in toReturn.keys():
            listing.append(toReturn[k])
        return jsonify(listing)
  


if __name__ == "__main__":
    app.run(debug=True)