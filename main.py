import cv2
import pytesseract
import pandas as pd
import time
import requests
import numpy as np
import imutils
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://172.20.10.7:8080/shot.jpg"



while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    image = cv2.imdecode(img_arr, -1)
    image = imutils.resize(image, width=720, height=480)
    cv2.imshow("Android_cam", image)

    ################### Converting the input image to greyscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("greyed image", gray_image)
    #################### Reducing the noise in the greyscale image
    smooth = cv2.bilateralFilter(gray_image, 11, 17, 17)
    #cv2.imshow("smoothened image", smooth)
    #################### Detecting the edges of the smoothened image
    edged = cv2.Canny(smooth, 30, 200)
   # cv2.imshow("edged image", edged)
    #################### Finding the contours from the edged image
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
   # cv2.imshow("contours", image1)

    #################### Finding the top 30 contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    screenCnt = None
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    #cv2.imshow("Top 30 contours", image2)

    #################### Finding the contour with four sides
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(image, "numberplate", (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

            #################### croping numberplate
            new_img = image[y:y + h, x:x + w]
            cv2.imwrite("numberplate.jpg", new_img)
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)




        ################## saving the numberplate
        Cropped_loc = "numberplate.jpg"


        ################## reading text from cropped numberplate
        plate = pytesseract.image_to_string(Cropped_loc, lang='eng')

        if plate:
           cv2.imwrite("car/carimage.jpg", image)
           cv2.imwrite("number plate/numberplate.jpg", new_img)
           cv2.imshow("image with detected license plate", image)
           cv2.imshow("cropped", cv2.imread(Cropped_loc))
           print("Number plate is:", plate)

           ################# Data is stored in CSV file
           raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'v_number': [plate]}
           df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
           df.to_csv('data.csv')

        else:
             print("no plateplate detected")
             break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break