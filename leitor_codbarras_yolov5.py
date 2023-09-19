"""Leitor de código de barras com YOLO para detecção dos códigos."""

import cv2
import numpy as np
from pyzbar.pyzbar import decode


def bbox(img, box):
    """Calcula coordenadas das caixas delimitadoras."""
    x = int(box[0])
    y = int(box[1])
    w = int(box[2])
    h = int(box[3])
    roi = img[y:y+h, x:x+w]
    return roi


url = 0
cap = cv2.VideoCapture(url)
net = cv2.dnn.readNetFromONNX("barcode.onnx")
file = open("labels.txt","r")
classes = file.read().split('\n')

while True:
    img = cap.read()[1]
    if img is None:
        break
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.7:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.7:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
                roi = bbox(img, box)
                barra = decode(roi)
                if barra == []:
                    print('Vazio')
                else:
                    print(barra[0].data.decode())
                    print(barra[0].polygon)
                    print(barra[0].rect)
                    print(barra[0].type)

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.7,0.7)

    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + " {:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0), 1)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,255,0), 2)

    cv2.imshow("Barcode",img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break