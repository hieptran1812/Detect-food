import numpy as np
import argparse
import time
import cv2
import os
from config import *


class DETECTMODEL(object):
    def __init__(self):
        self.config = config
        self.weights = weights
        self.classes = open(classes).read().strip().split("\n")
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.threshold = 0.3	
        self.confidence = 0.5

    def detect(self, image):
        
        net = cv2.dnn.readNetFromDarknet(self.config, self.weights)

        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
              
                if confidence > self.confidence:
                    
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                 
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        
        return image

if __name__ == "__main__":
    image = cv2.imread('./photo.jpg')
    model = DETECTMODEL()
    image = model.detect(image)
    cv2.imwrite("./result.jpg", image)
