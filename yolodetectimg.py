import cv2
import argparse
import numpy as np
import os

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default='/path/to/yolov3-tiny.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default='/path/to/yolov3-tiny_finally.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default='/path/to/objects.names')
ap.add_argument('-im', '--image', 
                help = 'path to image file',default='/path/to/image')

args = ap.parse_args()


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23'] - python3 yolodetect.py -c /home/bhavya/codes/darknet/yolov3.cfg -w /home/bhavya/codes/darknet/yolov3_final.weights -cl /home/bhavya/codes/darknet/classes.names

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])+str(" ")+str("{0:.3f}".format(confidence)+str("%"))

    color = (0, 0, 255)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
# Define a window to show the cam stream on it
window_title= "Human Detection in Aerial Image"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)


# Load names classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Generate color for each class randomly
#COLORS = np.random.uniform(255, 255, size=(len(classes), 3))

# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(args.weights,args.config)

# Define video capture for default cam
#cap = cv2.VideoCapture(0)
cap = cv2.imread(args.image)

while cv2.waitKey(1) < 0:
    
    #hasframe, image = cap.read()
    image=cv2.resize(cap, (4032, 3042))
    #image = cap
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.40
    nms_threshold = 10
    
    
    #print(len(outs))
    
    # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
    # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]
    
    # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
    # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
    # and the second output will be = 2028x6=26x26x18 (18=3*6) 
    
    for out in outs: 
        #print(out.shape)
        for detection in out:
            
        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence*100))
                boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   
    head_tail = os.path.split(args.image)
    cv2.imwrite(os.path.join(head_tail[0],"result",head_tail[1]),image)
    cv2.imshow(window_title, image)
    
