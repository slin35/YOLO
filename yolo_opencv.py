import cv2
import argparse
import numpy as np


# usage: 
# python yolo_opencv.py --image multiple_4.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt --original r_multiple_4.jpg

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-o', '--original', required=True,
                help = 'path to original image')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

    
image = cv2.imread(args.image)

o_img = cv2.imread(args.original)
o_img = cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
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
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

file = open("test_data.txt", "w+")

for i in indices:
    i = i[0]
    box = boxes[i]
    x = round(box[0])
    y = round(box[1])
    w = round(box[2])
    h = round(box[3])
    color = COLORS[class_id]
    cv2.rectangle(o_img, (x,y), (x+w,y+h), color, 2)

    r_value = np.mean(o_img[x:x+w, y:y+h, 0])
    g_value = np.mean(o_img[x:x+w, y:y+h, 1])
    b_value = np.mean(o_img[x:x+w, y:y+h, 2])
    ratio = w/h

    file.write(str(r_value) + " " + str(g_value) + " " + str(b_value) + " " + str(ratio) + "\n")
    
file.close()

o_img = cv2.cvtColor(o_img, cv2.COLOR_RGB2BGR)
cv2.imshow("object detection", o_img)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", o_img)
cv2.destroyAllWindows()

