from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import time
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = YOLO("yolo11x.pt")
model.fuse()
CLASS_NAMES_DICT = model.model.names
classes= [2, 3, 5, 7]  
cap = cv2.VideoCapture('C:/Users/USUARIO/Documents/Recursos/56310-479197605_small.mp4')
pTime=0

while True: 
    success, img = cap.read() 
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) 
    cv2.waitKey(1) 
    box_annotator = sv.BoxAnnotator(thickness=6)
    
    # Detección de objetos con YOLO 
    results = model(img, stream=True)
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id,classes)]
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        labels = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            label = f"{CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            labels.append(label)
        
        for box, label in zip(detections.xyxy, labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.6,          
                    color=(255, 255, 255),
                    thickness=2)
        cv2.imshow("Image", annotated_frame)  
    if not success:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break