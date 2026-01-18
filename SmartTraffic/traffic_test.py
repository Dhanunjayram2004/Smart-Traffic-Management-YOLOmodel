import cv2
from ultralytics import YOLO
import math
model=YOLO('yolov8n.pt')
cap=cv2.VideoCapture(r"C:\Users\vanig\OneDrive\Desktop\SmartTraffic\highway-auto-traffic-road-drive-vehicles-speed-driving-a-car-movement.jpg")
while True:
    vehicles_count=0;
    success,img=cap.read()
    result=model(img,stream=True)
    for r in result:
        boxes=r.boxes
        for box in boxes:
            cls=int(box.cls[0])
            conf=math.ceil((box.conf[0]*100))/100
            if(cls ==2 or cls==3 or cls==5 or cls==7)and conf>0.5:
                vehicles_count += 1
                x1,y1,x2,y2= box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)

    if vehicles_count > 8:
        signal_color = (0, 0, 255) 
        timer_text = "Heavy Traffic: 60s Green"
    else:
        signal_color = (0, 255, 0) 
        timer_text = "Normal Traffic: 20s Green"

  
    cv2.circle(img, (50, 50), 30, signal_color, -1) 
    cv2.putText(img, timer_text, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)

    cv2.imshow("Detection Test: ", img)
    
    if cv2.waitKey(30)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
