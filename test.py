import multi_estimation_module
import cv2

cap = cv2.VideoCapture(0)  
Descript = multi_estimation_module.Description(False, False)

human = []
pose = []
while True:                                         
    _, Image = cap.read()    
    multi_estimation_module.Multi_people_estimation(Image, Descript, human, pose)   
