import multi_estimation_module
import cv2

cap = cv2.VideoCapture(0)  
Descript = multi_estimation_module.Description(True, True) # создаем описание со следующими параметрами (1 - вывод изображения, 2 - рисование на изображении)

human = []          #массив с номерами людей
pose = []           #массив с указаниями поз для каждого человека по номеру
while(True): 
    ret, frame = cap.read()
    multi_estimation_module.Multi_people_estimation(frame, Descript, human, pose)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
