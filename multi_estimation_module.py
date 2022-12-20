import cv2                                          # подключаем openCV2
import mediapipe as mp                              # подключаем mediapipe
import time                                         #подключаем time

class Description:
    def __init__ (self, show_video, draw_mode):
        self.show_video = show_video                                  #ключ показа видео
        self.draw_mode = draw_mode                                         #ключ рисования на видео

        self.mp_pose = mp.solutions.pose                         #подключаем раздел распознавания тела
        self.pose = self.mp_pose.Pose(static_image_mode=True)         #объект класса "поза"
        if draw_mode:
            self.mp_draw = mp.solutions.drawing_utils            #подключаем инструмент для рисования
            self.prev_frame_time = 0
            self.next_frame_time = 0


def distance(point1, point2):                       #функция, возвращающая расстояние между точками по модулю
    return abs(point1 - point2)

def BP_estimation(landmarks, Image):                       #функция распознавания позы (Возвращает одно из 6 состояний)
    
    py = [0 for i in range(33)]                         #массив для хранения точек скелета по оси Y
    px = [0 for i in range(33)]                         #массив для хранения точек скелета по оси X

    hand_raised = [0 for i in range(2)]                 #индикатор поднятых рук
    hand_on_head = [0 for i in range(2)]                #индикатор рук на голове
    hand_on_shoulder = [0 for i in range(2)]            #индикатор рук на плечах (для перекрещенных рук)
    elbow_near_hip = [0 for i in range(2)]              #индикатор правильного положения локтей (для перекрещенных рук)

    
    for id, point in enumerate(landmarks) :                           #создаем список с координатами точек
        width, height, color = Image.shape                            #получаем размеры изображения с камеры и масштабируем
        width, height = int(point.x * height), int(point.y * width)

        py[id] = height             #Заполняем массив координатами по оси Y
        px[id] = width              #Заполняем массив координатами по оси X
                
    Good_distance_for_raised_hands = distance(py[12], py[24]) * 5/3     # получаем расстояние, с которым будем сравнивать каждую руку 
                                                                        #(берем расстояние от плеча до бедра и умножаем на кооэффициент, необходимый для заданной точности распознавания)       
    # 0 - правая, 1 - левая                                             #по этому принципу будем работать с каждой позой
    hand_raised[0] = 1 if distance(py[24], py[16]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли правая рука
    hand_raised[1] = 1 if distance(py[23], py[15]) > Good_distance_for_raised_hands else 0 #Распознаем, поднята ли левая рука
    #возвращаем значения в зависимости от комбинаций поднятых рук
    if (hand_raised[0]) and (hand_raised[1]):
        return 1
    if (hand_raised[0]) and not (hand_raised[1]):
        return 2
    if not (hand_raised[0]) and (hand_raised[1]):
        return 3

    # получаем расстояние, с которым будем сравнивать руки на голове по горизонтали и вертикали 
    Good_distance_for_hands_on_head_X = distance(px[8], px[7]) * 8/15 
    Good_distance_for_hands_on_head_Y = distance(py[6], py[9]) * 2
    #Распознаем руки на голове (0 - правая , 1 - левая)
    hand_on_head[0] = 1 if (distance(px[8], px[16]) < Good_distance_for_hands_on_head_X and distance(py[8], py[16]) < Good_distance_for_hands_on_head_Y) else 0
    hand_on_head[1] = 1 if (distance(px[7], px[15]) < Good_distance_for_hands_on_head_X and distance(py[7], py[15]) < Good_distance_for_hands_on_head_Y) else 0
    #если обе руки на голове, выводим сообщение
    if (hand_on_head[0]) and (hand_on_head[1]):
        return 4
       
    #Получаем расстояние по осям X и Y для сравнения кистей на противоположных плечах
    Good_distance_for_cross_hands_X = (distance(px[12], px[11]) / 3)
    Good_distance_for_cross_hands_Y = (distance(px[12], px[11]) / 3)
    #Распознаем руки (0 - правая, 1 - левая) на противоположных плечах по обеим координатам
    hand_on_shoulder[0] = 1 if (distance(px[20], px[11]) < Good_distance_for_cross_hands_X and distance(py[19], py[11]) < Good_distance_for_cross_hands_Y)  else 0
    hand_on_shoulder[1] = 1 if (distance(px[15], px[12]) < Good_distance_for_cross_hands_X and distance(py[15], py[12]) < Good_distance_for_cross_hands_Y)  else 0
    #Получаем расстояние по оси Y для правильного положения локтя (рядом с соответствующим бедром)
    Good_distance_for_elbow_Y = distance(px[24], px[23]) * 3/2
    #Убеждаемся, что локти (0 - правый, 1 - левый) находятся рядом с соответствующими бедрами
    elbow_near_hip[0] = 1 if distance(py[14], py[24]) < Good_distance_for_elbow_Y else 0
    elbow_near_hip[1] = 1 if distance(py[13], py[23]) < Good_distance_for_elbow_Y else 0
    #Выводим на экран распознанные перекрещенные руки
    if (hand_on_shoulder[0] and hand_on_shoulder[1] and elbow_near_hip[0] and elbow_near_hip[1]):
        return 5
    return 0 
    
    
    
def Multi(origin_image, image, people_count, Descript, out_msg_human_id, out_msg_state_id):   #функция распознавания нескольких людей (глубина рекурсии показывает максимальное количество людей)
                                                #В оригинальном изображении хранится наш видеоряд, который в конечном итоге выводится, 
                                                #в копии хранится картинка, в которой распознанные люди закрашиваются с каждым новым вызовом функции
    people_detected = False                     #распознан ли человек    
    results = Descript.pose.process(image)               #получаем результат
    if results.pose_landmarks:                  #если человек распознан, передаем переменной true                
        people_detected = True
    X , Y = [], []                       #создаем массивы, которые в дальнейшем заполним координатами X и Y
    h, w = image.shape[:2]               #получаем масштаб изображения (height и weight)
    cnt = 1
    pose_state = 0                       # состояние позы человека
    body_pose = [                        #массив с расшифровкой состояний
    'BP_UNKNOWN',
    'BP_HANDS_UP',
    'BP_RIGHT_HAND_RAISED',
    'BP_LEFT_HAND_RAISED',
    'BP_HANDS_ON_HEAD',
    'BP_CROSS']
    
    while((cnt <= people_count) and people_detected):                 #вызываем цикл, который будет работать до тех пор, пока распознаются люди, и их кол-во меньше заданного при вызове функции Multi_people_estimation
        out_msg_state_id.append(BP_estimation(results.pose_landmarks.landmark, origin_image))    #вызываем функцию распознавания человека и присваиваем вернувшееся значение переменной состояния
        out_msg_human_id.append(cnt) 
         
       # print(cnt, " : ", body_pose[pose_state])                      #выводим на экран номер человека и расшифровку его позы 
        
        if Descript.draw_mode:
            Descript.mp_draw.draw_landmarks(origin_image, results.pose_landmarks, Descript.mp_pose.POSE_CONNECTIONS) #рисуем точки и соединяем их, получая скелет на изображении
        
        for i in range(len(results.pose_landmarks.landmark)):         #получаем два массива координат по x и y
            X.append(results.pose_landmarks.landmark[i].x)
            Y.append(results.pose_landmarks.landmark[i].y)
        people_detected = True                                        #говорим, что удалось распознать человека
        
        #получаем координаты наиольших и наименьших точек по обеим осям (В данном случае наименьшая координата по оси Y будет самой выскокой на изображении, наибольшая координата - самой низкой)
        x1 = int(min(X) * w)
        y1 = int(min(Y) * h)
        x2 = int(max(X) * w)
        y2 = int(max(Y) * h)
        
        padding = 20                                                   #добавляем необходимый отступ вокруг человека 
        x1 = x1 - padding if x1 - padding > 0 else 0
        y1 = y1 - 10 * padding if y1 - 10 * padding > 0 else 0
        x2 = x2 + padding if x2 + padding < w else w
        y2 = y2 + padding if y2 + padding < h else h
        
        if Descript.draw_mode:
            cv2.putText(origin_image, str(cnt), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2) #рисуем номер человека на изображении
            cv2.rectangle(origin_image, (x1,y2), (x2,y1), color = (0,255,0), thickness = 1) #рисуем прямоугольник вокруг человека
        image[y1:y2, x1:x2] = 0                                                             #Делаем каждый пиксель прямоугольника, где есть человек, черным 
        
        cnt += 1                                #увеличиваем счетчик на 1
        people_detected = False                 #сбрасываем переменную на false т.к. от текущего человека получили всю необходимую информацию
        results = Descript.pose.process(image)           #переходим к следующему человеку
        if results.pose_landmarks:              #если удалось распознать, переменной приравниваем значение true. Цикл повторится, если мы не достигли максимального требуемого количества людей.
            people_detected = True
    
def Multi_people_estimation(Image, Descript, out_msg_human_id, out_msg_state_id):                    #Функция распознавания позы нескольких людей (получает на вход изображение)
    img = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)       #делаем копию изображения
    out_msg_human_id.clear()
    out_msg_state_id.clear()
    Multi(Image, img, 5, Descript, out_msg_human_id, out_msg_state_id)                               #вызываем функцию распознавания нескольких людей, внутри нее для каждого человека вызовется функция распознавания позы
    if Descript.draw_mode:    
        Descript.new_frame_time = time.time()
        fps = 1/(Descript.new_frame_time - Descript.prev_frame_time)
        Descript.prev_frame_time = Descript.new_frame_time
    
        fps = int(fps)                                      #Преобразуем строку для вывода на выходное изображение
        fps_str = "FPS: " + str(fps)

        cv2.putText(Image, fps_str, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)   #выводим частоту кадров на изображение
    if Descript.show_video:
        cv2.imshow("cam", Image)                    #выводим изображение на экран
        #cv2.imshow("cam2", img)                    #выводим изображение, в котором удалены люди


