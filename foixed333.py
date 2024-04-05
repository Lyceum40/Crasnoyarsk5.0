import socket
import cv2
import time
import sys
from math import *
import numpy as np
from collections import deque

### IMPORTANT CONSTANTS ###
cameraIndex = 1 #номер камеры
pauseFrame = 0 #при значении 1 для перехода к следующему кадру требуется нажать любую кнопку
bfsAnimation = False #анимация нахождения пути
resize = True #изменение разрешения
cornerarea = 500 #минимальная площадь чёрных маркеров краёв
objectarea = 250 #минимальная площадь объектов
startarea = 4000#минимальная площадь стартовых зон
workingResolution = (1280, 720) #рабочее разрешение
showingResolution = (1280, 720) #разрешение для показа
skipPath = False #пропустить проезд по траектории
baseSpeed = 120 #базовая мощность робота
showMask = False #показывать маску трекинга
printFPS = True #отображать FPS
showError = False #отображать накопленную ошибку
showCommand = True #отображать посылаемые команды
cornerEps = 30 #допустимое дополнительно расстояние, на которое может выступать объект относительно маркеров
green_min_mask = np.array((2, 71, 179),np.uint8)
green_max_mask = np.array((2, 71, 179),np.uint8)
yellow_min_mask = np.array((2, 71, 179),np.uint8)
yellow_max_mask = np.array((2, 71, 179),np.uint8)
start_min_mask = np.array((2, 71, 179),np.uint8)
start_max_mask = np.array((2, 71, 179),np.uint8)
corner_min_mask = np.array((2, 71, 179),np.uint8)
corner_max_mask = np.array((2, 71, 179),np.uint8)
persik_min_mask = np.array((2, 71, 179),np.uint8)
persik_max_mask = np.array((2, 71, 179),np.uint8)
magenta_min_mask = np.array((2, 71, 179),np.uint8)
magenta_max_mask = np.array((2, 71, 179),np.uint8)
### END OF CONSTANTS ###

err = 0 #текущая накопленная ошибка
lasttime = time.time() #последнее обновление ошибки
lastframe = time.time() #последний обработанный кадр
pi = atan2(0, -1) #число пи
serverMACAddress = '78:21:84:7D:A9:F6'
port = 1
sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
sock.connect((serverMACAddress,port))

def bfs(point): #алгоритм поиска в ширину (BFS) (полезно будет почитать Википедию и подобные ресурсы)
    global visited #отметки посещённости точек
    global hsv #изображение
    global p #предок каждой точки
    global q #очередь точек
    
    mex, mey = point #текущая точка
    if (bfsAnimation): #анимация
        hsv[mey][mex] = (0, 255, 255) #покрасить текущую точку
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #преобразовать и отобразить кадр
        if (resize):
            res = cv2.resize(res, showingResolution)
        cv2.imshow('frame', res)
        cv2.waitKey(1 - pauseFrame)
    moves = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]] #возможные движения из данной точки (вверх, вниз, влево, вправо и различные диагонали)
    for x, y in moves: #перебираем движения
        if (mex + x >= 0 and mex + x < m): #проверяем, что точка существует по х
            if (mey + y >= 0 and mey + y < n): #и по y
                if (not visited[mey + y][mex + x]): #и не была посещена до этого
                    if (a[mey + y][mex + x] == 1): #и принадлежит траектории
                        visited[mey + y][mex + x] = True #помечаем её посещённой
                        q.append((mex + x, mey + y)) #добавляем в очередь
                        p[mey + y][mex + x]  = (mex, mey) #запоминаем предка точки (откуда мы в неё пришли)
                        
def find_large_clusters(image, color_lower, color_upper, min_area=300): #найти большие цветные пятна, функцию дал Дима (видимо с Байкала)
    mask = cv2.inRange(hsv, color_lower, color_upper) #применяем маску
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #находим пятна

    large_clusters = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area: #если текущее пятно достаточной площади
            moments = cv2.moments(contour) #запоминаем его центр
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                large_clusters.append((cx, cy)) #добавляем в массив

    return large_clusters

### GEOMETRIC FUNCS ###
def vec(x1, y1, x2, y2): #векторное произведение
    return (x1 * y2) - (x2 * y1)

def scal(x1, y1, x2, y2): #скалярное произведение
    return (x1 * x2) + (y1 * y2)

def angle(x1, y1, x2, y2): #угол между векторами
    return atan2(vec(x1, y1, x2, y2), scal(x1, y1, x2, y2))

def dist(x1, y1, x2, y2): #расстояние между точками
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
### END OF GEO FUNCS ###

def move(xr1, yr1, xr2, yr2, xp, yp, mindist = 22, minangle = 0.25): #считаем необходимое движение
    distance = min(dist(xr2, yr2, xp, yp), dist(xr1, yr1, xp, yp)) #расстояние до требуемой точки
    ang = angle(xr1 - xr2, yr1 - yr2, xp - xr2, yp - yr2) #угол между направлением робота и направлением на точку
    if (distance < mindist): #приехали
        return 0
    if (abs(ang) < minangle): #направление хорошее, едем прямо
        return 1
    if (ang > 0): #поворачиваем налево
        return 3
    return 2 #поворачиваем направо

def pmove(xr1, yr1, xr2, yr2, xp, yp, onlyTurn = False, mindist = 22, p = 50, i = 0): #как move, только с PI регулятором
    global err
    global lasttime
    
    distance = min(dist(xr2, yr2, xp, yp), dist(xr1, yr1, xp, yp))
    ang = angle(xr1 - xr2, yr1 - yr2, xp - xr2, yp - yr2)
    err += ang * (time.time() - lasttime) #накапливаем ошибку
    if (abs(ang) < 0.2): #если мы близко к правде, обнуляем ошибку (помогает избежать дальнейшего влияния старой ошибки, но костыль)
        err = 0
    lasttime = time.time()
    if (showError):
        print('Error: ', err)
    if (not onlyTurn and distance < mindist): #флаг onlyTurn подразумевает необходимость только направить робота в нужную точку, даже если он очень близко к ней
        return 0
    if (onlyTurn and abs(ang) < 0.15 and ang <= 0): #в режиме onlyTurn робот старается повернуться как можно точнее, но находиться не правее цели (поскольку захват слева)
        return 0
    val = 128 - int(ang * p + err * i) #вычислить желаемую степень поворота через PI регулятор и прибавить 128 (тк мы передаём только числа от 0 до 255)
    if (val < -10): #если число сльшком маленькое, поворачиваем в режиме burnout
        return 8
    if (val > 270): #аналогично, только в другую сторону
        return 7
    if (val < 10): #обычный поворот
        return 3
    if (val > 254): #аналогично
        return 2
    if (onlyTurn): #аналогично
        if (val < 128):
            return 3	
        return 2
    return val #ехать по дуге (один мотор быстрее, другой медленнее)

def drive_to_point(x, y, p = 200, mindist = 22, minangle = 0.2, useP = True): #доехать до точки
    global lastframe
    
    while True:
        (xr1, yr1, xr2, yr2) = track_robot() #координаты робота
        
        if (useP):
            action = pmove(xr1, yr1, xr2, yr2, x, y, False, mindist, p) #считаем движение
        else:
            action = move(xr1, yr1, xr2, yr2, x, y, mindist) #считаем движение без PI регулятора

        ### SEND COMMAND ###
        if (action != 0):
            send(action) #посылаем команду роботу
    
        ### SHOW FRAME ###
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.circle(res, (xr1, yr1), 10, (255, 0, 0), -1) #рисуем круги в определённых координатах робота
        cv2.circle(res, (xr2, yr2), 10, (0, 0, 255), -1)
        cv2.circle(res, (x, y), 10, (0, 255, 0), -1) #рисуем целевую точку
        if (resize):
            res = cv2.resize(res, showingResolution)
        cv2.imshow('frame', res)
        cv2.waitKey(1 - pauseFrame)
        if (printFPS):
            print('FPS', 1 / (time.time() - lastframe))
        lastframe = time.time()

        if (action == 0): #приехали
            break

def dovorot(x, y): #довернуть на точку без движения
    global lastframe
    
    while True:
        (xr1, yr1, xr2, yr2) = track_robot()

        action = pmove(xr1, yr1, xr2, yr2, x, y, True) #calculate desired movement

        ### SEND COMMAND ###
        if (action != 0):
            send(action)
        else:
            break

        ### SHOW FRAME ###
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.circle(res, (xr1, yr1), 10, (255, 0, 0), -1)
        cv2.circle(res, (xr2, yr2), 10, (0, 0, 255), -1)
        cv2.circle(res, (x, y), 10, (0, 255, 0), -1)
        
        
        if (resize):
            res = cv2.resize(res, showingResolution)       
        cv2.imshow('frame', res)
        cv2.waitKey(1 - pauseFrame)
        print('FPS', 1 / (time.time() - lastframe))
        lastframe = time.time() 

def setSpeed(speed): #задать мощность
    send(9) #режим установки мощности
    send(speed) #посылаем требуемую мощность

def send(action):
    action = min(255, action) #на случай, если значение выходит из допустимого интервала
    action = max(0, action)
    if (showCommand):
        print(action)
    sock.send(action.to_bytes(1, 'little'))
    
def track_robot():
    global img
    global hsv
    
    ### READ AND RESIZE FRAME ###
    ret, img = cap.read() #считываем кадр
    if (resize):
        img = cv2.resize(img, workingResolution)
    height = img.shape[0] #высота
    width = img.shape[1] #ширина
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #переводим в hsv

    ### DETECTING ROBOT ###
    thresh = cv2.inRange(hsv, persik_min_mask, persik_max_mask) #маска красного
    moments = cv2.moments(thresh, 1) #находим центр красного
    dMo1 = moments['m01']
    dMo2 = moments['m10']
    dArea = moments['m00']
    xr1 = int(dMo2 / dArea)
    yr1 = int(dMo1 / dArea)
    if (showMask):
        cv2.imshow('mask1', thresh)
        cv2.waitKey(1)
    thresh = cv2.inRange(hsv, magenta_min_mask, magenta_max_mask) #маска мадженты
    moments = cv2.moments(thresh, 1) #находим центр мадженты
    dMo1 = moments['m01']
    dMo2 = moments['m10']
    dArea = moments['m00']
    xr2 = int(dMo2 / dArea)
    yr2 = int(dMo1 / dArea)
    if (showMask):
        cv2.imshow('mask2', thresh)
        cv2.waitKey(1)         
        
    return (xr1, yr1, xr2, yr2)

# def sort_points_by_angle(p): #сортируем объекты так, чтобы объезжать их без сильных поворотов
#     for i in range(len(p)): #алгоритм сортировки пузырьком (тк я не умею в питоне нормально сортировать встроенными функциями)
#         for j in range(len(p) - i - 1): #но мы здесь не будем обсуждать приемущества c++
#             if (angle(p[j][0], p[j][1], cty, ctx) > angle(p[j + 1][0], p[j + 1][1], cty, ctx)): #сравниваем углы поворота
#                 p[j], p[j + 1] = p[j + 1], p[j] #меняем местами элементы
#     turn = 0
#     for i in range(len(p) - 1): #ищем место, где робот сильно поворачивает
#         if (abs(angle(p[i][0], p[i][1], p[i + 1][0], p[i + 1][1])) > pi * (7 / 16)): #считаем углы между вектормаи центр->цилиндр
#             turn = i + 1
#     res = p
#     for i in range(len(p)): #сдвигаем массив, чтобы робот не делал больших поворотов
#         res[i] = p[(i + turn) % len(p)]
#     return res

# def get_returning_path(x, y): #считаем координаты точки для объезда цилиндра перед финишем
#     d = dist(cty, ctx, x, y)
#     len = d * sin(pi / 8) / sin(pi * 7 / 4)
#     ang = angle(x - cty, y - ctx, 1, 0)
#     ang += pi / 8
#     return (cty + len * cos(ang), ctx + len * sin(ang))

def nearest_point(a, x, y): #найти ближайшую точку траектории к данной
    mind = 1e9
    point = (0, 0)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if (a[i][j] == 1):
                if (i != 0 and a[i - 1][j] == 1):
                    if (j != 0 and a[i][j - 1] == 1):
                        if (i != len(a) - 1 and a[i + 1][j] == 1):
                            if (j != len(a[i]) - 1 and a[i][j + 1] == 1):
                                if (dist(j, i, x, y) < mind):
                                    mind = dist(j, i, x, y)
                                    point = (j, i)
    return point

def get_center(corner_points):
    maxdist = 0
    point = (0, 0)
    for i in range(len(corner_points)):
        for j in range(len(corner_points)):
            d = dist(corner_points[i][0], corner_points[i][1], corner_points[j][0], corner_points[j][1])
            if (d > maxdist):
                maxdist = d
                point = ((corner_points[i][0] + corner_points[j][0]) // 2, (corner_points[i][1] + corner_points[j][1]) // 2)
    return point

### ОСНОВНОЙ КОД ###

### UDP INITIALIZATION ###
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #инициализация UDP протокола по Wi-Fi

cv2.setNumThreads(24) #задаём кол-во потоков для работы OpenCV

### CAMERA INITIALIZATION ###
cap = cv2.VideoCapture(cameraIndex) #объект камеры
points = [] #список точек траектории
ret, img = cap.read() #считываем кадр
if (resize): #при необходимости меняем разрешение
    img = cv2.resize(img, workingResolution)
height = img.shape[0]
width = img.shape[1]

setSpeed(35)

while True: #показываем картинку в реальном времени для настройки камеры
    ret, img = cap.read()
    if (resize):
        img = cv2.resize(img, workingResolution)
    cv2.circle(img, (width // 2, height // 2), 10, (255, 255, 255), -1) #обозначаем точку центра
    cv2.imshow('frame', img) #отображаем картинку
    if (cv2.waitKey(1 - pauseFrame) & 0xFF == ord('q')): #выходим из цикла, если была нажата клавиша q
        break
    
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #конвертируем в hsv
#setSpeed(30)
### FIND OBJECTS ###
corner_points = find_large_clusters(hsv, corner_min_mask, corner_max_mask, cornerarea) #находим крайние маркеры и объекты
start_objects = find_large_clusters(hsv, start_min_mask, start_max_mask, startarea)

a = [] #массив, помечающий пиксели, принадлежащие траектории
visited = [] #массив пометок о посещении
p = [] #массив предков (погуглите 'восстановление пути в BFS')

### MARK ROUTE ###
for i in range(height): #идём по картинке
    a.append([0] * width) #формируем массивы
    visited.append([False] * width)
    p.append([(-1, -1)] * width)
    for j in range(width):
        h, s, v = hsv[i][j] #получаем h, s и v компоненты пикселя
        if (h > 97 and h < 121 and s > 90 and v > 90): #проверяем, подходят ли они под условия траектории
            hsv[i][j] = (55, 255, 255) #отмечаем их зелёным цветом
            a[i][j] = 1 #помечаем пиксель проходимым

### SHOW FRAME ###
res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
if (resize):
    res = cv2.resize(res, showingResolution)
cv2.imshow('frame', res)
cv2.waitKey(0) #ждём до нажатия любой клавиши

(xr1, yr1, xr2, yr2) = track_robot()

for i in range(len(start_objects)):
    for j in range(i, len(start_objects) - 1):
        if (dist(width // 2, height // 2, start_objects[j][0], start_objects[j][1]) > dist(width // 2, height // 2, start_objects[j + 1][0], start_objects[j + 1][1])):
            start_objects[j], start_objects[j + 1] = start_objects[j + 1], start_objects[j]
            
if (dist(xr1, yr1, start_objects[0][0], start_objects[0][1]) < dist(xr1, yr1, start_objects[1][0], start_objects[1][1])):
    start_objects[0], start_objects[1] = start_objects[1], start_objects[0]

ystx = start_objects[0][0] #координаты стартов для жёлтых и зелёных объектов
ysty = start_objects[0][1]
gstx = start_objects[1][0]
gsty = start_objects[1][1]

(stx, sty) = nearest_point(a, gstx, gsty)
(fnx, fny) = nearest_point(a, ystx, ysty)
(ctx, cty) = get_center(corner_points)
res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
if (resize):
    res = cv2.resize(res, showingResolution)
cv2.circle(res, (gstx, gsty), 10, (120, 120, 0), -1)
cv2.circle(res, (ystx, ysty), 10, (120, 0, 120), -1)
cv2.imshow('frame', res)
cv2.waitKey(0)
print(stx, sty) #стартовые координаты
print(fnx, fny) #финишные координаты

### FIND ROUTE ###
n = height
m = width
q = deque() #создаём очередь для BFS
q.append((stx, sty)) #добавляем точку старта
visited[sty][stx] = True #помечаем её посещённой
while (len(q) > 0): #пока есть точки в очереди
    bfs(q.popleft()) #запускаемся от них
while (p[fny][fnx] != (-1, -1)): #восстанавливаем путь, пока есть предыдущая точка
    points.append([fnx, fny]) #добавляем точку
    fnx, fny = p[fny][fnx] #переходим в её предка

points2 = []
for i in range(len(points)):
    points2.append(points[len(points) - i - 1])
points = points2

print(points) #выводим полученный массив

### SHOW FRAME ###
res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
if (resize):
    res = cv2.resize(res, showingResolution)
cv2.imshow('frame', res)
cv2.waitKey(1 - pauseFrame)

points.append([ystx, ysty])

### FOLLOW THE PATH ###
tekpoint = 0 #текущая точка
while (tekpoint < len(points)): #если не все точки пройдены
    if (not skipPath):
        drive_to_point(points[tekpoint][0], points[tekpoint][1], 200, 25, 0.15, False) #едем в точку
    if (tekpoint + 12 > len(points) and tekpoint != len(points - 1)):
        tekpoint = len(points - 1)
    else:
        tekpoint += 12 #пропускаем 24 точки

send(255)
time.sleep(0.5)

### READ AND RESIZE FRAME ###
ret, img = cap.read()
if (resize):
    img = cv2.resize(img, workingResolution)
height = img.shape[0]
width = img.shape[1]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

green_points = find_large_clusters(hsv, green_min_mask, green_max_mask, objectarea)
yellow_points = find_large_clusters(hsv, yellow_min_mask, yellow_max_mask, objectarea)

### FILTER OBJECTS ###
meancorner = 0 #старое название, это расстояние до самой дальней крайней точки
for i in range(len(corner_points) - 1): #сортировка крайних точек 'пузырьком'
    for j in range(i, len(corner_points) - 1):
        if (dist(ctx, cty, corner_points[i][0], corner_points[i][1]) > dist(ctx, cty, corner_points[i + 1][0], corner_points[i + 1][1])): #сравниваем расстояния от центра до точек
            corner_points[i], corner_points[i + 1] = corner_points[i + 1], corner_points[i]
meancorner = dist(ctx, cty, corner_points[len(corner_points) - 1][0], corner_points[len(corner_points)  - 1][1]) #расстояние до самой дальней точки
print(meancorner)

cube = (0, 0)
maxgreen = 0 #расстояние до самого далёкого зелёного объекта
max_green_point = (0, 0) #координаты самого далёкого зелёного объекта
maxyellow = 0 #аналогично, только для жёлтых объектов
max_yellow_point = (0, 0)
minsummdist = 1e9
er_list = [] #список на удлаение (объекты за полем)
for i in range(len(green_points)): # проходим по зелёным точкам
    dpoint = dist(ctx, cty, green_points[i][0], green_points[i][1]) #считаем расстояние от центра
    print(dpoint)
    if (dpoint > meancorner + 30): #если объект слишком далеко, добавляем в список для удаления
        er_list.append(i)
    elif (dpoint > maxgreen): #обновляем самый далёкий объект
        maxgreen = dpoint
        max_green_point = green_points[i]
    dsumm = dist(gstx, gsty, green_points[i][0], green_points[i][1]) + dist(ystx, ysty, green_points[i][0], green_points[i][1])
    if (dsumm < minsummdist):
        minsummdist = dsumm
        cube = green_points[i]

plus = 0 #сдвиг индексов из-за удаления
for i in er_list: #удаляем объекты из списка на удаление
    cv2.circle(res, green_points[i - plus], 10, (255, 255, 0), -1) #помечаем их кружками
    green_points.pop(i - plus) #удаляем элемент
    plus += 1 #увеличиваем сдвиг

er_list = [] #аналогично для жёлтых объектов
for i in range(len(yellow_points)):
    dpoint = dist(ctx, cty, yellow_points[i][0], yellow_points[i][1])
    print(dpoint)
    if (dpoint > meancorner + cornerEps):
        er_list.append(i)
    elif (dpoint > maxyellow):
        maxyellow = dpoint
        max_yellow_point = yellow_points[i]
plus = 0
for i in er_list:
    cv2.circle(res, yellow_points[i - plus], 10, (255, 255, 0), -1)
    yellow_points.pop(i - plus)
    plus += 1

### MARK POINTS ###
for (x, y) in corner_points: #помечаем маркеры краёв кружками
    cv2.circle(res, (x, y), 10, (255, 255, 255), -1)
for (x, y) in green_points: #зелёные объекты
    cv2.circle(res, (x, y), 10, (255, 0, 255), -1)
for (x, y) in yellow_points: #жёлтые объекты
    cv2.circle(res, (x, y), 10, (0, 255, 255), -1)
cv2.circle(res, (ctx, cty), int(meancorner + cornerEps), (0, 0, 0), 2) #радиус отсечки объектов
cv2.circle(res, (ctx, cty), 10, (255, 255, 255), -1) #центр поля
if (resize):
    res = cv2.resize(res, showingResolution)
cv2.imshow('frame', res)
cv2.waitKey(1 - pauseFrame)

send(255) #остановка робота

# if (color): #если желаемые объекты жёлтые, меняем местами два массива
#     green_points, yellow_points = yellow_points, green_points #далее будет предполагаться, что нужные объеткы - зелёные

# green_points = sort_points_by_angle(green_points) #сортируем точки

for (x, y) in green_points:
    point = (x, y) #требуемая точка
    if (point == cube):
        continue
    setSpeed(35)
    drive_to_point(ctx, cty, 100, 30, 0.17) #едем в центр
    send(5) #открываем захват
    d = dist(ctx, cty, x, y) #расстояние до точки
    x = int(ctx + (x - ctx) * (1 + 30 / d)) #вычисляем координаты точки на 20 пикселей дальше от центра
    y = int(cty + (y - cty) * (1 + 30 / d)) # (чтобы точно забрать цилиндр)
    drive_to_point(x, y, 100, 35, 0.17) #доехать до точки
    #dovorot(x, y) #довернуть для точного захвата
    send(255) #остановиться
    send(4) #закрыть захват
    setSpeed(40)
    drive_to_point(ctx, cty, 200, 30, 0.17) #вернуться в центр

d = dist(ctx, cty, gstx, gsty)
x = int(ctx + (gstx - ctx) * (1 + 30 / d)) #вычисляем координаты точки на 20 пикселей дальше от центра
y = int(cty + (gsty - cty) * (1 + 30 / d)) # (чтобы точно забрать цилиндр)    
drive_to_point(x, y, 200, 30, 0.17) #отвезти на базу
send(255) #остановиться
send(5) #открыть захват
send(6) #отъехать назад

drive_to_point(ystx, ysty, 100, 50, 0.17)
send(4)
drive_to_point(gstx, gsty, 100, 50, 0.17) #отвезти куб

for (x, y) in yellow_points:
    setSpeed(35)
    drive_to_point(ctx, cty, 100, 30, 0.17) #едем в центр
    point = (x, y) #требуемая точка
    send(5) #открываем захват
    d = dist(ctx, cty, x, y) #расстояние до точки
    x = int(ctx + (x - ctx) * (1 + 30 / d)) #вычисляем координаты точки на 20 пикселей дальше от центра
    y = int(cty + (y - cty) * (1 + 30 / d)) # (чтобы точно забрать цилиндр)
    drive_to_point(x, y, 200, 35, 0.17) #доехать до точки
    #dovorot(x, y) #довернуть для точного захвата
    send(255) #остановиться
    send(4) #закрыть захват
    setSpeed(40)
    drive_to_point(ctx, cty, 200, 30, 0.17) #вернуться в центр

d = dist(ctx, cty, ystx, ysty)
x = int(ctx + (ystx - ctx) * (1 + 30 / d)) #вычисляем координаты точки на 20 пикселей дальше от центра
y = int(cty + (ysty - cty) * (1 + 30 / d)) # (чтобы точно забрать цилиндр)
drive_to_point(x, y, 200, 30, 0.17) #отвезти на базу
send(255) #остановиться
send(6)
setSpeed(35)
drive_to_point(yellow_points[0][0], yellow_points[0][1], 100, 35, 0.17)

# ret_point = get_returning_path(max_point[0], max_point[1]) #получить координаты точки для объезда
# drive_to_point(ret_point[1], ret_point[0]) #ехать туда
# drive_to_point(max_point[1], max_point[0]) #ехать в точку финиша
# send(4) #закрыть захват
# send(255) #остановаиться
sock.close()
cv2.waitKey(0) #ждать нажатия кнопки
cv2.destroyAllWindows() #закрыть все окна
