import cv2
import time 
import numpy as np
import time 
import keras

#tf.logging.set_verbosity(tf.logging.debug)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
#tf.logging.set_verbosity(tf.logging.WARN)



#TODO                                                                                      *   *
# Для каждого времени суток подобрать параметры для      l,imgRoi = cv2.threshold(imgRoi, 70, 255, cv2.THRESH_BINARY)
#
#
#
#
#

##########################################################################
''' Функции '''
def analysis(your_list):
    """Функция поиска одинаковых значений в списке

    Args:
        your_list ([list]): [Список со значениями]

    Returns:
        [dict]: [Возвращает словарь с цифрой и сколько раз повторяется]
    """

    your_dict = {}
    for i in your_list:
        if i in your_dict:
            your_dict[i] += 1
        else:
            your_dict[i] = 1
    return  your_dict

def num(img):
    """Функция более точного поиска номерного знака

    Returns:
        [img]: [Возвращает фотографию номерного знака]
    """
    
    #print("111")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    area_count = 0
    numberPlates = plateCascade_1.detectMultiScale(imgGray, 1.1, 4)
    for (x,y,w,h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"Number Plate",(x,y-5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
        area_count += 1
        #if area_count > 1:
        #    print("Error")
        #    break
        #else:
        imgRoi = img[y:y+h, x:x+w]
        cv2.imshow("Roi",imgRoi)
        #imgRoi = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2GRAY)
    return imgRoi

def ImgResize(img):
    """Функция изменения размера фотографии

    Args:
        img ([type]): [Фото]

    Returns:
        [type]: [Фото]
    """
    ResizedImg = cv2.resize(img,(480,240))
    return ResizedImg

def words(img,x,y,w,h):
    ## TODO ОПТИМИЗИРОВАТЬ И ПОДСТАВИТЬ ВМЕСТО Простого алгоритма в другой функции 
    letters = []
    out_size = 28

    letter_crop = img[y:y + h, x:x + w]

    # Resize letter canvas to square
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        # Enlarge image top-bottom
        # ------
        # ======
        # ------
        y_pos = size_max//2 - h//2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    elif w < h:
        # Enlarge image left-right
        # --||--
        x_pos = size_max//2 - w//2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop
    else:
        letter_square = letter_crop

    # Resize letter to 28x28 and add letter and its X-coordinate
    letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key = lambda x: x[0], reverse=False)

    for i in range (0, len(letters)):
        print(i)
        #letters[i][2] = ImgResize(letters[i][2])
        recognitionNumber(letters[i][2])
        cv2.imshow(str(i), letters[i][2])

def RecogSymbols(contours,imgRoi ):
        
    output = imgRoi.copy()
    output =  cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
    #outcolor =  outcolor.reshape(outcolor.shape[1],outcolor.shape[0],3)
    #img = img.reshape(1,out_size,out_size,1)
    letters = []
    xV = []

    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour) # Находим границы букв (x,y,w,h ) координаты и ширину с высотой
        xV.append((x, y, w, h))

    xV.sort(key = lambda x: x[0], reverse=False)
    
    #print((xV))
    # print((xV[1][0]))
    # print(len(xV))

    x_last_id = 0
    h_mean = 0
    count =0
    for i in range (0, len(xV)): # x y w h 
        x = xV[i][0]
        y = xV[i][1]
        w = xV[i][2]
        h = xV[i][3]

        ## Если Х больше 0,тк в 0 точке не может быть цифры 
        if xV[i][0] > 0 and xV[i][1] > 0:
            ## Если ширина будет меньше чем макисмально возможная ширина  
            if xV[i][2] < MaxW and xV[i][2] > MinW:
                ## Высота будет больше определеного,но меньше критических значений
                if xV[i][3] < MaxH and xV[i][3] > MinH:
                    #print("x_last_id = {}  i= {}".format(x_last_id, i))
                    #print("Разница между {} и {} = {}".format(xV[i][0], xV[x_last_id][0],xV[i][0] - xV [x_last_id][0]))
                    # if x_last_id == 0:
                    #     #print("ПОМЕНЯЛИ ЛАСТ")
                    #     x_last_id = 0
                    # elif x_last_id != i:
                    #     x_last_id = i
                    #print(x_last_id != i and x_last_id != 0)
                    # 0 != 5 and 0 !=0
                    #print("x_last_id----- = {}  i= {}".format(x_last_id, i))
                        #print("x_last_id = {}  i= {}".format(x_last_id, i))
                    #print( i," - ",xV[i][0] - xV [x_last_id][0] > MinD and x_last_id != i, "   ", x_last_id == 0)
                    if xV[i][0] - xV [x_last_id][0] > MinD and (x_last_id != i or x_last_id == 0) :
                        #print("Разница между {} и {} = {}".format(xV[i][0], xV[x_last_id][0],xV[i][0] - xV [x_last_id][0]))
                        if x_last_id != i:
                            x_last_id = i
                        count += 1
                        #print("Проверка  x_last_id = {}  i= {}".format(x_last_id, i))
                        #print("Порядковый номер элемента {} ,значение элемента {}".format(i,xV[i]))
                        letter_crop = imgRoi[y:y + h, x:x + w]
                        # Resize letter canvas to square
                        size_max = max(w, h)
                        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                        if w > h:
                            # Enlarge image top-bottom
                            # ------
                            # ======
                            # ------
                            y_pos = size_max//2 - h//2
                            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                        elif w < h:
                            # Enlarge image left-right
                            # --||--
                            x_pos = size_max//2 - w//2
                            letter_square[0:h, x_pos:x_pos + w] = letter_crop
                        else:
                            letter_square = letter_crop

                        # Resize letter to 28x28 and add letter and its X-coordinate
                        letters.append((x, y, cv2.resize(letter_square, (out_sizeW, out_sizeH), interpolation=cv2.INTER_AREA), w, h))
                        
    Answer = " "
    ## x w [] w h
    w_mean = 0
    h_mean = 0
    w_mean_1 = 0
    h_mean_1 = 0
    for i in range(0,len(letters)):
    #x_mean = np.mean(letters[0][4])
        
        w_mean += letters[i][3] /len(letters)
        h_mean += letters[i][4] / len(letters)
    #print(letters[i])
    #print(letters[0][4])
    #w_mean = np.mean(letters[:0])
    print(h_mean)
    corPos=0
    global nameimg 
    import os
    path = 'E:\\Python Projects\\Licence_plate_recognition\\train\\NewData\\'
    for i in range (0, len(letters)):
        last_time_1 = time.time()
        if h_mean + 25 > letters[i][4]:
            cv2.rectangle(output, (letters[i][0], letters[i][1]), (letters[i][0] + letters[i][3], letters[i][1] + letters[i][4]), color = (0, 0, 255), thickness= 2)
            #cv2.imwrite(path + str(nameimg)+'.jpg',letters[i][2])
            nameimg += 1
        #print(i)
        #letters[i][2] = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2GRAY)
        #letters[i][2]= cv2.adaptiveThreshold(imgRoi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY ,6,1) 
        #letters[i][2] = cv2.erode(imgRoi, np.ones((2, 2), np.uint8), iterations = 1)

            Answer = Answer + (str(recognitionNumber(letters[i][2])))
            corPos+=1
       #cv2.imshow(str(i), letters[i][2])
    print("\nНомер машины ",Answer,"\n")
    print("Loop take to recognise numberplate {} seconds ".format(time.time()-last_time_1))
    return output

def PatternCorrect (st,i):
    # Номер может иметь след вид  Н 666 УР 177 (одна буква 3 цифры две буквы и (две или три цифры) )
    if i == 0:
        if int(st) == 8:
            st ='B' 
        elif int(st) == 0:
            st ='O' 
        elif int(st) == 0: 
            st ='С'
    elif i > 0 and i < 4:
        if st == 'B':
            st == str(6)
    #elif  i > 3 and i < 7



    return st

def ImgThresh(img):
    #img = cv2.resize(img,(frameWidth,frameHeight))
    img = ImgResize(img)
    imgRoi = img
    cv2.imshow("Приход ",imgRoi)
    imgRoi = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray ",imgRoi)
    #cv2.imshow("2 - Bilateral Filter", imgRoi)
    #l,imgRoi = cv2.threshold(imgRoi, 100, 255, cv2.THRESH_BINARY)
    imgRoi = cv2.blur(imgRoi,(9, 9))
    #cv2.imshow("Blur ",imgRoi)
    imgRoi = cv2.bilateralFilter(imgRoi, 13, 19, 19)
    #cv2.imshow("Billin ",imgRoi)
    #cv2.imshow("3 - Canny Edges", imgRoi)
    imgRoi = cv2.adaptiveThreshold(imgRoi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY , 13,1) 
    #cv2.imshow("Threshold ",imgRoi)
    imgRoi = cv2.erode(imgRoi, np.ones((2, 2), np.uint8), iterations = 1)
    #cv2.imshow("Erode ",imgRoi)
    #cv2.imshow("3 - Erode Edges", imgRoi)
    #imgRoi = cv2.Canny(imgRoi, 150, 200)
    #contours, hierarchy = cv2.findContours(imgRoi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(imgRoi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours, imgRoi

#### PREPORCESSING FUNCTION
def preProcessing(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.equalizeHist(img)
    #cv2.imshow("TEST1",img)
    img = img/255
    #cv2.imshow("TEST",img)
    return img


def recognitionNumber(img):
    img = preProcessing(img)
    #### PREDICT
    img = img.reshape(1,out_sizeW,out_sizeH,1)
    classIndex = int(model.predict_classes(img))

    if classIndex== 10:
        classIndex='A'
    if classIndex == 11:
        classIndex ='B'
    if  classIndex ==12:
        classIndex ='С'
    if classIndex == 13:
        classIndex ='Е'
    if  classIndex ==14:
        classIndex ='Н'
    if classIndex == 15:
        classIndex ='К'
    if  classIndex ==16:
        classIndex ='М'
    if classIndex == 17:
        classIndex ='О'
    if  classIndex ==18:
        classIndex ='Р'
    if classIndex == 19:
        classIndex ='Т'
    if  classIndex ==20:
        classIndex = 'X'
    if  classIndex == 21:
        classIndex ='У'
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal =  np.amax(predictions)
    #print("ПРЕДСКАЗАНИЕ",classIndex,probVal)
    if probVal > threshold:
        #cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
        #            (50,50),cv2.FONT_HERSHEY_COMPLEX,
         #           1,(0,0,255),1)
        #print(classIndex, end=" ")
        print(classIndex,probVal)
    return classIndex
def number_rec(img):
    cv2.imshow("Test",img)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    img = cv2.resize(img,(640,360))
    #img = cv2.resize(img,(320,212))
    
    cv2.imshow("RESIZE",img)
    flag = False # Нашли ли номер
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
    area_count = 0
    for (x, y, w, h) in numberPlates:
        area = w * h
        print(area)

        if area > minArea:
            area_count +=1
            print("area_count- ",area_count)
            if area_count > 1:
               # print("Flag = ", flag)
                flag = True
                imgRoi = num(img)
                #break
            else:
                flag = False
                cv2.rectangle(img, (x, y),(x + w, y + h),(255, 0, 255), 2)
                cv2.putText(img, "Number Plate", (x, y - 5),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
                imgRoi = img[y:y + h, x:x + w]
                
            #print(img.shape)
            imgRoi = ImgResize(imgRoi)
            #l,imgRoi = cv2.threshold(imgRoi, 150, 255, cv2.THRESH_BINARY)
            #imgRoi = cv2.erode(imgRoi, np.ones((2, 2), np.uint8), iterations = 3)

            contours,imgRoi = ImgThresh(imgRoi)
            output = RecogSymbols(contours,imgRoi)


            cv2.imshow("Output", output)
            cv2.imshow("Roi",imgRoi)
    #text = pytesseract.image_to_string(output,lang='rus')
    #print (" Текст на фото ", text)
    #if (area_count == 1):
        #("Roi",imgRoi)



##########################################################################


##############################
''' Области и подгружаемые файлы   '''
minArea = 50 #№ Минимальная область для знака
minAreaSign = 1000 ## Минимальная площадь найденных букв,цифр на знаке ## TODO Найти более оптимальные значения ,возможно нужна функция,которая после тренировки будет сама подставлять нужные значения от освещенности,суток и тд 
frameWidth = 640  ## 480
frameHeight = 480 ## 240 
out_sizeW = 32
out_sizeH = 32


frameWidth_1 = 720
frameHeight_1= 576

maxHeight = 50 ##Сделать адаптивными
maxWeght = 10 ##Сделать адаптивными


###############################################
######### НАСТРОЙКИ НЕЙРОННОЙ СЕТИ ############
threshold = 0.6 # MINIMUM PROBABILITY TO CLASSIFY
path_learn_model ='E:\\Python Projects\\Licence_plate_recognition\\train\\TrainModel\\fcn_best_res.h5'
model = keras.models.load_model(path_learn_model)

###############################################

#pytesseract.pytesseract.tesseract_cmd = r"E:\\Tesseract\\Tesseract-OCR\\tesseract.exe" ## Не работает, ГОВНО
plateCascade = cv2.CascadeClassifier(f"E:/Python Projects/Licence_plate_recognition/Resourses/haarcascades/haarcascade_russian_plate_number.xml")
plateCascade_1 = cv2.CascadeClassifier(f"E:/Python Projects/Licence_plate_recognition/Resourses/haarcascades/haarcascade_licence_plate_rus_16stages.xml")

##############################

### НУЖНО ПОДВЕРГНУТЬ КОРРЕКТИРОВКЕ 
#####  НАСТРОЙКИ ДЛЯ ОПРЕДЕЛЕНИЯ РАЗМЕРА ЗНАКОВ  ДЛЯ ИХ ОПРЕДЕЛЕНИЯ #############
MinW = 20 # Минимальная ширина
MaxW = 90 # Мфаксимальная ширина
MinH = 59 #Минимальная  высота
MaxH = 1000 #Максимальная  высота
MinD = 13 # Минимальная дальность между буквами
MaxY = 15 # Максимальная дальность между буквами

################################




##############################
''' Цвета  '''
color = (255, 0, 255) #RGB
color_cube = (255, 10, 25)

##############################


#cap = cv2.VideoCapture(1) ## Захватываем видео с вебкамеры

#cap.set(3,frameWidth)
#cap.set(4,frameHeight)
#cap.set(10,150)
#while True:
    #success,img = cap.read()



a=0
#numberPlate()
import os
path = 'E:\\Python Projects\\Licence_plate_recognition\\train\\TrainForNumberCoor\\'

path_img_bmp ='E:\\Python Projects\\Licence_plate_recognition\\train\\PicCar\\1\\'
path_img_jpg ='E:\\Python Projects\\Licence_plate_recognition\\train\\PicCar\\'

nameimg = 0
# cap = cv2.VideoCapture(f"E:/Python Projects/Licence_plate_recognition/Resourses/2.mp4")
# cap.set(3,frameWidth)
# cap.set(4,frameHeight)
# cap.set(10,150)
img = cv2.imread(f"E:/Python Projects/Licence_plate_recognition/Resourses/0.jpg")## Открытие папки с фото номеров 
while (a<1):
    last_time = time.time()
    #nameimg = 0
    # from PIL import Image
    
    # for item in (os.listdir(path_img_bmp)):

    #     img = Image.open(path_img_bmp +'/'+item)
    #     img.save(path_img_jpg + '/' + item +'.jpg')

    # myPicList = os.listdir(path_img_jpg) 
    # for y in myPicList:
    #     img = cv2.imread(path_img_jpg +'/'+ y )
    #     print( img.shape)
    #     #break
    #     if img.shape[0] < 340 and img.shape[1] < 300:
    #         cv2.resize(img,(720,480))
    #     if img.shape[0] > 340 and img.shape[1] > 300:

    #         print(path_img_jpg + y )
    #         cv2.imshow("1",img)
        #cv2.waitKey(100)
    #img = cv2.imread(f"E:/Python Projects/Licence_plate_recognition/Resourses/16.jpg")## Открытие папки с фото номеров 

    #l,img = cap.read()
    number_rec(img)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Result ",img)

    print("Loop take {} seconds ".format(time.time()-last_time))
    cv2.waitKey(0)
    a += 2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




 