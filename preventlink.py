from time import sleep
import cv2 as cv
import numpy as np
import time

from threading import Timer

import httpexample

# ---------------------------------------------------------------------------------------

cara = 0
# gafas = False
casco = False
maquina_activa = False
tiempo_ultima_aparicion = 0.0

# Tiempos
TIEMPO_APAGADO_MAQUINA = 15.0  # Segundos
TIEMPO_ALARMA = 5.0  # Segundos
TIEMPO_DANGER = 10.0

# Imagenes
imagen = None

# ----------------------------------------------------------------------------------------------------------------

# from PIL import image
frame_count = 0  # used in mainloop  where we're extracting images., and then to drawPred( called by post process)
frame_count_out = 0  # used in post process loop, to get the no of specified class value.
# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    print(classes)

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./yolov3-obj.cfg"
modelWeights = "./yolov3-obj_2400.weights"


# ----------------------------------------------------------------------------------------------------------------

class Predictor:
    def __init__(self, modelConfiguration, modelWeights):
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom, frame):
        global frame_count
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % ("casco", label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # print(label)            #testing
        # print(labelSize)        #testing
        # print(baseLine)         #testing

        label_name, label_conf = label.split(':')  # spliting into class & confidance. will compare it with person.
        # print(f"Label Name = {label_name}")
        if label_name == 'helmet':
            # will try to print of label have people.. or can put a counter to find the no of people occurance.
            # will try if it satisfy the condition otherwise, we won't print the boxes or leave it.
            cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                         (left + round(1.5 * labelSize[0]), top + baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            frame_count += 1


        if frame_count > 0:
            return frame_count

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out = 0
        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []  # have to fins which class have hieghest confidence........=====>>><<<<=======
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    # print(classIds)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        count_person = 0  # for counting the classes in this loop.
        for i in indices:
            # i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # this function in  loop is calling drawPred so, try pushing one test counter in parameter , so it can
            # calculate it.
            frame_count_out = self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)
            # increase test counter till the loop end then print...

            # checking class, if it is a person or not

            my_class = 'helmet'  # ======================================== mycode .....
            unknown_class = classes[classId]

            if my_class == unknown_class:
                count_person += 1
        # if(frame_count_out > 0):
        print(f"Count person = {count_person}")

        if count_person == 0:
            cv.putText(frame, "NO CASCO!!", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        cv.imshow('Operador', frame)

        return count_person >= 1

        # cv.imwrite(frame_name, frame)
        # ======================================mycode.........

    def predict(self, frame):
        global frame_count

        frame_count = 0

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())

        # Remove the bounding boxes with low confidence
        return self.postprocess(frame, outs)


# ----------------- TIMERS -----------------------------
class RepeatedTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def start_an_alarm():
    global tiempo_ultima_aparicion, maquina_activa
    global casco, TIEMPO_ALARMA, TIEMPO_DANGER, TIEMPO_APAGADO_MAQUINA

    tiempo_actual = time.time()
    if tiempo_ultima_aparicion > 0:
        diferencia = tiempo_actual - tiempo_ultima_aparicion
        print(f"Han pasado {tiempo_actual} - {tiempo_ultima_aparicion} = {diferencia} segundos")
        if diferencia >= TIEMPO_APAGADO_MAQUINA:
            if maquina_activa:
                print("Apagar Máquina")
                httpexample.machine_stop()
                maquina_activa = False
        elif diferencia >= TIEMPO_DANGER:
            if maquina_activa:
                print("PELIGRO! PELIGRO!!")
                httpexample.machine_danger()
        elif diferencia >= TIEMPO_ALARMA:
            if maquina_activa:
                print("ALARMA ALARMA")
                httpexample.machine_warning()
    elif casco:
        if not maquina_activa:
            print("Encender Máquina")
            maquina_activa = True
            httpexample.machine_start_normal()
        else:
            print("Arranca normal ")
            httpexample.machine_start_normal(False)


# -------------------- Programa principal ------------------------------------------

if __name__ == '__main__':
    print("Creando el predictor")
    predictor = Predictor(modelConfiguration, modelWeights)

    # Process inputs
    winName = 'PreventLink'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    print("Arrancando la captura de video")
    cap = cv.VideoCapture(0)

    print("Creando el timer...")
    t = RepeatedTimer(2.0, start_an_alarm)
    t.start()

    # Encendemos la máquina
    print("Encendiendo la máquina. Conectando!")
    httpexample.machine_start()

    while cap.isOpened():
        # start_time = time.time()
        print("Leyendo la imagen...")
        ret, img = cap.read()
        imagen = img.copy()
        frame_count = 0
        if ret:
            casco = predictor.predict(imagen)
            if casco:
                tiempo_ultima_aparicion = 0.0
                print(f"Hay casco!")
            else:
                print("No hay casco!!")
                if tiempo_ultima_aparicion == 0:
                    tiempo_ultima_aparicion = time.time()

        print(f"Sleeping...")
        time.sleep(2.0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    t.cancel()

    # Al final apagamos la máquina
    httpexample.machine_stop()
