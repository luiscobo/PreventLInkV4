import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

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

class YoloPredictor:
    def __init__(self, onnx_model, data_yaml_file) -> None:
        with open(data_yaml_file, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']

        # Load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        global casco
        row, col, d = image.shape

        # get the YOLO prediction from the image

        # step 1: conver image into square image (array)
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # step 2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # detection or prediction from YOLO

        # Non maximum supression

        # step-1: filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # width and height of th eimage (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # Confidence of detection of an object
            if confidence > 0.4:
                class_score = row[5:].max()  # maximum probability of the object
                class_id = row[5:].argmax()  # get the index position at wihich max probability

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding from four values
                    # left, top, width and height
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

        if len(index) == 0:
            print("Exiting")
            return False, 0.0, None

        index = index.flatten()

        exist_helmet = False
        helmet_confidence = 0.0

        # Draw the bounding
        for i in index:
            # Extract the bounding box
            x, y, w, h = boxes_np[i]
            bb_conf = int(confidences_np[i] * 100)
            class_id = classes[i]
            class_name = self.labels[class_id]

            text = f'{class_name}: {bb_conf}%'
            if class_name == 'helmet':
                if not exist_helmet:
                    exist_helmet = True
                    helmet_confidence = bb_conf
                elif bb_conf > helmet_confidence:
                    helmet_confidence = bb_conf
            print(text)
            if casco:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 255, 255), -1)

            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        return exist_helmet, helmet_confidence, image


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
    yolo = YoloPredictor('helmet.onnx', 'data.yaml')

    print("Creando el timer...")
    t = RepeatedTimer(2.0, start_an_alarm)
    t.start()

    # Encendemos la máquina
    print("Encendiendo la máquina. Conectando!")
    httpexample.machine_start()

    print("Arrancando la captura de video")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start_time = time.time()
        print("Leyendo la imagen...")
        ret, img = cap.read()
        imagen = img.copy()

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord(' '):
            casco = False

        if k == ord('x'):
            casco = True

        if ret:
            resultado = yolo.predictions(imagen)
            cara = resultado[0]
            if casco:
                tiempo_ultima_aparicion = 0.0
                print(f"Hay casco! confidencia = {resultado[1]}%")
            else:
                print("No hay casco!!")
                if tiempo_ultima_aparicion == 0:
                    tiempo_ultima_aparicion = time.time()
            if not(resultado[2] is None):
                cv2.imshow('Casco', resultado[2])
        print(f"Sleeping...")
        time.sleep(2.0)

    cap.release()
    cv2.destroyAllWindows()
    t.cancel()

    # Al final apagamos la máquina
    httpexample.machine_stop()
