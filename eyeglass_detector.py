# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np

import time
from threading import Timer

import httpexample

cara = 0
gafas = False
maquina_activa = False
tiempo_ultima_aparicion = 0.0

# Tiempos
TIEMPO_APAGADO_MAQUINA = 15.0   # Segundos
TIEMPO_ALARMA = 5.0 # Segundos
TIEMPO_DANGER = 10.0


# ----------------- TIMERS -----------------------------
class RepeatedTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def start_alarm():
    global tiempo_ultima_aparicion, maquina_activa
    global gafas, TIEMPO_ALARMA, TIEMPO_DANGER, TIEMPO_APAGADO_MAQUINA

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
    elif gafas:
        if not maquina_activa:
            print("Encender Máquina")
            maquina_activa = True
            httpexample.machine_start_normal()
        else:
            httpexample.machine_start_normal(False)


# ------------------------------------------------------


def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    x_left = (EYE_LEFT_OUTTER[0] + EYE_LEFT_INNER[0]) / 2
    x_right = (EYE_RIGHT_OUTTER[0] + EYE_RIGHT_INNER[0]) / 2
    LEFT_EYE_CENTER = np.array([np.int32(x_left), np.int32(x_left * k + b)])
    RIGHT_EYE_CENTER = np.array([np.int32(x_right), np.int32(x_right * k + b)])

    pts = np.vstack((LEFT_EYE_CENTER, RIGHT_EYE_CENTER))
    cv2.polylines(img, [pts], False, (255, 0, 0), 1)
    cv2.circle(img, (LEFT_EYE_CENTER[0], LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    cv2.circle(img, (RIGHT_EYE_CENTER[0], RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)

    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER


def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5

    eyescenter = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)  # 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx * dx + dy * dy)  # 瞳距
    scale = desired_dist / dist  # 缩放比例
    angle = np.degrees(np.arctan2(dy, dx))  # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter, angle, scale)  # 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img, M, (desired_w, desired_h))

    return aligned_face


def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11, 11), 0)

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    cv2.imshow('sobel_y', sobel_y)

    edgeness = sobel_y

    retVal, thresh = cv2.threshold(edgeness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    d = len(thresh) * 0.5
    x = np.int32(d * 6 / 7)
    y = np.int32(d * 3 / 4)
    w = np.int32(d * 2 / 7)
    h = np.int32(d * 2 / 4)

    x_2_1 = np.int32(d * 1 / 4)
    x_2_2 = np.int32(d * 5 / 4)
    w_2 = np.int32(d * 1 / 2)
    y_2 = np.int32(d * 8 / 7)
    h_2 = np.int32(d * 1 / 2)

    roi_1 = thresh[y:y + h, x:x + w]  # 提取ROI
    roi_2_1 = thresh[y_2:y_2 + h_2, x_2_1:x_2_1 + w_2]
    roi_2_2 = thresh[y_2:y_2 + h_2, x_2_2:x_2_2 + w_2]
    roi_2 = np.hstack([roi_2_1, roi_2_2])

    measure_1 = sum(sum(roi_1 / 255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])  # 计算评价值
    measure_2 = sum(sum(roi_2 / 255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])  # 计算评价值
    measure = measure_1 * 0.3 + measure_2 * 0.7

    # cv2.imshow('roi_1', roi_1)
    # cv2.imshow('roi_2', roi_2)
    # print(measure)

    if measure > 0.15:
        judge = True
    else:
        judge = False
    # print(judge)
    return judge


if __name__ == '__main__':

    t = RepeatedTimer(1, start_alarm)
    t.start()

    # Encendemos la máquina
    httpexample.machine_start()

    predictor_path = "./data/shape_predictor_5_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        cara = len(rects)
        # if cara == 0:
        #     print("NO HAY CARA!")
        #     if tiempo_ultima_aparicion == 0:
        #         tiempo_ultima_aparicion = time.time()
        # else:
        #     tiempo_ultima_aparicion = 0

        for i, rect in enumerate(rects):
            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face

            cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
            cv2.putText(img, "Cara #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            landmarks = predictor(gray, rect)
            landmarks = landmarks_to_np(landmarks)
            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

            LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)

            aligned_face = get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
            cv2.imshow("cara_alineada #{}".format(i + 1), aligned_face)

            judge = judge_eyeglass(aligned_face)
            gafas = judge
            if judge:
                tiempo_ultima_aparicion = 0.0
                print("Hay gafas!")
                cv2.putText(img, "con gafas", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2,
                            cv2.LINE_AA)
            else:
                print("No hay gafas!!")
                if tiempo_ultima_aparicion == 0:
                    tiempo_ultima_aparicion = time.time()
                cv2.putText(img, "sin gafas", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2,
                            cv2.LINE_AA)

        cv2.imshow("Resultado", img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    t.cancel()

    # Al final apagamos la máquina
    httpexample.machine_stop()
