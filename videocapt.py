import cv2

if __name__ == '__main__':

    captura = cv2.VideoCapture(0)
    captura.set(3, 640)
    captura.set(4, 480)
    counter = 2

    while (captura.isOpened()):
        ret, imagen = captura.read()
        if ret:
            cv2.imshow('video', imagen)
            key = cv2.waitKeyEx(1)
            if key & 0xFF == ord('s'):
                break
            elif key & 0xFF == ord(' '):
                image_name = f"imagen{counter}.png"
                cv2.imwrite(image_name, imagen)
                print(f"{image_name} imagen guardada!")
                counter += 1
        else:
            break

    captura.release()
    cv2.destroyAllWindows()
