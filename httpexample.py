# Programa de ejemplo del uso del URLLib
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import urllib.parse
import time

# constantes
GPIO_ON = 1
GPIO_OFF = 0
GPIO_IP_ADDRESS = "192.168.1.50"
RED_PORT = 7
GREEN_PORT = 9
YELLOW_PORT = 8
RELAY_PORT_1 = 3
RELAY_PORT_2 = 5
RELAY_PORT_3 = 6
RELAYS = [RELAY_PORT_1, RELAY_PORT_2, RELAY_PORT_3]


# Para hacerle requerimientos al GPIO
def gpio_request(ip_address, port, pin, command):
    try:
        url = f'http://{ip_address}:{port}/'
        params = [
            ("op", "output"),
            ("data", f"port.{pin}"),
            ("data", f"state.{command}")
        ]
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"
        data = query_string.encode("ascii")

        with urlopen(url) as response:
            print(response.status)
            return response.read()

    except HTTPError as error:
        print(error.status, error.reason)
    except URLError as error:
        print(error.reason)
    except Exception as ex:
        print(ex)


# Para encender el un pin dado
def pin_on(pin, sleep_time=0.0):
    result = gpio_request(GPIO_IP_ADDRESS, 80, pin, command=GPIO_ON)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return result


# Para apagar un pin dado
def pin_off(pin, sleep_time=0.0):
    result = gpio_request(GPIO_IP_ADDRESS, 80, pin, command=GPIO_OFF)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return result


# Encender el rojo
def red_on(sleep_time=0.0):
    print(pin_on(RED_PORT, sleep_time))


# Apagar el rojo
def red_off(sleep_time=0.0):
    print(pin_off(RED_PORT, sleep_time))


# Encender el verde
def green_on(sleep_time=0.0):
    print(pin_on(GREEN_PORT, sleep_time))


# Apagar el led verde
def green_off(sleep_time=0.0):
    print(pin_off(GREEN_PORT, sleep_time))


# Encender el led amarillo
def yellow_on(sleep_time=0.0):
    print(pin_on(YELLOW_PORT, sleep_time))


# Apagar el led amarillo
def yellow_off(sleep_time=0.0):
    print(pin_off(YELLOW_PORT, sleep_time))


# Encender el relay dado
def relay_on(relay_num, sleep_time=0.0):
    print(pin_on(relay_num, sleep_time))


# Apagar el relay dado
def relay_off(relay_num, sleep_time=0.0):
    print(pin_off(relay_num, sleep_time))


# Enciende todos los relays
def relays_on(sleep_time=0.5):
    for relay in RELAYS:
        relay_on(relay, sleep_time)


# Apagar todos los relays
def relays_off(sleep_time=0.5):
    for relay in RELAYS:
        relay_off(relay, sleep_time)


# Encender la máquina
def machine_start():
    green_on(0.25)
    red_on(0.25)
    yellow_on(0.25)

    relays_on(0.25)

    green_off(0.25)
    red_off(0.25)
    yellow_off(0.25)

    relays_off(0.25)


# Encender la máquina en operacion normal
def machine_start_normal(start_relays=True):
    green_on(0.25)
    red_off()
    yellow_off(0.25)

    if start_relays:
        relays_on(0)


# Colocar en warning
def machine_warning():
    green_off(0)
    red_off(0)
    yellow_on(0)


# Colocar en modo danger
def machine_danger():
    green_off(0)
    red_on(0)
    yellow_on(0)


# Apagar la máquina
def machine_stop():
    green_off(0.25)
    red_off(0.25)
    yellow_off(0.25)

    relays_off(0.25)


# Programa principal
if __name__ == '__main__':
    machine_start_normal()
    time.sleep(5.0)
    machine_stop()
