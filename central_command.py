# Archivo con la interface del comando cental de PreventLink
import PySimpleGUI as sg
import csv


# Escoger el tema de la aplicacion
sg.theme('SystemDefault1')
EPP_FILENAME = "./data/epp.csv"
GPIO_FILENAME = "./data/gpio.csv"


def load_epp_table(filename):
    data = [["Código", "Nombre", "Descripción"]]
    try:
        arch = open(filename, "r")
        reader = csv.reader(arch)
        data.extend(list(reader))
        arch.close()
    except:
        print("Archivo no existe!")

    return data


def make_epp_tab_layout(data):
    """
    Permite crear el contenido del tab de la EPP
    :return:
    """
    layout = [
        [sg.Text("Administración de Elementos de Protección Personal (EPP)", font='Helvetica 16')],
        [sg.Table(values=data[1:][:], headings=data[0], max_col_width=25,
                  auto_size_columns=True,
                  # cols_justification=('left','center','right','c', 'l', 'bad'),       # Added on GitHub only as of June 2022
                  display_row_numbers=True,
                  justification='left',
                  num_rows=20,
                  alternating_row_color='lightblue',
                  key='-EPP-DATA-TABLE-',
                  selected_row_colors='red on yellow',
                  enable_events=True,
                  expand_x=False,
                  expand_y=True,
                  vertical_scroll_only=False,
                  enable_click_events=True,  # Comment out to not enable header and other clicks
                  tooltip='EPP')],
        [sg.Button('Crear Nuevo EPP', key='-NEW-EPP-'), sg.Button('Editar EPP', key='-EDIT-EPP-'),
         sg.Button('Eliminar EPP', key='-DELETE-EPP-')]
    ]
    return layout


# Crear un nuevo EPP
def create_new_epp_window():
    result = (False, [])
    win_layout = [
        [sg.Text('Crear Elemento de Protección Personal', font='Arial 16')],
        [sg.Text('Código', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-EPP-CODIGO-')],
        [sg.Text('Nombre', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-EPP-NOMBRE-')],
        [sg.Text('Descripción', size=(10, 1)), sg.Multiline(size=(40, 10), key='-NEW-EPP-DESCRIPCION-')],
        [sg.Button('Guardar'), sg.Button('Cancelar')]
    ]

    new_epp_window = sg.Window('Crear Nuevo EPP', win_layout, modal=True, font='Helvetica 14', )
    while True:
        event, values = new_epp_window.read()
        if event == sg.WIN_CLOSED or event == "Cancelar":
            break
        elif event == "Guardar":
            codigo = values['-NEW-EPP-CODIGO-']
            nombre = values['-NEW-EPP-NOMBRE-']
            descripcion = values['-NEW-EPP-DESCRIPCION-']
            print(values)
            result = (True, [codigo, nombre, descripcion])
            break

    new_epp_window.close()
    return result




# -------- GPIO ----------
def make_gpio_tab_layout(data):
    layout = [
        [sg.Text("Administración de Dispositivos GPIO", font='Helvetica 16')],
        [sg.Table(values=data[1:][:], headings=data[0], max_col_width=25,
                  auto_size_columns=True,
                  # cols_justification=('left','center','right','c', 'l', 'bad'),       # Added on GitHub only as of June 2022
                  display_row_numbers=True,
                  justification='left',
                  num_rows=20,
                  alternating_row_color='lightblue',
                  key='-GPIO-DATA-TABLE-',
                  selected_row_colors='red on yellow',
                  enable_events=True,
                  expand_x=False,
                  expand_y=True,
                  vertical_scroll_only=False,
                  enable_click_events=True,  # Comment out to not enable header and other clicks
                  tooltip='EPP')],
        [sg.Button('Crear Nuevo GPIO', key='-NEW-GPIO-'), sg.Button('Editar GPIO', key='-EDIT-GPIO-'),
         sg.Button('Eliminar GPIO', key='-DELETE-GPIO-')]
    ]
    return layout


def load_gpio_table(filename):
    data = [["Dirección IP", "Puerto", "Verde", "Ambar", "Rojo", "Relé"]]
    try:
        arch = open(filename, "r")
        reader = csv.reader(arch)
        data.extend(list(reader))
        arch.close()
    except:
        print("Archivo no existe!")

    return data


def create_new_gpio_window():
    result = (False, [])
    win_layout = [
        [sg.Text('Crear GPIO', font='Arial 16')],
        [sg.Text('IP', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-IP-')],
        [sg.Text('Puerto', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-PUERTO-')],
        [sg.Text('Led Verde', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-VERDE-')],
        [sg.Text('Led Ambar', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-AMBAR-')],
        [sg.Text('Led Rojo', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-ROJO-')],
        [sg.Text('Puerto Relé', size=(10, 1)), sg.Input(expand_x=True, key='-NEW-GPIO-RELE-')],
        [sg.Button('Guardar', key='-GUARDAR-GPIO-'), sg.Button('Cancelar', key='-CANCELAR-GPIO-')]
    ]

    new_epp_window = sg.Window('Crear GPIO', win_layout, modal=True, font='Helvetica 14', )
    while True:
        event, values = new_epp_window.read()
        if event == sg.WIN_CLOSED or event == '-CANCELAR-GPIO-':
            break
        elif event == '-GUARDAR-GPIO-':
            print(values)
            result = (True, values.copy())
            break

    new_epp_window.close()
    return result


def save_data(filename, data):
    with open(filename, 'w') as arch:
        writer = csv.writer(arch)
        writer.writerows(data[1:])

# ---- Programa Principal -------


# Tab de los EPP
epp_data = load_epp_table(EPP_FILENAME)
epp_tab_layout = make_epp_tab_layout(epp_data)

# Tab de los GPIO
gpio_data = load_gpio_table(GPIO_FILENAME)
gpio_tab_layout = make_gpio_tab_layout(gpio_data)

# The TabgGroup layout - it must contain only Tabs
main_tab_group_layout = [
    [sg.Tab('EPPs', epp_tab_layout, key='-TAB1-')],
    [sg.Tab('GPIOs', gpio_tab_layout, key='-TAB2-', expand_x=True)]
]

# The window layout - defines the entire window
layout = [
    [sg.Text('PreventLink', font='Helvetica 18')],
    [sg.TabGroup(main_tab_group_layout, enable_events=False)],
    [sg.Button('Guardar', key='-SAVE-ALL-'), sg.Button('Salir')]
]

# ------ Create Window ------
window = sg.Window('PreventLink', layout,
                   # ttk_theme='clam',
                   font='Helvetica 14',
                   resizable=True
                   )

# ------ Event Loop ------
selected_row = -1
selected_gpio = -1
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Salir':
        break

    if event == '-SAVE-ALL-':
        save_data(EPP_FILENAME, epp_data)
        save_data(GPIO_FILENAME, gpio_data)
        sg.Popup("Datos almacenados correctamente!", font='Helvetica 14')

    if event == '-NEW-EPP-':
        (guardar, datos) = create_new_epp_window()
        if guardar:
            epp_data.append(datos)
            window['-EPP-DATA-TABLE-'].update(values=epp_data[1:][:])
            sg.Popup("Nuevo EPP almacenado")
    if event == '-EDIT-EPP-':
        sg.PopupError('Conexión Imposible!')

    if event == '-DELETE-EPP-':
        if selected_row == -1:
            sg.PopupError("No hay EPP seleccionado")
        else:
            epp_data.pop(selected_row)
            window['-EPP-DATA-TABLE-'].update(values=epp_data[1:][:])
            sg.Popup("EPP Eliminado exitosamente")

    if event == '-EPP-DATA-TABLE-':
        seleccionados = values['-EPP-DATA-TABLE-']
        if len(seleccionados) == 0:
            selected_row = -1
        else:
            selected_row = seleccionados[0]
        print(f"EPP Seleccionado {selected_row}")

    if event == '-GPIO-DATA-TABLE-':
        seleccionados = values['-GPIO-DATA-TABLE-']
        if len(seleccionados) == 0:
            selected_gpio = -1
        else:
            selected_gpio = seleccionados[0]
        print(f"GPIO Seleccionado {selected_gpio}")

    if event == '-NEW-GPIO-':
        (guardar, valores) = create_new_gpio_window()
        if guardar:
            datos = [valores['-NEW-GPIO-IP-'], valores['-NEW-GPIO-PUERTO-'], valores['-NEW-GPIO-VERDE-'],
                     valores['-NEW-GPIO-AMBAR-'], valores['-NEW-GPIO-ROJO-'], valores['-NEW-GPIO-RELE-']]
            gpio_data.append(datos)
            window['-GPIO-DATA-TABLE-'].update(values=gpio_data[1:][:])
            sg.Popup("Nuevo GPIO guardado")

    if event == '-DELETE-GPIO-':
        if selected_gpio == -1:
            sg.PopupError("No hay GPIO seleccionado")
        else:
            gpio_data.pop(selected_gpio)
            window['-GPIO-DATA-TABLE-'].update(values=gpio_data[1:][:])
            sg.Popup("GPIO Eliminado exitosamente", font='Helvetica 14')

window.close()
