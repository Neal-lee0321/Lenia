import sys
import random
import json
import os
import time
import single_channel as SC
import numpy as np
from fractions import Fraction
from single_channel import Automaton as Automaton
from single_channel import Board as Board
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QAction, QActionGroup, QToolBar, QLineEdit, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen, QIcon


L = 4  # The width of a single cell by pixels 
D = 80  # The width of the painter by cells

class LifeGame(QMainWindow, Automaton, Board):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Life Game")
        self.setGeometry(100, 100, 1200, 1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.item_index = [[[0, 0]] * SC.SIZEX for i in range(SC.SIZEY)]
        # self.setViewportUpdateMode(QGraphicsView.NoViewportUpdate) 

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.timeout.connect(self.view.update)
        self.timer.start(100)

        self.height = SC.SIZEX
        self.width = SC.SIZEY
        self.cells = [[False] * self.height for i in range(self.width)]

        self.setMouseTracking(True)
        self.draw_state = False

        self.init_lenia()
        self.init_ui()


    def init_ui(self):
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.view)

        self.central_widget.setLayout(vbox)

        self.get_color()
        self.random_start()

        self.start_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.toggle_game)
        hbox.addWidget(self.start_button)

        self.random_button = QPushButton("Random")
        self.random_button.clicked.connect(self.random_start)
        hbox.addWidget(self.random_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_scene)
        hbox.addWidget(self.clear_button)

        vbox.addLayout(hbox)


        menubar = self.menuBar()
        pattern_menu = menubar.addMenu("Patterns")
        self.create_pattern_menu(pattern_menu)

        scale_menu = menubar.addMenu("Scale")
        self.create_scale_menu(scale_menu)

        kernel_core_menu = menubar.addMenu("Kernel_Core")
        self.create_kernel_core_menu(kernel_core_menu)

        field_function_menu = menubar.addMenu("Field_Function")
        self.create_field_function_menu(field_function_menu)

        self.create_toolbar()


    def tool_make_action(self, param):
        def action():
            text = self.sender().text()
            try:
                if param in ["s", "m"]:
                    value = float(text)
                    if 0 <= value <= 1:
                        self.lenia.world.params[param] = value                       
                elif param in ["R", "T"]:
                    value = int(text)
                    self.lenia.world.params[param] = value
                elif param == "b":
                    self.lenia.world.params[param] = [Fraction(val) for val in text.split(",")]
                self.lenia.calc_kernel()
            except ValueError:
                pass
        return action
    
    def update_param_text(self):
        self.R.setText(str(self.lenia.world.params["R"]))
        self.T.setText(str(self.lenia.world.params["T"]))
        self.s.setText(str(self.lenia.world.params["s"]))
        self.m.setText(str(self.lenia.world.params["m"]))

        b_lst = ""
        for fact in self.lenia.world.params["b"]:
            if len(b_lst) > 0:
                b_lst += ","
            if isinstance(fact, Fraction):
                b_lst += str(fact.numerator) + "/" + str(fact.denominator)
            else:
                b_lst += str(fact)
        self.b.setText(b_lst)
    
    def create_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)
        label_1 = QLabel("Parameters  ")
        label_R = QLabel("R:")
        label_T = QLabel("T:")
        label_b = QLabel("b:")
        label_s = QLabel("s:")
        label_m = QLabel("m:")
        
        self.R = QLineEdit()
        self.T = QLineEdit()
        self.b = QLineEdit()
        self.s = QLineEdit()
        self.m = QLineEdit()
        self.R.setFixedWidth(100)
        self.T.setFixedWidth(100)
        self.b.setFixedWidth(140)
        self.s.setFixedWidth(120)
        self.m.setFixedWidth(120)
        self.update_param_text()

        self.R.returnPressed.connect(self.tool_make_action("R"))
        self.T.returnPressed.connect(self.tool_make_action("T"))
        self.b.returnPressed.connect(self.tool_make_action("b"))
        self.s.returnPressed.connect(self.tool_make_action("s"))
        self.m.returnPressed.connect(self.tool_make_action("m"))

        self.toolbar.addWidget(label_1)
        self.toolbar.addWidget(label_R)
        self.toolbar.addWidget(self.R)
        self.toolbar.addWidget(label_T)
        self.toolbar.addWidget(self.T)
        self.toolbar.addWidget(label_b)
        self.toolbar.addWidget(self.b)
        self.toolbar.addWidget(label_s)
        self.toolbar.addWidget(self.s)
        self.toolbar.addWidget(label_m)
        self.toolbar.addWidget(self.m)

    

    def init_lenia(self):
        self.lenia = Automaton()
        self.lenia.world.cells = np.zeros((SC.SIZEX, SC.SIZEY))
        self.lenia.calc_kernel()



    def create_pattern_menu(self, menu):
        ustc_icon = QAction("USTC", self)
        ustc_icon.triggered.connect(self.paint_ustc)
        menu.addAction(ustc_icon)

        HERE = os.path.dirname(__file__)
        FILE_PATH = os.path.join(HERE, "animals.json")
        with open(FILE_PATH, encoding = 'utf-8') as file:
            self.animal_data = json.load(file)
        print(len(self.animal_data))

        menulist = [[0, menu]]
        location = 0
        for menu_data in self.animal_data:

            code = menu_data["code"]
            name = menu_data["name"] + " " + menu_data["cname"]
            if code[0] == '>':
                level = int(code[1])
                while len(menulist) > 0 and menulist[-1][0] >= level:
                    menulist.pop()
                cur_menu = menulist[-1][1].addMenu(name)
                menulist.append([level, cur_menu])
            else:
                animal = QAction(name, self)
                menulist[-1][1].addAction(animal)
                # animal.setStatusTip("Code: " + code)
                def make_action(location):
                    def on_action_triggered():
                        self.paint_animal(location)
                    return on_action_triggered
                animal.triggered.connect(make_action(location))
            location += 1


    def create_scale_menu(self, menu):
        scale_group = QActionGroup(self)
        scale_group.setExclusive(True)

        scale = [1, 2, 3, 4, 8]
        acts = []
        for i in scale:
            acts.append(QAction(str(i) + "x", self, checkable = True))
            scale_group.addAction(acts[-1])
            menu.addAction(acts[-1])
        acts[1].setChecked(True)

        def on_triggered(action):
            num = int(action.text()[0])
            if SC.SCALE != num:
                self.lenia.world.params["R"] //= SC.SCALE
                self.lenia.world.params["R"] *= num
                SC.SCALE = num
                self.lenia.calc_kernel()
                self.update_param_text()
        scale_group.triggered.connect(on_triggered)


    def create_kernel_core_menu(self, menu):
        kernel_group = QActionGroup(self)
        kernel_group.setExclusive(True)

        kernels = {"polynomial" : 1, "exponential / gaussian bump" : 2, "step" : 3, "staircase (life game)" : 4}
        self.kernel_button = []
        for kernel in kernels.keys():
            act = QAction(kernel, self, checkable = True)
            self.kernel_button.append(act)
            kernel_group.addAction(act)
            menu.addAction(act)
        self.kernel_button[self.lenia.world.params["kn"]-1].setChecked(True)
        
        def on_triggered(action):
            Id = kernels[action.text()]
            self.lenia.world.params["kn"] = Id
            self.lenia.calc_kernel()
        kernel_group.triggered.connect(on_triggered)


    def create_field_function_menu(self, menu):
        field_group = QActionGroup(self)
        field_group.setExclusive(True)

        fields = {"polynomial" : 1, "exponential / gaussian" : 2, "step" : 3}
        self.field_button = []
        for field in fields.keys():
            act = QAction(field, self, checkable = True)
            self.field_button.append(act)
            field_group.addAction(act) 
            menu.addAction(act)
        self.field_button[self.lenia.world.params["gn"]-1].setChecked(True)

        def on_triggered(action):
            Id = fields[action.text()]
            self.lenia.world.params["gn"] = Id
            self.lenia.calc_kernel()
        field_group.triggered.connect(on_triggered)


    def get_color(self):
        # purple = [69, 13, 84]
        # blue = [45, 112, 142]
        # green = [86, 198, 103]
        # yellow = [253, 231, 37]
        red = [144, 12, 0]
        orange = [255, 132, 29]
        yellow = [243, 201, 43]
        green = [150, 250, 80]
        blue = [37, 199, 214]
        purple = [72, 57, 163]
        black = [35, 23, 27]
        self.color = []

        for val in range(256):
            state = val / 255.0
            state = 1-state
            clr = [0, 0, 0]
            for i in range(3):
                if(state < 1/4):
                    clr[i] = 4 * ((1/4 - state) * red[i] + state * orange[i])
                elif(state < 3/8):
                    clr[i] = 8 * ((3/8 - state) * orange[i] + (state - 1/4) * yellow[i])
                elif(state < 1/2):
                    clr[i] = 8 * ((1/2 - state) * yellow[i] + (state - 3/8) * green[i])
                elif(state < 3/4):
                    clr[i] = 4 * ((3/4 - state) * green[i] + (state - 1/2) * blue[i])
                elif(state < 11/12):
                    clr[i] = 6 * ((11/12 - state) * blue[i] + (state - 3/4) * purple[i])
                else:
                    clr[i] = 12 * ((1 - state) * purple[i] + (state - 11/12) * black[i])
                clr[i] = int(clr[i])
            self.color.append(clr)


    def add_cell(self, x, y, state):
        rect = QGraphicsRectItem(0, 0, L, L)
        rect.setPen(QPen(Qt.NoPen))
        rect.setPos(L * x, L * y)
        # rect.setBrush(Qt.black if state else Qt.white)

        clr = self.color[int(state * 255)]
        rect.setBrush(QColor(clr[0], clr[1], clr[2]))
        self.scene.addItem(rect)
        self.item_index[x][y] = [rect, clr]

    def clear_scene(self):
        clr = self.color[0]
        # clr = [255, 255, 255]
        for item in self.scene.items():
            item.setBrush(QColor(clr[0], clr[1], clr[2]))
        self.lenia.world.cells.fill(0)
        self.scene.update()

    def random_start(self):
        for item in self.scene.items():
            self.scene.removeItem(item)
        for x in range(self.width):
            for y in range(self.height):
                self.lenia.world.cells[x][y] = random.uniform(0, 1)
                self.add_cell(x, y, self.lenia.world.cells[x][y])
        self.scene.update()

    def toggle_game(self):
        if(not self.timer.isActive()):
            self.timer.start()
            self.start_button.setText("Stop")
        else:
            self.timer.stop()
            self.start_button.setText("Continue")
        # self.timer.start() if not self.timer.isActive() else self.timer.stop()

    def paint_ustc(self):
        0
    def paint_animal(self, location):
        print("location: ", location)
        # string = self.animal_data[location]["cells"]
        # print(string)
        # self.current_pattern = decode(string)
        # print(self.current_pattern)
        self.current_data = self.animal_data[location]
        self.draw_state = True

        board = Board.from_data(self.current_data)
        board.params['R'] *= SC.SCALE
        self.lenia = Automaton(board)
        self.lenia.world.cells = np.zeros((SC.SIZEX, SC.SIZEY))
        self.calc_kernel()
        self.clear_scene()
        self.update_param_text()
        self.kernel_button[self.lenia.world.params["kn"]-1].setChecked(True)
        self.field_button[self.lenia.world.params["gn"]-1].setChecked(True)


    def mousePressEvent(self, event):
        pos = self.view.mapToScene(event.pos())
        x = int((pos.x() - 15) // L)
        y = int((pos.y() - 44) // L)
        print("[", pos.x(), ",", pos.y(), "]")
        print("pos_x: ", x, " pos_y: ", y)

        # if(0 <= x < self.width and 0 <= y < self.height):
        #     side = event.button() == Qt.LeftButton
        #     n = m = 1
        #     if(side and self.draw_state == True):
        #         n = len(self.current_pattern)
        #         m = len(self.current_pattern[0])
        #         x = (x - (n-1)/2 + self.width) % self.width
        #         y = (y - (m-1)/2 + self.height) % self.height

        #         # self.lenia.world.add(SC.Board.from_data({"cells":self.current_pattern / 256}),[x,y])
        #         print("n:", n), print("m:", m), print(self.current_pattern)
        #     else:
        #         self.current_pattern = [[side]]
            
        #     for item in self.scene.items():
        #         if isinstance(item, QGraphicsRectItem):
        #             tx = int(item.x() / 10)
        #             ty = int(item.y() / 10)
        #             px = int((tx - x + self.width) % self.width)
        #             py = int((ty - y + self.height) % self.height)
        #             if(px >= 0 and px < n and py >= 0 and py < m):
        #                 self.cells[tx][ty] = self.current_pattern[px][py]
        #                 # item.setBrush(Qt.black if self.cells[tx][ty] else Qt.white)
        #                 clr = 255*(1-self.cells[tx][ty])
        #                 item.setBrush(QColor(clr, clr, clr))
        # self.draw_state = False

        if(0 <= x < self.width and 0 <= y < self.height):
            side = event.button() == Qt.LeftButton
            if(side and self.draw_state == True):
                x = (x + self.width) % self.width
                y = (y + self.height) % self.height
                x -= self.width//2
                y -= self.height//2
                
                self.lenia.world.add(Board.from_data(self.current_data) ,[int(x), int(y)], SC.SCALE)
                print(Board.from_data(self.current_data).cells.shape)

                for x in range(self.width):
                    for y in range(self.height):
                        state = self.lenia.world.cells[x][y]
                        clr = self.color[int(state * 255)]
                        # c = int((1-state) * 255)
                        # clr = [c, c, c]
                        cur = self.item_index[x][y]
                        cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                        cur[1] = clr
                # self.scene.update()
            
            elif side == False:
                x = int((x - D/2 + self.width) % self.width)
                y = int((y - D/2 + self.height) % self.height)
                for i in range(D):
                    for j in range(D):
                        px = (x + i + self.width) % self.width
                        py = (y + j + self.height) % self.height
                        state = self.lenia.world.cells[px][py]
                        state = max(0, min(state + random.uniform(-0.5, 0.5), 1))
                        self.lenia.world.cells[px][py] = state
                        clr = self.color[int(state* 255)]
                        cur = self.item_index[px][py]
                        cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                        cur[1] = clr

    

    def update_game(self):
        start_time = time.time()
        self.lenia.calc_once()
        mid_time = time.time()

        to_update = []
        non_zero = np.nonzero(self.lenia.change)

        for x, y in zip(non_zero[0], non_zero[1]):
            if(abs(self.lenia.change[x][y]) > 3/255):
                state = self.lenia.world.cells[x][y]
                c = int((1-state) * 255)
                clr = self.color[int(state * 255)]
                # clr = [c, c, c]
                cur = self.item_index[x][y]
                # to_update.append((cur[0], clr[0], clr[1], clr[2]))
                cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                cur[1] = clr

        end_time = time.time()

        for item, r, g, b in to_update:
            item.setBrush(QColor(r, g, b))
        # self.scene.update()
        print(int((mid_time - start_time)*1000), "ms ", int((time.time() - mid_time)*1000), "ms ", 
              int((time.time() - end_time)*1000), "ms ", len(self.scene.items()), len(to_update))


        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.add_cell(x, y, self.lenia.world.cells[x][y])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LifeGame()
    ex.show()
    sys.exit(app.exec_())
    
