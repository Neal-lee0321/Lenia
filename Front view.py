import sys
import random
import json
import os
import time
import single_channel as SC
import numpy as np
from single_channel import Automaton as Automaton
from single_channel import Board as Board
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPen


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
                
                self.lenia.world.add(Board.from_data(self.current_data) ,[int(x), int(y)])
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
    
