import sys
import random
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QAction
)
from PyQt5.QtCore import Qt, QTimer

# 0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO
def expand(coding):
    s = ""
    num = 1
    i = 0
    while i < len(coding):
        if coding[i].isdigit():
            j = i
            while coding[j].isdigit():
                j += 1
            num = int(coding[i:j])
            i = j-1
        else:
            cur = coding[i]
            if cur.islower() and cur != 'b' and cur != 'o':
                i += 1
                cur += coding[i]
            for _ in range(num):
                s += cur
            num = 1
        i += 1
    return s

def decode(string):
    string = expand(string)
    cells = [[]]
    i = 0
    while i < len(string)-1:
        c = string[i]
        i += 1
        if(c == '$'):
            cells.append([])
            continue
        num = 0
        if(c == '0' or c == 'b' or c == '.'):
            num = 0
        if(c == '1' or c == 'o'):
            num = 1
        if(c.isupper()):
            num = ord(c) - ord('A') + 1
        if(c.islower() and ord(c) >= ord('p')):
            num = (ord(c) - ord('p') + 1) * 24 + (ord(string[i]) - ord('A') + 1)
            i += 1
        cells[-1].append(num)

    mx = 0
    for lst in cells:
        mx = max(mx, len(lst))
    for lst in cells:
        while(len(lst) < mx):
            lst.append(0)
    return cells


class LifeGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Life Game")
        self.setGeometry(100, 100, 1600, 1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_game)
        self.timer.start(100)

        self.height = 100
        self.width = 150
        self.cells = [[False] * self.height for i in range(self.width)]

        self.setMouseTracking(True)
        self.draw_state = False

        self.init_ui()

    def init_ui(self):
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.view)

        self.central_widget.setLayout(vbox)

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


    def add_cell(self, x, y, state):
        rect = QGraphicsRectItem(0, 0, 10, 10)
        rect.setPos(10 * x, 10 * y)
        rect.setBrush(Qt.black if state else Qt.white)
        self.scene.addItem(rect)

    def clear_scene(self):
        for x in range(self.width):
            for y in range(self.height):
                self.cells[x][y] = False
                self.add_cell(x, y, False)

    def random_start(self):
        for x in range(self.width):
            for y in range(self.height):
                self.cells[x][y] = random.choice([True, False])
                self.add_cell(x, y, self.cells[x][y])

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
        string = self.animal_data[location]["cells"]
        print(string)
        self.current_pattern = decode(string)
        print(self.current_pattern)
        self.draw_state = True


    def mousePressEvent(self, event):
        pos = self.view.mapToScene(event.pos())
        x = int((pos.x() - 15) // 10)
        y = int((pos.y() - 44) // 10)
        print("[", pos.x(), ",", pos.y(), "]")
        print("pos_x: ", x, " pos_y: ", y)

        if(0 <= x < self.width and 0 <= y < self.height):
            side = event.button() == Qt.LeftButton
            n = m = 1
            if(side and self.draw_state == True):
                n = len(self.current_pattern)
                m = len(self.current_pattern[0])
                x = (x - (n-1)/2 + self.width) % self.width
                y = (y - (m-1)/2 + self.height) % self.height
                print("n:", n), print("m:", m), print(self.current_pattern)
            else:
                self.current_pattern = [[side]]
            
            for item in self.scene.items():
                if isinstance(item, QGraphicsRectItem):
                    tx = int(item.x() / 10)
                    ty = int(item.y() / 10)
                    px = int((tx - x + self.width) % self.width)
                    py = int((ty - y + self.height) % self.height)
                    if(px >= 0 and px < n and py >= 0 and py < m):
                        self.cells[tx][ty] = self.current_pattern[px][py]
                        item.setBrush(Qt.black if self.cells[tx][ty] else Qt.white)
        self.draw_state = False
    

    def update_game(self):
        new_cells = [[False] * self.height for i in range(self.width)]

        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                live_neighbors = 0
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if i == 0 and j == 0:
                            continue

                        neighbor_x = (x + i + self.width) % self.width
                        neighbor_y = (y + j + self.height) % self.height
                        # if(neighbor_x < 0 or neighbor_x >= self.width or neighbor_y < 0 or neighbor_y >= self.height):
                            # continue
                        live_neighbors += self.cells[neighbor_x][neighbor_y]

                if(cell):
                    if live_neighbors < 2 or live_neighbors > 3:
                        cell = False
                else:
                    if live_neighbors == 3:
                        cell = True
                new_cells[x][y] = cell

        self.cells = new_cells

        for item in self.scene.items():
            self.scene.removeItem(item)

        for x in range(self.width):
            for y in range(self.height):
                self.add_cell(x, y, self.cells[x][y])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LifeGame()
    ex.show()
    sys.exit(app.exec_())
    
