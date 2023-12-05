import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QAction
)
from PyQt5.QtCore import Qt, QTimer

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
        ustc_icon = QAction("USTC", self)
        ustc_icon.triggered.connect(self.paint_ustc)
        pattern_menu.addAction(ustc_icon)


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

    def mousePressEvent(self, event):
        pos = self.view.mapToScene(event.pos())
        x = int((pos.x() - 5) // 10) - 1
        y = int((pos.y() - 5) // 10) - 1
        print("[", pos.x(), ",", pos.y(), "]")
        print("pos_x: ", x, " pos_y: ", y)

        if(0 <= x < self.width and 0 <= y < self.height):
            side = event.button() == Qt.LeftButton
            self.cells[x][y] = side

            for item in self.scene.items():
                if isinstance(item, QGraphicsRectItem):
                    if int(item.x() / 10) == x and int(item.y() / 10) == y:
                        item.setBrush(Qt.black if side else Qt.white)
    

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
    
