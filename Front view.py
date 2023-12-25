import sys
import random
import json
import os
import time
import threading
import single_channel as SC
import rebuild as RE
import numpy as np
import jax
import jax.numpy as jnp
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
E = 90  # The width of a new creature in flow lenia
HERE = os.path.dirname(os.path.realpath(__file__))

def resource_path(relative_path):
    base_path = getattr(
        sys, '_MEIPASS',
        os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)

class LifeGame(QMainWindow, SC.Automaton, SC.Board, RE.Automaton, RE.Board):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lenia")
        self.setWindowIcon(QIcon(resource_path("orbium.png")))
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
        # self.draw_state = False

        self.init_lenia()
        self.init_ui()
        self.mode = "Lenia"


    def init_ui(self):
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.view)

        self.central_widget.setLayout(vbox)

        self.get_color()
        self.random_start()
        self.clear_scene()
        self.paint_ustc(start = True)
        self.lenia.world.add(Board.from_data(self.current_data), k=1)
        self.paint_ustc()
        def vanish():
            self.lenia.world.params["s"] = 0.005
            self.s.setText(str(0.005))
            self.lenia.calc_kernel()
        countdown = threading.Timer(5, vanish)
        countdown.start()

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

        kernel_core_menu = menubar.addMenu("Kernel-Core")
        self.create_kernel_core_menu(kernel_core_menu)

        field_function_menu = menubar.addMenu("Field-Function")
        self.create_field_function_menu(field_function_menu)

        self.create_toolbar()

        flow_lenia_menu = menubar.addMenu("Flow-Lenia")
        self.create_flow_lenia_menu(flow_lenia_menu)


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
                    if value >= 1:
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
        self.mode = "Lenia"
        self.lenia = SC.Automaton()
        self.lenia.world.cells = np.zeros((SC.SIZEX, SC.SIZEY))
        self.lenia.calc_kernel()

    def init_flow_lenia(self):
        self.mode = "Flow Lenia"
        NBK = 8
        self.flow_lenia = RE.Automaton(RE.Board([SC.SIZEX, SC.SIZEY], 2), NBK)
        self.flow_lenia.is_flow = True
        self.flow_lenia.world.channels = np.zeros((2, SC.SIZEX, SC.SIZEY))
        self.flow_lenia.nb_k = NBK
        self.flow_lenia.world.P = jnp.full((NBK, SC.SIZEX, SC.SIZEY), 0)


        # 80 [0.75, 0.5, 0.75]
        # 30 [0.25, 1.0]
        # 10 [0.75, 0.25]

        # 27 [0.5, 1.0, 0.5]
        # 17 [0.75]
        # 24 [1.0, 1.0, 0.75]
        kernels = []
        # R = [80, 40, 20, 10]
        for _ in range(self.flow_lenia.nb_k):
            num = random.choice([1,2,3])
            b = []
            for i in range(num):
                b.append((random.choice(range(4))+1)/4)  # i/4
            R = random.choice(range(26)) + 5  # 5~30
            # kernels.append(self.flow_lenia.calc_kernel({"R":R, "b":b, "m":m, "s":s, "kn":1}))
            kernels.append(self.flow_lenia.calc_kernel({"R":R, "b":b, "kn":1}))
            print(R, b)
        # s = random.uniform(0, 0.1)
        # m = random.uniform(0.1, 0.5)
        s = 0.05
        m = 0.65  
        self.flow_lenia.world.params = {"s":s, "m":m, "fK":np.array(kernels), "gn":1}

        self.flow_lenia.step_fn = self.flow_lenia.build_system(
            is_flow = True, SX = SC.SIZEX, SY = SC.SIZEY,
            nb_k = 8, C = 2, dd = 5, dt = 0.1, sigma = 0.6, theta_A = 2,
            c0 = [0, 0, 0, 1, 0, 1, 1, 1], c1 = [[0, 2, 6, 7, 4], [1, 3, 5]], n = 1.2
        )

        for x in range(SC.SIZEX):
            for y in range(SC.SIZEY):
                cur = self.item_index[x][y]
                cur[0].setBrush(QColor(0, 0, 0))
                cur[1] = [0, 0, 0]

        # self.flow_lenia.world.add(RE.Board.from_data({"code":"4H3t","name":"Helicium serratus turbulentus","cname":"鋸卷虫(湍)","params":{"R":18,"T":10,"b":"1/4,1/2,3/4,1","m":0.2,"s":0.02,"kn":1,"gn":1},"cells":"53.J$42.pN8.GLG$42.qJ8.pIG.T$13.2LG26.E3.sC4.G2.pN$12.JQOLQpDpIpDL20.V3.LsP7.pPpUpXV$11.LpDVOVpN2pXpPpAJ19.OQ2.qJpU.2BO3.pSqWrLrG$9.LpUqMqJqCqEqMqTqOqEpP2pNpFTLE15.EQpPrGrQ2pDQsKsF3.qCsCrLrS$7.EpKqMqOqJqMqRrArGrNrIqJqEqMqOqMpUpAJB12.ELQpDqHrIrVpUpIpXvMsHBE.qJsFqCrSV$6.EpAqH2rGrDrIrNrQrGrNrVrNrLrNrLrDqJpNVLG7.E2QpFpNpPpXqEqJqOrG2rVtQuCqC2pXpUpNpAOsCrQO$6.BVqMrSrVrSsCsWtGsUsHsKsPtBsUsCrIqWqMqCpNpFpDrGsHtOsW2sHrSrNrDqWqO2qM2qHqErDsRuNtJqCpNpPqHqOqCpKpDsPtJsHqH$6.JqErVrQrIsCtBtVuCtOtBtGtQtVtQsWsC2rVrStJvJxMyIxRwQvEuFtJsPrXrIqTqJqEpNpIpPpNqEqOqRqEpApNrQxRtGpUpNpItJuCuFsW$5.OqOrLpXTpIqRrLrXtGuKuIuPuXuNuItQ3tGxC3yOwQtTrVqHpNpDpA2pDpAVOGJOJE.OpXqMrSwSyLqOqC2pXuNuXuPtV$4.pAqWrApULOqErVsCrXtDuXvTwIwDvJuSuNuPuU3yOyDsPpF17.JqHsMtJsCpDEGTpXuI2uCuI$3.TqRrLrSrIqOrLuAvRwA2vOwDwQwXwSwDvOvRwD3yOxMqM19.OpPqHpP4.BLpKqEtOvBpS$2.JpXrArS2sRrNsMvBwQ2xEwVxE2xPxE2wQxC4yOqJ18.BJTpPqEqC2.qCtBLJLpKuAwDtTqW$.BpApXqTrVsMqHLpKrQsPuAwSyByIyOyGxHwVxJ4yOsFE17.EVpFpKqEqTrAqOrX2wVJ.EpKuP2wAtT$.EVqErNrSpA3.qEqJrQvW4yOxRxEyI3yOxMqE15.pA.EpDpSqEqJqWrSsFsPrXsMpK3.pPuPvBvTuN$2.VrIrSpA3.rX2vTwDyL4yOyLyB4yOuST15.qTpUpIpUqOrGrLrVsPtBsUV5.pDsWtJvOtQ$2.qCsMrApKG.qTwX14yOtO16.sCsKqRqWrNsFsKsRtJtTtGQ.2L4.qJvWsK$.GrXsKrVrXqTOqEuPvJuPwNyL2yOyL7yOrG16.tDuSsMsCsMtDtJtLuFuPuK2sR2wIG3.pUvOsWpI$.qEsPsMsKrXQ3.QpNtBxRyDyBxUxW2yDyByI2yO17.uFwIuItLtQuFuIuFuXvOvR2vGxHuP4.rNuNuFrXO$BrGsWtBsHpI4.qTrVuPxMxRxEwVwXxEwXwVxMyIxR17.vMxJvTuPuS2vGuXvO2wIuIrDqH4.QsKtGuAsM$LqJsPtVrG4.uUxUwVwSxExCwIvWwAwDvTvRwNxWwQ17.wLyBwVvWwAwIwFvWwIwXwQsUJ5.rAsFsMtOrD$.LsHuNpN3.qCxCxHuSvE2vWvMuX2vBuPuKvGxEvM17.xMyIxUxCxExMxEwXxJxUxMvJsKtBtTpP.BqJrAsHtJpA$2.tJuCqRpS3.pPpIEqOuUvBuP3uAtOtGtOvOuI17.yL2yOyDyGyLyBxUyB2yLyBxRyOyGrL3.pPsWrVE$2.uSuAtBsRE5.qCuIuNuAtD2tBsPrXrVtOtD16.pU7yO2yL2yOwXvTwFtL4.rDsRqTT$.TvEuUuFsR4.qRsFsKuAtVtJsM2sCrNqWqJrIsC16.sH5yOyL4yOyDtDqJpX4.qTsFrXqWpK$.pKuNvWvGrG3.qHxRwDrXsMsRsMrSrGrDqOpXpFpDqOO14.TtG4yOxWxR5yOuSsHrQpP2.rDsRsCrSqWQ$.JrIuXwIpP3.qRuApU.L2rQrAqJqEpSpDE.LJ14.qEvW4yOxPxU5yOxWxMwVuF2qW2sPsCrSpX$3.sRwNrDpAE6.rArDqMpPpFpAE17.JsC4yOxJxCxPyLyOyIxUxJwXxEwLtOqHpNqOrLrVqWE$3.sHwDuKuUqRE3.EqErXrApXpAJB18.qMxR3yOxCwIwFwXxRxWxMwIuF2sPrQpA2.pNrLrGQ$3.sPvWvMvGrGpKQVuCxPuIrIqEpFL19.qTxC3yOwAvOvJvTwL2wSwLvWtOrGqRpSLEpUrSrAV$3.rSvEvGuSqHpXpUqHxUwIQ.BTG17.pKuC4yOuPuKuFuNvGvTvWvJuUvGvEuItLsFrGrSsMrDpFG$3.pKsPuFuNqEqCqE2qHOB.GrAqTpSJL2JQTpDpKpPpNpIpNpXqTrVtQxC3yOwItB2tDtGtTuNvBuSuFuAuIuSuNtLsPsKsPrSqEpAG$5.rVtQpPqCqJqEpKVpFqMtQuSsWqTpApFpIpSqEqCqJrArLsCsPtJuFvJwFxHxWwXvJrX2rLrQsCsPtDtOtTtOtBsFrXsMsRsC2rIrQrIqHV$5.pItBrAtBrQpNpPpUsFxEuFqOrGrAqMqOqJ2qO3qMqOqTrGrLsCsRsCrSrDVpDpNpXqEqOrIsF2sK2sFsMrXrGrDqWqMpXqHqWqMpD$6.sWsKtBrD.GJuCwFpU2qHsPsMrNqM2pUpPpFQJB8.EJVpNqCqTrNrQrLrDrGrXsKsFrNrD2qTrApXJ$6.sKsHsCqH2.GrNpPpU2qCtDsMqTpSpFOG14.JpFpPpXqHqMqOqRrDrLrIqO2qEqMqET$6.qHqWrIpU2.G2TQGsRrG.OpXpKL15.BOpAVQVpNqMqOqRqMpUpN2pSV$8.pPpU2.Q3.pPtV3.pUpS18.BEBGVpPqCpSpPpNpPpSpNL$9.pU.qJpS3.VrI3.pX22.G2LJGJQpDVJ$9.pNpFpUpD8.qM3.E$10.pPVL7.pAqH$25.L!"}),[0,0],0)
        # print(self.flow_lenia.world.channels.sum())

        # channels = RE.jnp_to_np(self.flow_lenia.world.channels)
        # P = RE.jnp_to_np(self.flow_lenia.world.P)
        # for x in range(SC.SIZEX):
        #     for y in range(SC.SIZEY):
        #         state = min((channels[0][x][y] + channels[1][x][y])/2, 1)
        #         # print(x, y, abs(self.flow_lenia.change[0][x][y]))
        #         clr = []
        #         for i in range(3):
        #             clr.append(state * P[i][x][y])
        #             clr[-1] = int(clr[-1] * 255)
        #         cur = self.item_index[x][y]
        #         cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
        #         cur[1] = clr


    def create_pattern_menu(self, menu):
        ustc_icon = QAction("USTC", self)
        ustc_icon.triggered.connect(self.paint_ustc)
        menu.addAction(ustc_icon)

        # HERE = os.path.dirname(__file__)
        # FILE_PATH = os.path.join(HERE, "animals.json")
        with open(resource_path("animals.json"), encoding = 'utf-8') as file:
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


    def create_flow_lenia_menu(self, menu):
        action = QAction("Switch to flow lenia", self, checkable = True)
        menu.addAction(action)

        def on_triggered(checked):
            if checked:
                self.init_flow_lenia()
            else:
                self.init_lenia()
                self.clear_scene()
        action.triggered.connect(on_triggered)


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
        if self.mode == "Lenia":
            for item in self.scene.items():
                item.setBrush(QColor(clr[0], clr[1], clr[2]))
            self.lenia.world.cells.fill(0)
        elif self.mode == "Flow Lenia":
            for item in self.scene.items():
                item.setBrush(QColor(0, 0, 0))
            self.flow_lenia.world.channels = np.zeros((2, SC.SIZEX, SC.SIZEY))
        self.scene.update()

    def random_start(self):
        if self.mode == "Flow Lenia":
            return
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

    def paint_ustc(self, start = False):
        data = {"params":{"R":50,"b":"1"},
                "cells":"$$3.2yO$3.2yO$3.2yO$3.21yO$3.24yO$3.25yO$3.26yO$3.2yO20.5yO$3.2yO22.3yO$3.2yO23.3yO$28.3yO$29.2yO$29.2yO$29.2yO$29.2yO$29.2yO$29.2yO$29.2yO$28.3yO$3.2yO23.2yO$3.2yO22.3yO$3.2yO21.3yO$3.4yO15.6yO$3.24yO$3.21yO$3.3yO$3.2yO$3.2yO$$$$$21.yO$8.5yO8.6yO$6.8yO8.9yO$5.10yO9.7yO$4.4yO2.6yO9.4yO$4.3yO5.4yO10.3yO$4.2yO7.4yO10.3yO$3.2yO8.4yO11.2yO$3.2yO9.4yO10.3yO$3.2yO9.4yO11.2yO$3.2yO9.4yO11.2yO$3.2yO9.4yO11.2yO$3.2yO10.4yO10.2yO$3.2yO10.4yO10.2yO$3.3yO9.4yO10.2yO$4.3yO9.4yO9.2yO$4.3yO9.4yO8.3yO$5.3yO8.5yO7.3yO$4.5yO8.4yO6.3yO$3.8yO6.6yO2.5yO$5.7yO6.11yO$9.3yO7.9yO$20.6yO$$$8.3yO$5.5yO$3.6yO$2.5yO$3.3yO$3.3yO$3.2yO$3.2yO25.yO$3.2yO24.2yO$3.2yO24.2yO$3.3yO22.3yO$3.28yO$3.28yO$3.28yO$3.3yO22.3yO$3.2yO24.2yO$3.2yO24.2yO$3.2yO25.yO$3.2yO$3.3yO$3.3yO$2.5yO$2.6yO$5.5yO$7.4yO$$$$12.11yO$10.15yO$8.18yO$7.20yO$6.6yO10.6yO$5.5yO14.5yO$5.3yO17.5yO$4.3yO19.4yO$4.3yO20.4yO$3.3yO22.3yO$3.2yO23.3yO$3.2yO24.2yO$3.2yO24.2yO$3.2yO24.2yO$3.2yO24.2yO$3.2yO24.2yO$3.2yO24.2yO$4.2yO23.2yO$4.2yO22.3yO$4.3yO21.2yO$5.3yO19.3yO$3.6yO18.2yO$3.8yO15.2yO$6.7yO12.2yO$10.3yO11.2yO$24.yO$$!"}
        if(start):
            data["cells"] = "$$$7.3yO$7.3yO$7.3yO$7.3yO$7.3yO$7.3yO$7.4yO$7.40yO$7.44yO$7.46yO$7.47yO$7.49yO$7.50yO$7.51yO$7.51yO$7.4yO38.10yO$7.3yO41.9yO$7.3yO43.7yO$7.3yO44.7yO$7.3yO45.6yO$7.3yO45.7yO$7.2yO47.6yO$57.5yO$57.5yO$57.6yO$58.5yO$58.5yO$58.5yO$58.5yO$58.5yO$58.5yO$58.5yO$58.5yO$58.5yO$58.4yO$57.5yO$57.5yO$57.5yO$56.5yO$7.3yO45.6yO$7.3yO45.5yO$7.3yO44.6yO$7.3yO43.6yO$7.3yO42.6yO$7.4yO39.8yO$7.5yO35.10yO$7.48yO$7.47yO$7.45yO$7.43yO$7.39yO$7.5yO$7.4yO$7.3yO$7.3yO$7.3yO$7.3yO$7.2yO$$$$$$$$$42.5yO$17.7yO18.11yO$14.12yO16.16yO$13.15yO14.20yO$12.17yO16.17yO$11.19yO16.16yO$10.21yO17.13yO$9.22yO18.9yO$9.6yO7.10yO19.6yO$8.6yO9.10yO19.6yO$8.5yO11.9yO20.6yO$7.5yO13.9yO20.5yO$7.5yO14.8yO21.5yO$7.4yO15.8yO21.6yO$6.5yO15.9yO21.5yO$6.4yO17.8yO21.5yO$6.4yO17.8yO22.5yO$6.4yO18.8yO21.5yO$6.4yO18.8yO22.4yO$6.4yO18.8yO22.5yO$6.4yO18.8yO22.5yO$6.4yO19.8yO21.5yO$6.4yO19.8yO22.4yO$6.4yO19.8yO22.4yO$6.5yO19.8yO21.4yO$6.5yO19.8yO21.4yO$6.5yO19.8yO21.4yO$7.5yO19.7yO20.5yO$7.5yO19.8yO19.5yO$7.6yO18.8yO19.5yO$8.6yO17.8yO19.4yO$8.6yO18.8yO17.5yO$9.6yO17.8yO17.5yO$10.6yO17.8yO15.6yO$10.7yO16.9yO13.6yO$8.10yO15.9yO12.7yO$7.13yO14.9yO10.7yO$7.14yO13.11yO6.9yO$7.16yO12.24yO$9.15yO12.22yO$13.12yO11.21yO$17.7yO13.19yO$21.3yO14.17yO$40.13yO$42.9yO$$$$20.yO$17.4yO$14.8yO$11.11yO$8.12yO$5.13yO$5.12yO$5.10yO$5.9yO$7.6yO$7.6yO$7.5yO$7.4yO$7.4yO$7.4yO$7.4yO49.yO$7.4yO48.3yO$7.3yO49.3yO$7.3yO49.3yO$7.3yO49.3yO$7.3yO48.4yO$7.4yO46.5yO$7.55yO$7.55yO$7.55yO$7.55yO$7.55yO$7.55yO$7.55yO$7.55yO$7.4yO46.5yO$7.3yO48.4yO$7.3yO49.3yO$7.3yO49.3yO$7.3yO49.3yO$7.3yO49.3yO$7.4yO48.3yO$7.4yO$7.4yO$7.4yO$7.5yO$7.5yO$7.6yO$5.9yO$5.10yO$5.11yO$5.13yO$8.12yO$11.11yO$14.8yO$16.6yO$19.2yO$$$$$29.11yO$25.19yO$22.25yO$20.29yO$19.31yO$17.35yO$16.37yO$15.39yO$14.19yO2.20yO$13.13yO16.14yO$12.11yO22.12yO$11.10yO26.11yO$10.9yO30.10yO$10.8yO32.9yO$9.7yO35.9yO$9.6yO37.8yO$8.6yO39.8yO$8.5yO41.7yO$7.6yO42.7yO$7.5yO43.7yO$7.5yO44.6yO$6.5yO46.5yO$6.5yO46.6yO$6.4yO47.6yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.4yO48.5yO$6.5yO47.4yO$7.4yO46.5yO$7.5yO45.5yO$8.4yO45.4yO$8.5yO43.5yO$9.4yO43.5yO$9.5yO41.5yO$10.5yO40.5yO$10.6yO38.5yO$7.11yO35.5yO$7.12yO34.5yO$7.14yO31.5yO$7.16yO28.5yO$10.15yO25.5yO$14.12yO23.5yO$17.9yO21.6yO$21.4yO22.5yO$48.3yO$48.2yO$$$$!"
        self.current_data = data

    def paint_animal(self, location):
        print("location: ", location)
        # string = self.animal_data[location]["cells"]
        # print(string)
        # self.current_pattern = decode(string)
        # print(self.current_pattern)
        self.current_data = self.animal_data[location]
        # self.draw_state = True

        board = SC.Board.from_data(self.current_data)
        board.params['R'] *= SC.SCALE
        self.lenia = SC.Automaton(board)
        self.lenia.world.cells = np.zeros((SC.SIZEX, SC.SIZEY))
        self.calc_kernel()
        self.clear_scene()
        self.update_param_text()
        self.kernel_button[self.lenia.world.params["kn"]-1].setChecked(True)
        self.field_button[self.lenia.world.params["gn"]-1].setChecked(True)


    def mousePressEvent(self, event):
        pos = self.view.mapToScene(event.pos())
        x = int((pos.x() - 14) // L)
        y = int((pos.y() - 76) // L)
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
            if side and self.mode == "Lenia":
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
            
            elif side == False and self.mode == "Lenia":
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

            elif side and self.mode == "Flow Lenia":
                x = int((x - E/2 + self.width) % self.width)
                y = int((y - E/2 + self.height) % self.height)

                # kc = random.choice(range(self.flow_lenia.nb_k))
                gene = [random.uniform(0, 1) for i in range(self.flow_lenia.nb_k)]
                # gene = [0 for i in range(self.flow_lenia.nb_k)]
                # gene[kc] = 1

                channels = RE.jnp_to_np(self.flow_lenia.world.channels)
                P = np.array(jax.device_get(self.flow_lenia.world.P))
                # print("P.shape ", self.flow_lenia.world.P)
                # print("numpy shape", P.shape)
                for i in range(E):
                    for j in range(E):
                        px = (x + i + self.width) % self.width
                        py = (y + j + self.height) % self.height
                        # r = np.sqrt((i-E/2.0) * (i-E/2.0) + (j-E/2.0) * (j-E/2.0)) / (E/2.0)
                        # channels[0][px][py] = random.uniform(0.7, 1) * self.flow_lenia.kernel_core[0](r) if r < 1 else 0
                        # channels[1][px][py] = random.uniform(0.7, 1) * self.flow_lenia.kernel_core[0](r) if r < 1 else 0
                        channels[0][px][py] = random.uniform(0, 1)
                        channels[1][px][py] = random.uniform(0, 1)
                        for _ in range(self.flow_lenia.nb_k):
                            P[_][px][py] = gene[_]
                        cur = self.item_index[px][py]
                        clr = [int(gene[i] * (channels[0][px][py] + channels[1][px][py])/2 * 255) for i in range(self.flow_lenia.nb_k)]
                        cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                        cur[1] = clr

                self.flow_lenia.world.channels = jnp.array(channels)
                self.flow_lenia.world.P = jnp.array(P)

    

    def update_game(self):
        start_time = time.time()
        new_P = jnp.array([])
        new_channels = jnp.array([])

        if self.mode == "Lenia":
            self.lenia.calc_once()

        elif self.mode == "Flow Lenia":
            new_channels, new_P = self.flow_lenia.step_fn(
                self.flow_lenia.world.channels, self.flow_lenia.world.P, self.flow_lenia.world.params)
            
            self.flow_lenia.change = np.abs(RE.jnp_to_np(new_channels - self.flow_lenia.world.channels))
            delta_P = new_P - self.flow_lenia.world.P
            for i in range(self.flow_lenia.nb_k):
                self.flow_lenia.change[0] += np.abs(RE.jnp_to_np(delta_P[i]))
            self.flow_lenia.world.channels = new_channels
            self.flow_lenia.world.P = new_P
            # print(np.nonzero(self.flow_lenia.change))

            new_P = jnp.transpose(new_P, (1, 2, 0))
            new_P = RE.jnp_to_np(new_P)
            new_channels = RE.jnp_to_np(new_channels)

        mid_time = time.time()


        non_zero = np.nonzero(self.lenia.change) if self.mode == "Lenia" else np.nonzero(self.flow_lenia.change)

        count = 0
        for x, y in (zip(non_zero[0], non_zero[1]) if self.mode == "Lenia" else zip(non_zero[1], non_zero[2])):
            if self.mode == "Lenia" and abs(self.lenia.change[x][y]) > 3/255:
                state = self.lenia.world.cells[x][y]
                c = int((1-state) * 255)
                clr = self.color[int(state * 255)]
                # clr = [c, c, c]
                cur = self.item_index[x][y]
                # to_update.append((cur[0], clr[0], clr[1], clr[2]))
                cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                cur[1] = clr

            elif self.mode == "Flow Lenia" and new_channels[0][x][y] + new_channels[1][x][y] > 0.05*2:
                state = min((new_channels[0][x][y] + new_channels[1][x][y])/2, 1)
                # print(x, y, abs(self.flow_lenia.change[0][x][y]))
                clr = []
                for i in range(3):
                    clr.append(state * new_P[x][y][i])
                    clr[-1] = int(clr[-1] * 255)
                cur = self.item_index[x][y]
                cur[0].setBrush(QColor(clr[0], clr[1], clr[2]))
                cur[1] = clr
                count += 1

        end_time = time.time()

        # print(int((mid_time - start_time)*1000), "ms ", int((time.time() - mid_time)*1000), "ms ", 
            #   int((time.time() - end_time)*1000), "ms ", len(self.scene.items()), count)


        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.add_cell(x, y, self.lenia.world.cells[x][y])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LifeGame()
    ex.show()
    sys.exit(app.exec_())
    
