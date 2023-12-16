import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np                          # pip3 install numpy
import scipy                                # pip3 install scipy
import scipy.ndimage as snd
import reikna.fft, reikna.cluda             # pip3 install pyopencl/pycuda, reikna
import PIL.Image, PIL.ImageTk               # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
try: import tkinter as tk
except: import Tkinter as tk
from fractions import Fraction
import copy, re, itertools, json, csv
import io, os, sys, subprocess, datetime, time, multiprocessing
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # suppress warning from scipy.ndimage.zoom()

X2, Y2, P2, PIXEL_BORDER = 8,8,2,0    # GoL 6,6,3,1   Lenia Lo 7,7,2,0  Hi 9,9,0,0   1<<9=512
SIZEX, SIZEY, PIXEL = 1 << X2, 1 << Y2, 1 << P2
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1280//PIXEL, 720//PIXEL    # 720p HD
# PIXEL, PIXEL_BORDER = 1,0; SIZEX, SIZEY = 1920//PIXEL, 1080//PIXEL    # 1080p HD
MIDX, MIDY = int(SIZEX / 2), int(SIZEY / 2)
DEF_R = max(min(SIZEX, SIZEY) // 4 //5*5, 13)
EPSILON = 1e-10
ROUND = 10
STATUS = []
is_windows = (os.name == 'nt')

SCALE = 2

class Board:
    def __init__(self, size=[0,0]):
        self.names = ['', '', '']
        self.params = {'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
        self.param_P = 0
        self.cells = np.zeros(size)

    @classmethod
    def from_values(cls, cells, params=None, names=None):
        self = cls()
        self.names = names.copy() if names is not None else None
        self.params = params.copy() if params is not None else None
        self.cells = cells.copy() if cells is not None else None
        return self

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
        self.params = data.get('params')
        if self.params:
            self.params = self.params.copy()
            self.params['b'] = Board.st2fracs(self.params['b'])
        self.cells = data.get('cells')
        if self.cells:
            if type(self.cells) in [tuple, list]:
                self.cells = ''.join(self.cells)
            self.cells = Board.rle2arr(self.cells)
        return self

    def to_data(self, is_shorten=True):
        rle_st = Board.arr2rle(self.cells, is_shorten)
        params2 = self.params.copy()
        params2['b'] = Board.fracs2st(params2['b'])
        data = {'code':self.names[0], 'name':self.names[1], 'cname':self.names[2], 'params':params2, 'cells':rle_st}
        return data

    def params2st(self):
        params2 = self.params.copy()
        params2['b'] = '[' + Board.fracs2st(params2['b']) + ']'
        return ','.join(['{}={}'.format(k,str(v)) for (k,v) in params2.items()])

    def long_name(self):
        # return ' | '.join(filter(None, self.names))
        return '{0} - {1} {2}'.format(*self.names)

    @staticmethod
    def arr2rle(A, is_shorten=True):
        ''' RLE = Run-length encoding: 
            http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
            http://golly.sourceforge.net/Help/formats.html#rle
            https://www.rosettacode.org/wiki/Run-length_encoding#Python
            0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO '''
        V = np.rint(A*255).astype(int).tolist()  # [[255 255] [255 0]]
        code_arr = [ [' .' if v==0 else ' '+chr(ord('A')+v-1) if v<25 else chr(ord('p')+(v-25)//24) + chr(ord('A')+(v-25)%24) for v in row] for row in V]  # [[yO yO] [yO .]]
        if is_shorten:
            rle_groups = [ [(len(list(g)),c.strip()) for c,g in itertools.groupby(row)] for row in code_arr]  # [[(2 yO)] [(1 yO) (1 .)]]
            for row in rle_groups:
                if row[-1][1]=='.': row.pop()  # [[(2 yO)] [(1 yO)]]
            st = '$'.join(''.join([(str(n) if n>1 else '')+c for n,c in row]) for row in rle_groups) + '!'  # "2 yO $ 1 yO"
        else:
            st = '$'.join(''.join(row) for row in code_arr) + '!'
        # print(sum(sum(r) for r in V))
        return st

    @staticmethod
    def rle2arr(st):
        rle_groups = re.findall('(\d*)([p-y]?[.boA-X$])', st.rstrip('!'))  # [(2 yO)(1 $)(1 yO)]
        code_list = sum([[c] * (1 if n=='' else int(n)) for n,c in rle_groups], [])  # [yO yO $ yO]
        code_arr = [l.split(',') for l in ','.join(code_list).split('$')]  # [[yO yO] [yO]]
        V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row if c!='' ] for row in code_arr]  # [[255 255] [255]]
        # lines = st.rstrip('!').split('$')
        # rle = [re.findall('(\d*)([p-y]?[.boA-X])', row) for row in lines]
        # code = [ sum([[c] * (1 if n=='' else int(n)) for n,c in row], []) for row in rle]
        # V = [ [0 if c in ['.','b'] else 255 if c=='o' else ord(c)-ord('A')+1 if len(c)==1 else (ord(c[0])-ord('p'))*24+(ord(c[1])-ord('A')+25) for c in row ] for row in code]
        maxlen = len(max(V, key=len))
        A = np.array([row + [0] * (maxlen - len(row)) for row in V])/255  # [[1 1] [1 0]]
        # print(sum(sum(r) for r in V))
        return A

    @staticmethod
    def fracs2st(B):
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        return [Fraction(st) for st in st.split(',')]

    def clear(self):
        self.cells.fill(0)

    def scale(self, cells, k):
        n, m = cells.shape
        new_cells = np.zeros((k*n, k*m))
        for i in range(n):
            for j in range(m):
                for a in range(k):
                    for b in range(k):
                        new_cells[i*k+a][j*k+b] = cells[i][j]
        return new_cells

    def add(self, part, shift=[0,0], k = SCALE):
        # assert self.params['R'] == part.params['R']
        part.cells = self.scale(part.cells, k)

        h1, w1 = self.cells.shape
        h2, w2 = part.cells.shape
        h, w = min(h1, h2), min(w1, w2)
        i1, j1 = (w1 - w)//2 + shift[1], (h1 - h)//2 + shift[0]
        i2, j2 = (w2 - w)//2, (h2 - h)//2
        # self.cells[j:j+h, i:i+w] = part.cells[0:h, 0:w]
        vmin = np.amin(part.cells)
        for y in range(h):
            for x in range(w):
                if part.cells[j2+y, i2+x] > vmin:
                    self.cells[(j1+y)%h1, (i1+x)%w1] = part.cells[j2+y, i2+x]
        return self

    def transform(self, tx, mode='RZSF', is_world=False):
        if 'R' in mode and tx['rotate'] != 0:
            self.cells = scipy.ndimage.rotate(self.cells, tx['rotate'], reshape=not is_world, order=0, mode='wrap' if is_world else 'constant')
        if 'Z' in mode and tx['R'] != self.params['R']:
            # print('* {} / {}'.format(tx['R'], self.params['R']))
            shape_orig = self.cells.shape
            self.cells = scipy.ndimage.zoom(self.cells, tx['R'] / self.params['R'], order=0)
            if is_world:
                self.cells = Board(shape_orig).add(self).cells
            self.params['R'] = tx['R']
        if 'F' in mode and tx['flip'] != -1:
            if tx['flip'] in [0,1]: self.cells = np.flip(self.cells, axis=tx['flip'])
            elif tx['flip'] == 2: self.cells[:, :-MIDX-1:-1] = self.cells[:, :MIDX]
            elif tx['flip'] == 3: self.cells[:, :-MIDX-1:-1] = self.cells[::-1, :MIDX]
            elif tx['flip'] == 4: i_upper = np.triu_indices(SIZEX, -1); self.cells[i_upper] = self.cells.T[i_upper]
        if 'S' in mode and tx['shift'] != [0, 0]:
            self.cells = scipy.ndimage.shift(self.cells, tx['shift'], order=0, mode='wrap')
            # self.cells = np.roll(self.cells, tx['shift'], (1, 0))
        return self

    def add_transformed(self, part, tx):
        part = copy.deepcopy(part)
        self.add(part.transform(tx, mode='RZF'), tx['shift'])
        return self

    def crop(self):
        vmin = np.amin(self.cells)
        coords = np.argwhere(self.cells > vmin)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        self.cells = self.cells[y0:y1, x0:x1]
        return self

    def restore_to(self, dest):
        dest.params = self.params.copy()
        dest.cells = self.cells.copy()
        dest.names = self.names.copy()

class Automaton:
    kernel_core = {
        0: lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
        1: lambda r: np.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
        2: lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        3: lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
    }
    field_func = {
        0: lambda n, m, s: np.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        1: lambda n, m, s: np.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        2: lambda n, m, s: (np.abs(n-m)<=s) * 2 - 1  # step (stpz)
    }

    def __init__(self, world = Board()):
        self.world = world
        self.world_FFT = np.zeros(world.cells.shape)
        self.potential_FFT = np.zeros(world.cells.shape)
        self.potential = np.zeros(world.cells.shape)
        self.field = np.zeros(world.cells.shape)
        self.field_old = None
        self.change = np.zeros(world.cells.shape)
        self.X = None
        self.Y = None
        self.D = None
        self.gen = 0
        self.time = 0
        self.is_multi_step = False
        self.is_soft_clip = False
        self.is_inverted = False
        self.kn = 1
        self.gn = 1
        self.is_gpu = True
        self.has_gpu = True
        self.compile_gpu(self.world.cells)
        self.calc_kernel()

    def kernel_shell(self, r):
        B = len(self.world.params['b'])
        Br = B * r
        bs = np.array([float(f) for f in self.world.params['b']])
        b = bs[np.minimum(np.floor(Br).astype(int), B-1)]
        kfunc = Automaton.kernel_core[(self.world.params.get('kn') or self.kn) - 1]
        return (r<1) * kfunc(np.minimum(Br % 1, 1)) * b

    @staticmethod
    def soft_max(x, m, k):
        ''' Soft maximum: https://www.johndcook.com/blog/2010/01/13/soft-maximum/ '''
        return np.log(np.exp(k*x) + np.exp(k*m)) / k

    @staticmethod
    def soft_clip(x, min, max, k):
        a = np.exp(k*x)
        b = np.exp(k*min)
        c = np.exp(-k*max)
        return np.log( 1/(a+b)+c ) / -k
        # return Automaton.soft_max(Automaton.soft_max(x, min, k), max, -k)

    def compile_gpu(self, A):
        ''' Reikna: http://reikna.publicfields.net/en/latest/api/computations.html '''
        self.gpu_api = self.gpu_thr = self.gpu_fft = self.gpu_fftshift = None
        try:
            self.gpu_api = reikna.cluda.any_api()
            self.gpu_thr = self.gpu_api.Thread.create()
            self.gpu_fft = reikna.fft.FFT(A.astype(np.complex64)).compile(self.gpu_thr)
            self.gpu_fftshift = reikna.fft.FFTShift(A.astype(np.float32)).compile(self.gpu_thr)
        except Exception as exc:
            # if str(exc) == "No supported GPGPU APIs found":
            self.has_gpu = False
            self.is_gpu = False
            print(exc)
            # raise exc

    def run_gpu(self, A, cpu_func, gpu_func, dtype, **kwargs):
        if self.is_gpu and self.gpu_thr and gpu_func:
            op_dev = self.gpu_thr.to_device(A.astype(dtype))
            gpu_func(op_dev, op_dev, **kwargs)
            return op_dev.get()
        else:
            return cpu_func(A)
            # return np.roll(potential_shifted, (MIDX, MIDY), (1, 0))

    def fft(self, A): return self.run_gpu(A, np.fft.fft2, self.gpu_fft, np.complex64)
    def ifft(self, A): return self.run_gpu(A, np.fft.ifft2, self.gpu_fft, np.complex64, inverse=True)
    def fftshift(self, A): return self.run_gpu(A, np.fft.fftshift, self.gpu_fftshift, np.float32)
    def calc_once(self, is_update=True):
        A = self.world.cells
        # print("???????")
        # print(A)
        # print("???????")
        dt = 1 / self.world.params['T']
        self.world_FFT = self.fft(A)
        self.potential_FFT = self.kernel_FFT * self.world_FFT
        self.potential = self.fftshift(np.real(self.ifft(self.potential_FFT)))
        # print(self.potential)
        gfunc = Automaton.field_func[(self.world.params.get('gn') or self.gn) - 1]
        #m = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['m']
        #s = (np.random.rand(SIZEY, SIZEX) * 0.4 + 0.8) * self.world.params['s']
        m, s = self.world.params['m'], self.world.params['s']
        self.field = gfunc(self.potential, m, s)
        # print(gfunc)
        if self.is_multi_step and self.field_old:
            D = 1/2 * (3 * self.field - self.field_old)
            self.field_old = self.field.copy()
        else:
            D = self.field
        if not self.is_soft_clip:
            A_new = np.clip(A + dt * D, 0, 1)  # A_new = A + dt * np.clip(D, -A/dt, (1-A)/dt)
        else:
            A_new = Automaton.soft_clip(A + dt * D, 0, 1, 1/dt)  # A_new = A + dt * Automaton.soft_clip(D, -A/dt, (1-A)/dt, 1)
        if self.world.param_P > 0:
            A_new = np.around(A_new * self.world.param_P) / self.world.param_P
        self.change = (A_new - A) / dt
        # print(self.change)
        # print("************")
        if is_update:
            self.world.cells = A_new
            self.gen += 1
            self.time = round(self.time + dt, ROUND)
        if self.is_gpu:
            self.gpu_thr.synchronize()

    def calc_kernel(self):
        I, J = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
        self.X = (I - MIDX) / self.world.params['R']
        self.Y = (J - MIDY) / self.world.params['R']
        self.D = np.sqrt(self.X**2 + self.Y**2)

        self.kernel = self.kernel_shell(self.D)
        self.kernel_sum = self.kernel.sum()
        kernel_norm = self.kernel / self.kernel_sum
        self.kernel_FFT = self.fft(kernel_norm)
        self.kernel_updated = False

    def reset(self):
        self.gen = 0
        self.time = 0
        self.field_old = None

# np.set_printoptions(threshold=np.inf)

# # 创建 Lenia 示例
# lenia = Automaton(Board.from_data({"code":"3U7p","name":"Heptaurium perlongus","cname":"庚尾虫(過長)","params":{"R":13,"T":10,"b":"3/4,1,1","m":0.333,"s":0.042,"kn":1,"gn":1},"cells":"27.2A$23.HUpJpTpVpPpGULD$20.CpFqLrNsLtEtKtFsSsBrGqLpPUF$19.pFrVuAvNwRxOxXxUxKwRvXvCuBsVrMqDWE$18.qSvCyC9yOyFwRuUsTqVpLOC$17.sHyD12yOyNwCtNrKpWXJA$16.tG16yOxMvNtWsJqPpBD$15.sU19yOxVwQvCsXqLO$14.rByI8yOyNyLyM8yOyBxBvWuMsRqTpDD$13.UwSyJ6yOyJxLwVwQwXxRyK7yOyKwPuUtGsAqVpPM$13.uOxE6yOxFuXtVtOtUuUwHxMyDyIyL6yOxNvHtNsDqVpQN$12.rOuTyF5yOwArWqOqHqLrLtFuWwBwOxDxSyIyN5yOyBwUuTsKqOpJL$12.rDvVyN4yOwAV4.pArBsPtFtKtVuRvJvNvP2vDvGvJvWwTxCvLsPqLpIN$12.sGwS4yOyM7.GpCpGpVrEsLtDtOuKvNwMwUwLwIwQxIxJvTsWqOpGJ$11.NtTxR4yOA11.pQqXrLsDtKvCwJxAxNyCyFyBxVxOwItPqRXC$11.qKvOyK3yOpB19.HsGxB4yOyDwKtDpTH$11.sKwVyL2yOsA22.sUyF4yOxMuHqTNA$10.JtRwQyEyOwPpV21.qGtCxF4yOxWvEsMpXI$10.pMsWvOyLyOtWpTC19.tEwUxXyDyJ3yOxLwIvFtLrDM$10.TrPwKyOxOtFqIN19.pExA7yOxGxNxAvBsAH$11.sWyCyOxJuLsBpD20.rRxS4yOyH2yOxLxTvWuDqG$10.VvI3yOwMtJpC8.RpCE9.pGvC4yOyByG2yOxIwFuRrXJ$10.qXwMyM2yOxKtGW7.qTsOqXU7.sBuNuWvHxL4yOyJyM2yOwPvFtNqAED$10.sGwCxTyLyOxAsDpDLKB2.FuFxDvGsUqHH7.wHyOyKvBrHxG3yOuBtKyOwUuWuLtBpJGD$10.sCuRwOyIyOvTrWqIqFpIA2.yN2yOxIuRsKqKW6.rByFyOyAqApKxByOyMsK.rUvBtGtBuTsOpAG$10.qWsXvTyJyOwGtVtArLU.G4yOyLvVuCsKqXpKK5.vByEyOyHrFrAwCxFuMOpLuOtRsWvAvTqUTB$10.XrQvMyJ2yOxAtNpA.pF6yOxJvOtNqXpOpIpKwC2yOsI.qRyNyOvM2.sJsH2.sIvNvExFxSsNqBN$11.rKvTyN2yOxOqJ2.xKyLyN5yOxLvTrLpFH2.3yO2.KyAuG7.uXyGyOyNtUsKrBP$11.rUwO3yOtF2.uFvQvXwCxHyN5yOwGtGqU3.2yOvX3.D7.tF3yOuVuKtTqVOB$11.sGwU2yOwFQ.tKtVtBsLtPwByF9yO3.2yO11.wUyI2yOwFuXuEqTNDB$11.sKwHyIyOsRGpOrIqSpIpTsBvNyB5yOwCwJyOyIyO3.yOW8.rGyEyOyMyJ2yOvRuBqSOCJI$10.FsIvPyGxHrOB5.qKuMxFyK4yOuA.O3yO12.sS3yOyN2yOyNxGvGsRrMrTrFpE$10.FrQvJyDvNqXD5.WsAuRxR2yOxKuQyO3.2yOM11.HvA7yOyFxLwRvJtVsJpU$11.qUvIwXuBrHS5.IpUsJwV4yOuRyO3.yOqH10.pPpGqVxO6yOyAxCwNvLuCtJsSqFB$11.qWvAvLuGsIpN6.pJsJxL2yO.qVyOvIyO13.wIyOyCvM8yOyNwBvSvAuKuJsUpJ$11.qRtMuXvItSqC6.pMtPyN2yO3.2yOtL13.7yOwRxI2yOwBrGtVwNwDvLtMpVIA$11.SrGtMuVuKqKE5.qEuKyJ3yO3.yOtI13.sC2yOxGyK2yOsRrHvPvE3.uQyAwQtWpOHJD$12.pFrNuAuKqVpEWE3.sQsJ.vF3yO12.HtFxMqI.vL2yOvOtWxQyHsRpJ.I3.uPyOxMwArSpEpOpQM$12.UrMuIuPsQ2rOqEC.rDyOwC3.2yOyN12.sS6yOvA.rBwBsS3.pVrGuOyLyOxJxKvUsVsFrPpUN2IB$12.UrVuGvOvTvAuBtCrLpFxO2yOxQ3.uTtF7.pCtTtKS2.6yOuW8.tQxO2yOyGyOyMwAuRtOrLpRpDpAOD$13.qCrRsHtGuLtPtGuCvWsGxR3yO12.wJ2yOpC.3yOqSrMyOrH8.pIsUyF5yOxLwWwGuRtBrJpSTC$13.CqDqUsLuR2uAyOyMtVqCqL3yO9.B2.7yO14.uG9yOwOuQsPrBpXO$14.pBpDqEtIwPyOxLvNyLtIpXpQvUrM7.R2yO2.wK6yOrG10.rLsGsXwO8yOwTtXtStVtItBsFqCH$18.rCvKwOuVvQyNvSpE10.3yOrAuS6yOqR10.sIwX5yOyJ4yOwPtRtKuJuLuNuLtLqVUD$18.QpBvOxLyIyFyOtH10.uE7yOrEqMrOH11.tMyD6yOyKxRvTvDuHuQwAwEvVvXvMtOrApMPC$16.qF2.pGwEyOyD2yOvTM6.yJyF2.7yOuV14.VtCxGyMyA3yOvP2.MrBrXuO2yOxUxGvLtArNqOpIH$20.qI2yOqV8.D3yOsI8yO12.qBtNsNsJuKxVxQvNxKxXqD3.pOpGpWwK3yOwUuKtVtNsApQLB$21.rPyOsN9.xF10yOxAH11.6yOxFQ.qD4.rVrLrQxG4yOvTvQwLvLsUqBNA$17.UuTxTyMyNyOwPpB9.7yOxU15.uF5yOxB7.sGuLwQ6yOxCyJxRvXtHqEKA$18.wPyOvGqOqGV6.yHyNrW.yG7yO12.G2.U2yOrBqFyNuX8.sSxD9yOxMvXtIpUJB$19.2yOK8.uG11yO11.sQ6yO12.qNuFyG9yOwWvPsWpMJA$18.qLxS2yOE8.xB7yO14.tWyOyG4yOvW10.qKsMrUuPyI8yOxMwEvHsEpCMC$15.qHvN4yOxUrD9.7yO11.sFsR.rO7yO10.uWxBvFuQwM5yOyK2yOyBwIvTvEqRpHpBB$16.wVyOrC9.2yOsNuU4yOyF2yO11.7yOpTTtRtQP9.pOvQxXyDxTxVyN3yOwWtQxJyOwRvSwTtLqLqKpJ$16.tT2yO9.7yOvG14.8yO2.ED10.pCwS2yOxIsVvT2yOwIpKsMyMxGvQxCwSsGrVrIO$14.qLxE4yOpK8.7yO11.rNqU.tN4yOyGuIyOwSH13.uHxLyOyBqOXuKyAvGCJxJyDvRwTyOuUtItCqK$13.pFuWyOxCtLrRtMtFP3.wP3.wE6yO11.12yOsF9.FuTwItSpPqPuRxPsD2.qBsR2.uLyAwAwQyOxUuVuJsGJ$14.wMyOtTB7.2yOxTtU3yOqW14.7yOyG2.yOxSrB10.sI2yOU2.sWsR3.pW2.rHxK2wX2yOwJvNtUpKGF$13.uFyN2yOuN8.7yO14.xD4yOvExDyOuO10.AL3.vFyOsS3.O4.F.VwPxCxU2yOxTwPvHqXpBWF$12.uN5yOxJ2.vLwV3.yN6yO11.13yO9.qA2yOtL2.vDxB11.vUwVyF3yOxTwWtAqLqCX$11.rI2yOsJ.SyEyHtIrA3yOvLpMyD2yOxN14.7yOtM.sK3yO10.wQ2yO2.VtI11.vHwOyD5yOvKsIrNpXF$11.wC3yOM6.xP7yO14.xI4yOwQ2yO3.K7.qWuU3.3yOJXqM10.EuRwAxV5yOxXuMtDrET$10.sAyK4yOJ6.7yO11.2yOvS9yOpK9.xJ2yOpB.qB2yOrUpFpSB9.VtRvAxO6yOwPuSsIpL$10.qS5yOwUC3yO2.wB6yOpN10.7yOwI.vJ2yOwU10.yOxDyOuJrN2yOvHqMqIpA9.BsBuFxI7yOwGtKqED$10.sDyOrG.rKuHP.4yOyM4yOpB13.xE7yO3.qV7.rCyO3.yOsVyA3yOyGtTsQrRV9.qLtRxAyJ6yOxOuHqSJ$8.pLuTuM2yOuLsLC4.8yO11.O.uI7yOsE10.tK2yO2.uGwXrAyD3yOvItStOrMO8.pJtEwDxLyF6yOvBrEK$8.xE6yOvA5.qTxV5yOE9.7yOqA.pU2yO11.3yOwCvXyOvAxU2yOwUuKtUuQtVqTpGE6.MrSuEvRxFyK5yOvPrHK$7.rOxB6yOyK3yO2.qExLvI.rH2yOA.FJ6.uC6yO4.tP8.wT3.yOvH6yOxDuVvKwDwSvVtKrID7.pOrRtUwCyD5yOvUrEH$5.VrJqDtAyGyOyMU4.2yOyJ.tGyOyH2.vBrK6.qWyOtS2.3yOwJ3yO12.2yO3.yOpQ5yOvNwR3yOyFvDqH9.pWsQvJxV5yOvQqSC$7.pOsNxD2yOsU5.tLQ2.2yO3.E6.W6yOxM2.UyOtC11.2yOyL2.yMxXyOyH2yOyD6yOsH10.XsBvBxU5yOvCqD$7.xH6yOyL8.yOtR.EH3.pMQ2.7yO13.yOR2.2yOwJtRwBwLvUvDwUyK6yOyI11.ErIuUyB5yOuEpJ$6.pTyOxEtE2yOxC3yO8.pQ5.A3yO.7yOvB11.A2yOX2.yJuVtOrGrLsLsUuCvEvWwTxN3yOqD12.qPvK6yOtAP$7.4yO4.yO3.tBqF10.xV6yOwB2.xPyO8.tTrUDW3yOVBsFrUpA2.pHqOqTqUsGuPwLyDyLrAB12.rMxT5yOxVrMA$4.rDqG.sX4yO7.qLyO11.6yOwS12.rN2yOtGsQ3yOrJsE8.pNtQwHxD.AH11.FwG6yOvTpO$4.tMvVxI5yOT15.vIyF2.tK7yO8.pO2.HyOxCyOuRwI2yOwBpNC8.tGvUpU.OpBE10.vV7yOsPB$4.wFxX2yOrNpErJ2yOwG14.qX2yOwWqKwS2yO.rE3yO7.2yOAqJyOrGwUyOxH2yOXJF8.sFqI2.pPpNF9.wN7yOvRW$4.wCyM2yOqE8.wQyOpV.pFsIT6.2yOtJtN2yO3.2yO7.5yOtTsEwMvAyMuWTJI11.pGqVpUVSpIpQQ3.Q8yOxXqD$4.tRxU4yO7.4yOyJyOrA7.uXrHpN3yO3.rV3.WpU2.wRyOwX2yOvRsTuGvHuKxMsMqUpW10.pMsQsVrSrJsHtKsUqSpIpOrX9yOrA$3.qIuAxI5yO6.6yOrM2.yOwO7.2yOvB3.F2.A2yOV2yO.qRuU2sSwDtQsQvOwFuOsWpJFC4.BpDrMvRwJvUvOwJxAwLvHtPtFvL7yOyHyArK$2.rBtVvJxPwL2pVqLE2.pWyOI.tI2yOvM3yO2.sTuV3.qT4.2yO2.GvSpF2.5yO4.KwQuHsWuWwPvUuQrUrLrVrCpKpCpNrHuI7yOyFxSxT7yOyDwUwIqP$2.tGwEwWyByOS5.qM10yOtC6.wJyO4.yOsJ.W2yOqMN4yOyI5.rOtPtKwBxHwPvTuTvAvJuHsCsIuSxA12yOyDyIyMyFxLwMuUtXW$2.qTxAxQxT2yOV5.rDwCyByOyDrQ4yOyA7.xDyO3.pRyOpVCyM4yOyN.qQsN6.pPqNuGwPwLwMwTwPxFxCwT6yOyHyCyDyHyLyMyHxEvVwIwUwAuOtOsGrL$3.vIxLxUyH2yOxFT5.vSyNyOrG2yOsCxQyHrD7.2yO2.qX2yOrEvR2yOyNyBtU11.pXtCtOuDvAvHwOyHyKxQwUxFxNxAvUvCvDvQwKwPvWtXsDsVtUsUqVpOOH$3.sRwExWvJsVqJI6.uNxO4yO2.tIxQ3.xJ4.2yOpBxE5yOvQtDtH13.pKpSqDrGrTtFvKuStLtCtNtIsBrCqWrIsJtEsOqMCLqBpM$pRqTrRtDvOvNsCpWL4.pSpXrVvKxQxP3yOuO2.D3.xNyO3.2yOwR4yOyMwLrI19.W2qHpUpLpKpIL3.BpDpC$XsItPuGvMxMtJrLpWA5.KuMyMxRxL3yOpU6.2yO2.vA5yOxFpU$.qJtTuXvPwXyJwDyOqQ6.pN2yO.uG2yO2.pIwF3.pK2yOtByG4yOyLrL$2.sCuPvRxAuGsLrTrAO6.2yO2.wExH2.qS2yO3.5yOvKtWsLqJ$2.qMsUuFwArTqXqLqEW6.DyOtP2.sRG2.3yOuGsJ4yOxNqI$.DqTsDtDuSwBrLrKrCqBOJpBS3.qUvFrPpSxMyOqB2.xU6yOxItC$qIrGrQsFsQtRvDtXsJsLrOpTKPpJR2.PrSrCWuR3yOPpX4yOxHpO$pPqTrErQsLtNuKvMvNsCrLrBpDMpUqVqBRpHtRvXrOrA8yOyGpQ$3.pLrNtPqIvMvUtNrLrPrWqRrJsVuGuHuDxW2yOtDvO3yO2yN2yOsX$4.pSsKtOuMrOtGrIpUrPtFtKuHtUwHwVyMwBxLyOyG2yOtXsHrArB$4.pLrB2rKWqXpTRqMtKtWuOtJvLvRyEwDrCrTvXyIyOuQ$2.MpEpTpV2pEqBtBuKpCpWsIsOtCrXtCvTxGxXrX.ApBpR$3.pHpNPBPqHsDtCqDTqHrArDpKqCrWuEuWtI$7.NqHrHqWqA.pTrIrX$8.CNG4.O!"},
# 				))

# lenia.world.cells = np.zeros((SIZEX,SIZEY))

# lenia.world.add(Board.from_data({"code":"3U7p","name":"Heptaurium perlongus","cname":"庚尾虫(過長)","params":{"R":13,"T":10,"b":"3/4,1,1","m":0.333,"s":0.042,"kn":1,"gn":1},"cells":"27.2A$23.HUpJpTpVpPpGULD$20.CpFqLrNsLtEtKtFsSsBrGqLpPUF$19.pFrVuAvNwRxOxXxUxKwRvXvCuBsVrMqDWE$18.qSvCyC9yOyFwRuUsTqVpLOC$17.sHyD12yOyNwCtNrKpWXJA$16.tG16yOxMvNtWsJqPpBD$15.sU19yOxVwQvCsXqLO$14.rByI8yOyNyLyM8yOyBxBvWuMsRqTpDD$13.UwSyJ6yOyJxLwVwQwXxRyK7yOyKwPuUtGsAqVpPM$13.uOxE6yOxFuXtVtOtUuUwHxMyDyIyL6yOxNvHtNsDqVpQN$12.rOuTyF5yOwArWqOqHqLrLtFuWwBwOxDxSyIyN5yOyBwUuTsKqOpJL$12.rDvVyN4yOwAV4.pArBsPtFtKtVuRvJvNvP2vDvGvJvWwTxCvLsPqLpIN$12.sGwS4yOyM7.GpCpGpVrEsLtDtOuKvNwMwUwLwIwQxIxJvTsWqOpGJ$11.NtTxR4yOA11.pQqXrLsDtKvCwJxAxNyCyFyBxVxOwItPqRXC$11.qKvOyK3yOpB19.HsGxB4yOyDwKtDpTH$11.sKwVyL2yOsA22.sUyF4yOxMuHqTNA$10.JtRwQyEyOwPpV21.qGtCxF4yOxWvEsMpXI$10.pMsWvOyLyOtWpTC19.tEwUxXyDyJ3yOxLwIvFtLrDM$10.TrPwKyOxOtFqIN19.pExA7yOxGxNxAvBsAH$11.sWyCyOxJuLsBpD20.rRxS4yOyH2yOxLxTvWuDqG$10.VvI3yOwMtJpC8.RpCE9.pGvC4yOyByG2yOxIwFuRrXJ$10.qXwMyM2yOxKtGW7.qTsOqXU7.sBuNuWvHxL4yOyJyM2yOwPvFtNqAED$10.sGwCxTyLyOxAsDpDLKB2.FuFxDvGsUqHH7.wHyOyKvBrHxG3yOuBtKyOwUuWuLtBpJGD$10.sCuRwOyIyOvTrWqIqFpIA2.yN2yOxIuRsKqKW6.rByFyOyAqApKxByOyMsK.rUvBtGtBuTsOpAG$10.qWsXvTyJyOwGtVtArLU.G4yOyLvVuCsKqXpKK5.vByEyOyHrFrAwCxFuMOpLuOtRsWvAvTqUTB$10.XrQvMyJ2yOxAtNpA.pF6yOxJvOtNqXpOpIpKwC2yOsI.qRyNyOvM2.sJsH2.sIvNvExFxSsNqBN$11.rKvTyN2yOxOqJ2.xKyLyN5yOxLvTrLpFH2.3yO2.KyAuG7.uXyGyOyNtUsKrBP$11.rUwO3yOtF2.uFvQvXwCxHyN5yOwGtGqU3.2yOvX3.D7.tF3yOuVuKtTqVOB$11.sGwU2yOwFQ.tKtVtBsLtPwByF9yO3.2yO11.wUyI2yOwFuXuEqTNDB$11.sKwHyIyOsRGpOrIqSpIpTsBvNyB5yOwCwJyOyIyO3.yOW8.rGyEyOyMyJ2yOvRuBqSOCJI$10.FsIvPyGxHrOB5.qKuMxFyK4yOuA.O3yO12.sS3yOyN2yOyNxGvGsRrMrTrFpE$10.FrQvJyDvNqXD5.WsAuRxR2yOxKuQyO3.2yOM11.HvA7yOyFxLwRvJtVsJpU$11.qUvIwXuBrHS5.IpUsJwV4yOuRyO3.yOqH10.pPpGqVxO6yOyAxCwNvLuCtJsSqFB$11.qWvAvLuGsIpN6.pJsJxL2yO.qVyOvIyO13.wIyOyCvM8yOyNwBvSvAuKuJsUpJ$11.qRtMuXvItSqC6.pMtPyN2yO3.2yOtL13.7yOwRxI2yOwBrGtVwNwDvLtMpVIA$11.SrGtMuVuKqKE5.qEuKyJ3yO3.yOtI13.sC2yOxGyK2yOsRrHvPvE3.uQyAwQtWpOHJD$12.pFrNuAuKqVpEWE3.sQsJ.vF3yO12.HtFxMqI.vL2yOvOtWxQyHsRpJ.I3.uPyOxMwArSpEpOpQM$12.UrMuIuPsQ2rOqEC.rDyOwC3.2yOyN12.sS6yOvA.rBwBsS3.pVrGuOyLyOxJxKvUsVsFrPpUN2IB$12.UrVuGvOvTvAuBtCrLpFxO2yOxQ3.uTtF7.pCtTtKS2.6yOuW8.tQxO2yOyGyOyMwAuRtOrLpRpDpAOD$13.qCrRsHtGuLtPtGuCvWsGxR3yO12.wJ2yOpC.3yOqSrMyOrH8.pIsUyF5yOxLwWwGuRtBrJpSTC$13.CqDqUsLuR2uAyOyMtVqCqL3yO9.B2.7yO14.uG9yOwOuQsPrBpXO$14.pBpDqEtIwPyOxLvNyLtIpXpQvUrM7.R2yO2.wK6yOrG10.rLsGsXwO8yOwTtXtStVtItBsFqCH$18.rCvKwOuVvQyNvSpE10.3yOrAuS6yOqR10.sIwX5yOyJ4yOwPtRtKuJuLuNuLtLqVUD$18.QpBvOxLyIyFyOtH10.uE7yOrEqMrOH11.tMyD6yOyKxRvTvDuHuQwAwEvVvXvMtOrApMPC$16.qF2.pGwEyOyD2yOvTM6.yJyF2.7yOuV14.VtCxGyMyA3yOvP2.MrBrXuO2yOxUxGvLtArNqOpIH$20.qI2yOqV8.D3yOsI8yO12.qBtNsNsJuKxVxQvNxKxXqD3.pOpGpWwK3yOwUuKtVtNsApQLB$21.rPyOsN9.xF10yOxAH11.6yOxFQ.qD4.rVrLrQxG4yOvTvQwLvLsUqBNA$17.UuTxTyMyNyOwPpB9.7yOxU15.uF5yOxB7.sGuLwQ6yOxCyJxRvXtHqEKA$18.wPyOvGqOqGV6.yHyNrW.yG7yO12.G2.U2yOrBqFyNuX8.sSxD9yOxMvXtIpUJB$19.2yOK8.uG11yO11.sQ6yO12.qNuFyG9yOwWvPsWpMJA$18.qLxS2yOE8.xB7yO14.tWyOyG4yOvW10.qKsMrUuPyI8yOxMwEvHsEpCMC$15.qHvN4yOxUrD9.7yO11.sFsR.rO7yO10.uWxBvFuQwM5yOyK2yOyBwIvTvEqRpHpBB$16.wVyOrC9.2yOsNuU4yOyF2yO11.7yOpTTtRtQP9.pOvQxXyDxTxVyN3yOwWtQxJyOwRvSwTtLqLqKpJ$16.tT2yO9.7yOvG14.8yO2.ED10.pCwS2yOxIsVvT2yOwIpKsMyMxGvQxCwSsGrVrIO$14.qLxE4yOpK8.7yO11.rNqU.tN4yOyGuIyOwSH13.uHxLyOyBqOXuKyAvGCJxJyDvRwTyOuUtItCqK$13.pFuWyOxCtLrRtMtFP3.wP3.wE6yO11.12yOsF9.FuTwItSpPqPuRxPsD2.qBsR2.uLyAwAwQyOxUuVuJsGJ$14.wMyOtTB7.2yOxTtU3yOqW14.7yOyG2.yOxSrB10.sI2yOU2.sWsR3.pW2.rHxK2wX2yOwJvNtUpKGF$13.uFyN2yOuN8.7yO14.xD4yOvExDyOuO10.AL3.vFyOsS3.O4.F.VwPxCxU2yOxTwPvHqXpBWF$12.uN5yOxJ2.vLwV3.yN6yO11.13yO9.qA2yOtL2.vDxB11.vUwVyF3yOxTwWtAqLqCX$11.rI2yOsJ.SyEyHtIrA3yOvLpMyD2yOxN14.7yOtM.sK3yO10.wQ2yO2.VtI11.vHwOyD5yOvKsIrNpXF$11.wC3yOM6.xP7yO14.xI4yOwQ2yO3.K7.qWuU3.3yOJXqM10.EuRwAxV5yOxXuMtDrET$10.sAyK4yOJ6.7yO11.2yOvS9yOpK9.xJ2yOpB.qB2yOrUpFpSB9.VtRvAxO6yOwPuSsIpL$10.qS5yOwUC3yO2.wB6yOpN10.7yOwI.vJ2yOwU10.yOxDyOuJrN2yOvHqMqIpA9.BsBuFxI7yOwGtKqED$10.sDyOrG.rKuHP.4yOyM4yOpB13.xE7yO3.qV7.rCyO3.yOsVyA3yOyGtTsQrRV9.qLtRxAyJ6yOxOuHqSJ$8.pLuTuM2yOuLsLC4.8yO11.O.uI7yOsE10.tK2yO2.uGwXrAyD3yOvItStOrMO8.pJtEwDxLyF6yOvBrEK$8.xE6yOvA5.qTxV5yOE9.7yOqA.pU2yO11.3yOwCvXyOvAxU2yOwUuKtUuQtVqTpGE6.MrSuEvRxFyK5yOvPrHK$7.rOxB6yOyK3yO2.qExLvI.rH2yOA.FJ6.uC6yO4.tP8.wT3.yOvH6yOxDuVvKwDwSvVtKrID7.pOrRtUwCyD5yOvUrEH$5.VrJqDtAyGyOyMU4.2yOyJ.tGyOyH2.vBrK6.qWyOtS2.3yOwJ3yO12.2yO3.yOpQ5yOvNwR3yOyFvDqH9.pWsQvJxV5yOvQqSC$7.pOsNxD2yOsU5.tLQ2.2yO3.E6.W6yOxM2.UyOtC11.2yOyL2.yMxXyOyH2yOyD6yOsH10.XsBvBxU5yOvCqD$7.xH6yOyL8.yOtR.EH3.pMQ2.7yO13.yOR2.2yOwJtRwBwLvUvDwUyK6yOyI11.ErIuUyB5yOuEpJ$6.pTyOxEtE2yOxC3yO8.pQ5.A3yO.7yOvB11.A2yOX2.yJuVtOrGrLsLsUuCvEvWwTxN3yOqD12.qPvK6yOtAP$7.4yO4.yO3.tBqF10.xV6yOwB2.xPyO8.tTrUDW3yOVBsFrUpA2.pHqOqTqUsGuPwLyDyLrAB12.rMxT5yOxVrMA$4.rDqG.sX4yO7.qLyO11.6yOwS12.rN2yOtGsQ3yOrJsE8.pNtQwHxD.AH11.FwG6yOvTpO$4.tMvVxI5yOT15.vIyF2.tK7yO8.pO2.HyOxCyOuRwI2yOwBpNC8.tGvUpU.OpBE10.vV7yOsPB$4.wFxX2yOrNpErJ2yOwG14.qX2yOwWqKwS2yO.rE3yO7.2yOAqJyOrGwUyOxH2yOXJF8.sFqI2.pPpNF9.wN7yOvRW$4.wCyM2yOqE8.wQyOpV.pFsIT6.2yOtJtN2yO3.2yO7.5yOtTsEwMvAyMuWTJI11.pGqVpUVSpIpQQ3.Q8yOxXqD$4.tRxU4yO7.4yOyJyOrA7.uXrHpN3yO3.rV3.WpU2.wRyOwX2yOvRsTuGvHuKxMsMqUpW10.pMsQsVrSrJsHtKsUqSpIpOrX9yOrA$3.qIuAxI5yO6.6yOrM2.yOwO7.2yOvB3.F2.A2yOV2yO.qRuU2sSwDtQsQvOwFuOsWpJFC4.BpDrMvRwJvUvOwJxAwLvHtPtFvL7yOyHyArK$2.rBtVvJxPwL2pVqLE2.pWyOI.tI2yOvM3yO2.sTuV3.qT4.2yO2.GvSpF2.5yO4.KwQuHsWuWwPvUuQrUrLrVrCpKpCpNrHuI7yOyFxSxT7yOyDwUwIqP$2.tGwEwWyByOS5.qM10yOtC6.wJyO4.yOsJ.W2yOqMN4yOyI5.rOtPtKwBxHwPvTuTvAvJuHsCsIuSxA12yOyDyIyMyFxLwMuUtXW$2.qTxAxQxT2yOV5.rDwCyByOyDrQ4yOyA7.xDyO3.pRyOpVCyM4yOyN.qQsN6.pPqNuGwPwLwMwTwPxFxCwT6yOyHyCyDyHyLyMyHxEvVwIwUwAuOtOsGrL$3.vIxLxUyH2yOxFT5.vSyNyOrG2yOsCxQyHrD7.2yO2.qX2yOrEvR2yOyNyBtU11.pXtCtOuDvAvHwOyHyKxQwUxFxNxAvUvCvDvQwKwPvWtXsDsVtUsUqVpOOH$3.sRwExWvJsVqJI6.uNxO4yO2.tIxQ3.xJ4.2yOpBxE5yOvQtDtH13.pKpSqDrGrTtFvKuStLtCtNtIsBrCqWrIsJtEsOqMCLqBpM$pRqTrRtDvOvNsCpWL4.pSpXrVvKxQxP3yOuO2.D3.xNyO3.2yOwR4yOyMwLrI19.W2qHpUpLpKpIL3.BpDpC$XsItPuGvMxMtJrLpWA5.KuMyMxRxL3yOpU6.2yO2.vA5yOxFpU$.qJtTuXvPwXyJwDyOqQ6.pN2yO.uG2yO2.pIwF3.pK2yOtByG4yOyLrL$2.sCuPvRxAuGsLrTrAO6.2yO2.wExH2.qS2yO3.5yOvKtWsLqJ$2.qMsUuFwArTqXqLqEW6.DyOtP2.sRG2.3yOuGsJ4yOxNqI$.DqTsDtDuSwBrLrKrCqBOJpBS3.qUvFrPpSxMyOqB2.xU6yOxItC$qIrGrQsFsQtRvDtXsJsLrOpTKPpJR2.PrSrCWuR3yOPpX4yOxHpO$pPqTrErQsLtNuKvMvNsCrLrBpDMpUqVqBRpHtRvXrOrA8yOyGpQ$3.pLrNtPqIvMvUtNrLrPrWqRrJsVuGuHuDxW2yOtDvO3yO2yN2yOsX$4.pSsKtOuMrOtGrIpUrPtFtKuHtUwHwVyMwBxLyOyG2yOtXsHrArB$4.pLrB2rKWqXpTRqMtKtWuOtJvLvRyEwDrCrTvXyIyOuQ$2.MpEpTpV2pEqBtBuKpCpWsIsOtCrXtCvTxGxXrX.ApBpR$3.pHpNPBPqHsDtCqDTqHrArDpKqCrWuEuWtI$7.NqHrHqWqA.pTrIrX$8.CNG4.O!"},
# 				))


# # lenia.world.cells=np.random.rand(*lenia.world.cells.shape)

# lenia.calc_kernel()

# # print(lenia.world.params)

# # print(lenia.world.cells.shape)

# # print(lenia.world.cells)

# # 设置图形窗口
# fig, ax = plt.subplots()
# im = ax.imshow(lenia.world.cells, cmap='jet', interpolation='none')

# def update(frame):
#     lenia.calc_once()
#     im.set_array(lenia.world.cells)
#     return im,

# # 设置动画
# # ani = Func
# ani = FuncAnimation(fig, update, frames=200, interval=1, blit=True)

# plt.show()
