import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np                          # pip3 install numpy
import scipy                                # pip3 install scipy
import scipy.ndimage as snd
import PIL.Image, PIL.ImageTk               # pip3 install pillow
import PIL.ImageDraw, PIL.ImageFont
try: import tkinter as tk
except: import Tkinter as tk
from fractions import Fraction
import copy, re, itertools, json, csv
import io, os, sys, subprocess, datetime, time, multiprocessing
import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
import jax.scipy as jsp
from tqdm import tqdm
import typing as t
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


def jnp_to_np(item):
    if isinstance(item, dict):
        return {key: jnp_to_np(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [jnp_to_np(element) for element in item]
    elif isinstance(item, tuple):
        return tuple(jnp_to_np(element) for element in item)
    elif isinstance(item, jnp.ndarray):
        return np.array(item)
    else:
        return item

class Board:
    def __init__(self, size=[0,0], Chn=1):
        self.names = ['', '', '']
        self.params = {'R':DEF_R, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1, 'chn':Chn} # chn:channels_num
        self.param_P = 0
        print(size, Chn)
        self.channels = jnp.zeros(tuple([Chn]+size))

    @classmethod
    def from_values(cls, channels, params=None, names=None):
        self = cls()
        self.names = names.copy() if names is not None else None
        self.params = params.copy() if params is not None else None
        self.channels = channels.copy() if channels is not None else None
        return self

    @classmethod
    def from_data(cls, data):
        self = cls()
        self.names = [data.get('code',''), data.get('name',''), data.get('cname','')]
        self.params = data.get('params')
        if self.params:
            self.params = self.params.copy()
            self.params['b'] = Board.st2fracs(self.params['b'])
        self.params['chn']=1
        self.channels = []
        cells=data.get('cells')
        if cells:
            if type(cells) in [tuple, list]:
                cells = ''.join(cells)
            cells = Board.rle2arr(cells)
            self.channels.append(cells)
        return self
    
    @staticmethod
    def arr2rle(A, is_shorten=True):
        ''' RLE = Run-length encoding: 
            http://www.conwaylife.com/w/index.php?title=Run_Length_Encoded
            http://golly.sourceforge.net/Help/formats.html#rle
            https://www.rosettacode.org/wiki/Run-length_encoding#Python
            0=b=.  1=o=A  1-24=A-X  25-48=pA-pX  49-72=qA-qX  241-255=yA-yO '''
        V = jnp.rint(A*255).astype(int).tolist()  # [[255 255] [255 0]]
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
        A = jnp.array([row + [0] * (maxlen - len(row)) for row in V])/255  # [[1 1] [1 0]]
        # print(sum(sum(r) for r in V))
        return A

    @staticmethod
    def fracs2st(B):
        return ','.join([str(f) for f in B])

    @staticmethod
    def st2fracs(st):
        return [Fraction(st) for st in st.split(',')]

    def clear(self):
        for i in range(0, self.params['chn']):
            self.channel[i].fill(0)

    def add(self, part, shift=[0,0], i=0):
        # assert self.params['R'] == part.params['R']
        h1, w1 = self.channels[i].shape
        h2, w2 = part.channels[0].shape
        h, w = min(h1, h2), min(w1, w2)
        i1, j1 = (w1 - w)//2 + shift[1], (h1 - h)//2 + shift[0]
        i2, j2 = (w2 - w)//2, (h2 - h)//2
        # self.cells[j:j+h, i:i+w] = part.cells[0:h, 0:w]
        vmin = jnp.amin(part.channels[0])
        
        self.channels = np.array(self.channels)
        
        for y in range(h):
            for x in range(w):
                if part.channels[0][j2+y, i2+x] > vmin:
                    self.channels[i][(j1+y)%h1, (i1+x)%w1] = part.channels[0][j2+y, i2+x]
                    
        self.channels = jnp.array(self.channels)
        return self

class Automaton:
    
    kernel_core = [
        lambda r: (4 * r * (1-r))**4,  # polynomial (quad4)
        lambda r: jnp.exp( 4 - 1 / (r * (1-r)) ),  # exponential / gaussian bump (bump4)
        lambda r, q=1/4: (r>=q)*(r<=1-q),  # step (stpz1/4)
        lambda r, q=1/4: (r>=q)*(r<=1-q) + (r<q)*0.5 # staircase (life)
    ]
    field_func = [
        lambda n, m, s: jnp.maximum(0, 1 - (n-m)**2 / (9 * s**2) )**4 * 2 - 1,  # polynomial (quad4)
        lambda n, m, s: jnp.exp( - (n-m)**2 / (2 * s**2) ) * 2 - 1,  # exponential / gaussian (gaus)
        lambda n, m, s: (jnp.abs(n-m)<=s) * 2 - 1  # step (stpz)
    ]
    
    def build_system(self, is_flow : bool, SX : int, SY : int, nb_k : int, C : int, c0 : t.Iterable = None,
                 c1 : t.Iterable = None, dt : float = .5, dd : int = 5,
                 sigma : float = .65, n = 2, theta_A = None) -> t.Callable:
        """
        returns step function State x Params(compiled) --> State

        Params :
            SX, SY : world size
            nb_k : number of kernels
            C : number of channels
            c0 : list[int] of size nb_k where c0[i] is the source channel of kernel i
            c1 : list[list[int]] of size C where c1[i] is the liste of kernel indexes targetting channel i
            dt : float integrating timestep
            dd : max distance to look for when moving matter
            sigma : half size of the Brownian motion distribution
            n : scaling parameter for alpha
            theta_A : critical mass at which alpha = 1
        Return :
        callable : array(C, SX, SY) x params --> array(C, SX, SY)
        """
        kx = jnp.array([
                    [1., 0., -1.],
                    [2., 0., -2.],
                    [1., 0., -1.]
        ])

        ky = jnp.transpose(kx)

        def sobel_x(A):
            return jnp.stack([jsp.signal.convolve2d(A[c, :, :], kx, mode = 'same')
                            for c in range(A.shape[0])], axis = 0)
            
        def sobel_y(A):
            return jnp.stack([jsp.signal.convolve2d(A[c, :, :], ky, mode = 'same')
                            for c in range(A.shape[0])], axis = 0)

        @jax.jit
        def sobel(A):
            return jnp.concatenate((sobel_y(A)[:, :, :, None], sobel_x(A)[:, :, :, None]),
                                axis = 3)


        if theta_A is None:
            theta_A = C

        if c0 is None or c1 is None :
            c0 = [0] * nb_k
            c1 = [[i for i in range(nb_k)]]

        xs = jnp.concatenate([jnp.arange(SX) for _ in range(SY)])
        ys = jnp.arange(SY).repeat(SX)

        x, y = jnp.arange(SX), jnp.arange(SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)

        rollxs = []
        rollys = []
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                rollxs.append(dx)
                rollys.append(dy)
        rollxs = jnp.array(rollxs)
        rollys = jnp.array(rollys)

        @partial(jax.vmap, in_axes = (0, 0, None, None, None))
        def step_flow(rollx, rolly, A, P, mus):
            rollA = jnp.roll(A, (rollx, rolly), axis = (1, 2))
            rollP = jnp.roll(P, (rollx, rolly), axis = (1, 2))
            dpmu = jnp.absolute(pos[None, ...] - jnp.roll(mus, (rollx, rolly), axis = (1, 2))) # (c, x, y, 2)
            sz = .5 - dpmu + sigma #(c, x, y, 2)
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2*sigma)) , axis = 3) / (4 * sigma**2) # (c, x, y)
            nA = rollA * area
            return nA, rollP

        def step_nf(A, P, params):
            """
            Main step
            A : state of the system (SX, SY, C) correct: (C, SX, SY)
            params : compiled paremeters (dict) must contain m, s, h and fK (computed kernels fft)
            """
            #---------------------------Original Lenia------------------------------------
            fA = jnp.fft.fft2(A, axes=(1,2))  # (c,x,y)
            print("C00000000")
            print(c0)
            fAk = fA[c0, :, :]  # (k,x,y)
            '''
            这一行代码通过索引 c0 从 fA 中选择了特定的频率分量。c0 是一个包含每个内核源通道索引的列表。
            因此，fAk 的形状是 (nb_k, SX, SY)，表示选择了特定的内核频率分量。
            '''

            U = jnp.real(jnp.fft.fftshift(jnp.fft.ifft2(params['fK'] * fAk, axes=(1,2))))  # (k,x,y)
            
            gfunc = Automaton.field_func[self.world.params['gn'] - 1]
            G = gfunc(U, params['m'], params['s']) * G # (k,x,y); params['h'] 权重
            H = jnp.stack([G[c1[c], :, :] for c in range(C) ], axis = 0)  # (c,x,y)
            return jnp.clip(A + dt * H, 0., 1)
        
        def step_f(A, P, params):
            """
            Main step
            A : state of the system (SX, SY, C) correct: (C, SX, SY)
            params : compiled paremeters (dict) must contain m, s, h and fK (computed kernels fft)
            """
            #---------------------------Original Lenia------------------------------------
            fA = jnp.fft.fft2(A, axes=(1,2))  # (c,x,y)

            fAk = fA[c0, :, :]  # (k,x,y)
            '''
            这一行代码通过索引 c0 从 fA 中选择了特定的频率分量。c0 是一个包含每个内核源通道索引的列表。
            因此，fAk 的形状是 (nb_k, SX, SY)，表示选择了特定的内核频率分量。
            '''

            U = jnp.real(jnp.fft.fftshift(jnp.fft.ifft2(params['fK'] * fAk, axes=(1,2))))  # (k,x,y)
            
            gfunc = Automaton.field_func[self.world.params['gn'] - 1]
            G = gfunc(U, params['m'], params['s']) * P  # (k,x,y); params['h'] 权重
            H = jnp.stack([G[c1[c], :, :].sum(axis=0) for c in range(C) ], axis = 0)  # (c,x,y)

            #-------------------------------FLOW------------------------------------------

            F = sobel(H) #(c, x, y, 2)

            C_grad = sobel(A.sum(axis = 0, keepdims = True)) #(1, x, y, 2)

            alpha = jnp.clip((A[:, :, :, None]/theta_A)**n, .0, 1.) #(c, x, y, ?)

            F = jnp.clip(F * (1 - alpha) - C_grad * alpha, -(dd - sigma), +(dd-sigma))

            mus = pos[None, ...] + dt * F #(c, x, y, 2) : target positions (distribution centers)
            nA, nP = step_flow(rollxs, rollys, A, P, mus) #((2d+1)**2, c, x, y) ((2d+1)**2, nb_k, x, y)
            
            nP = jnp.sum(nP * nA.sum(axis = 1, keepdims = True), axis = 0)
            nA = jnp.sum(nA, axis = 0)
            nP = nP / (nA.sum(axis = 0, keepdims = True) + 1e-10)
            
            return nA, nP

        if is_flow == False:
            return jax.jit(step_nf)
        else:
            return jax.jit(step_f)
    

    def __init__(self, world, nb_k):
        self.world = world
        self.world.P = jnp.array((nb_k, SIZEX, SIZEY))
        self.chn = world.params['chn']
        self.nb_k = 0

    def kernel_shell(self, r, params):
        B = len(params['b'])
        Br = B * r
        bs = np.array([float(f) for f in params['b']])
        b = bs[np.minimum(np.floor(Br).astype(int), B-1)]
        kfunc = Automaton.kernel_core[(params.get('kn')) - 1]
        return (r<1) * kfunc(np.minimum(Br % 1, 1)) * b
    
    def calc_kernel(self, params):
        I, J = np.meshgrid(np.arange(SIZEX), np.arange(SIZEY))
        self.X = (I - MIDX) / params['R']
        self.Y = (J - MIDY) / params['R']
        self.D = np.sqrt(self.X**2 + self.Y**2)
        self.kernel = self.kernel_shell(self.D, params)
        self.kernel_sum = self.kernel.sum()
        kernel_norm = self.kernel / self.kernel_sum
        self.kernel_FFT = jnp.fft.fft2(kernel_norm)
        return self.kernel_FFT


"""
returns step function State x Params(compiled) --> State

Params :
    SX, SY : world size
    nb_k : number of kernels
    C : number of channels
    c0 : list[int] of size nb_k where c0[i] is the source channel of kernel i
    c1 : list[list[int]] of size C where c1[i] is the liste of kernel indexes targetting channel i
    dt : float integrating timestep
    dd : max distance to look for when moving matter
    sigma : half size of the Brownian motion distribution
    n : scaling parameter for alpha
    theta_A : critical mass at which alpha = 1
Return :
callable : array(C, SX, SY) x params --> array(C, SX, SY)
"""

"""
np.set_printoptions(threshold=np.inf)

# 创建 Lenia 示例
lenia = Automaton(Board([SIZEX,SIZEY],3), 3)
lenia.world = Board.from_data({"code":"2GA2v","name":"Aerogeminium volitans","cname":"空雙子虫(飛)","params":{"R":18,"T":10,"b":"1,11/12","m":0.32,"s":0.051,"kn":1,"gn":1},"cells":"12.pIqMrGrLrGqOpSTB$8.rXvR8yOyLwDtGqEE$5.pDwS14yOwDrXV$4.vOxW16yOwXsWpXG$2.pSuFxCyL18yOvEsFpXTG5B$.TsFvOyB21yOxMuNsRrSrNrSsMtBsRrGpFB$.pNtJwVyL8yOvBsCrXuPyI12yOyGyD4yOwXtJpN$.qRuUxW7yOsC6.uSyB18yOyIsRE$.sFwLyL6yO9.pPuUxP18yOwFpI$VuAxR6yOqM11.qTvMyD18yOqW$qMvMyI6yOrLB11.VuIxUyL17yOsC$rVwQ7yOtVpN2JOLB7.qHuXxCwVvOuNuFuSvRwQxJyD9yOsK$sUxH7yOxCrQqMrA2rQqRQ6.qJuPuXsFT8.sP8yOrI$tJxM7yOxHrGqOrQsRtDsPpX6.vTyIwSqE10.pA8yOB$tLxM7yOqW.EpDqHqWqMJ5.qR2yOyLqR11.pU7yOxM$sWwXyL6yO13.tG3yOB12.vJ6yOxUqM$rVvWyB5yOsC13.yD2yOxJ14.6yOxRvB$qCuCwSyByL3yOpX13.yB2yOuC5.GqEqMpKE3.B6yOxJuNL$.rQuNwIxPyL2yOsPJ11.rXxWyLxWpF5.pAsKtGsHqML2.pNyI5yOxCtTqH$.LrQtVvRxEyIyOwSrXV10.vM2yIwA7.rAsWtBrXqHpIqEtG6yOwQsWpF$3.qRsUuXwXyIyOyBuUsWsHsFrNqHVB2.vT3yOvE7.BpPqRqWqTrVvM7yOwFrV$4.pDrQuIwSyI8yOxHvMxM4yOyL11.pDtT7yOyLvOqT$6.qOtQwNyD15yOB11.uC7yOyGuSpP$7.pFtBwIxW2yIyGyDyGyL8yOxJrAJ9.yD7yOxPtQE$8.GtGwVyDxRvRvMwS2wXwQwDvGvJxM4yOxWtJpS7.8yOwNsC$10.rNtJsHpP2.L2pKO.pDsUwQyL5yOtVpN4.pS7yOyBuXqC$23.rXwDxUyGyL3yOxEsMqOqMrXuU6yOyLwQsR$24.pAvBwXxPyD4yOyBxHyB4yO2yLyGxCuFpU$26.pDtDxEyGyL5yOyIyBxRxPxMxEwIuKqW$28.sKxMyDyBxUxPxEwQwAvOvGvEuXuItBqO$28.VtTwSwFuUtTtBsKrXrQ2rNrDqHQ$30.pUqCT!"})

lenia.world.channels = np.zeros((1,SIZEX,SIZEY))

lenia.world.add(Board.from_data({"code":"2GA2v","name":"Aerogeminium volitans","cname":"空雙子虫(飛)","params":{"R":18,"T":10,"b":"1,11/12","m":0.32,"s":0.051,"kn":1,"gn":1},"cells":"12.pIqMrGrLrGqOpSTB$8.rXvR8yOyLwDtGqEE$5.pDwS14yOwDrXV$4.vOxW16yOwXsWpXG$2.pSuFxCyL18yOvEsFpXTG5B$.TsFvOyB21yOxMuNsRrSrNrSsMtBsRrGpFB$.pNtJwVyL8yOvBsCrXuPyI12yOyGyD4yOwXtJpN$.qRuUxW7yOsC6.uSyB18yOyIsRE$.sFwLyL6yO9.pPuUxP18yOwFpI$VuAxR6yOqM11.qTvMyD18yOqW$qMvMyI6yOrLB11.VuIxUyL17yOsC$rVwQ7yOtVpN2JOLB7.qHuXxCwVvOuNuFuSvRwQxJyD9yOsK$sUxH7yOxCrQqMrA2rQqRQ6.qJuPuXsFT8.sP8yOrI$tJxM7yOxHrGqOrQsRtDsPpX6.vTyIwSqE10.pA8yOB$tLxM7yOqW.EpDqHqWqMJ5.qR2yOyLqR11.pU7yOxM$sWwXyL6yO13.tG3yOB12.vJ6yOxUqM$rVvWyB5yOsC13.yD2yOxJ14.6yOxRvB$qCuCwSyByL3yOpX13.yB2yOuC5.GqEqMpKE3.B6yOxJuNL$.rQuNwIxPyL2yOsPJ11.rXxWyLxWpF5.pAsKtGsHqML2.pNyI5yOxCtTqH$.LrQtVvRxEyIyOwSrXV10.vM2yIwA7.rAsWtBrXqHpIqEtG6yOwQsWpF$3.qRsUuXwXyIyOyBuUsWsHsFrNqHVB2.vT3yOvE7.BpPqRqWqTrVvM7yOwFrV$4.pDrQuIwSyI8yOxHvMxM4yOyL11.pDtT7yOyLvOqT$6.qOtQwNyD15yOB11.uC7yOyGuSpP$7.pFtBwIxW2yIyGyDyGyL8yOxJrAJ9.yD7yOxPtQE$8.GtGwVyDxRvRvMwS2wXwQwDvGvJxM4yOxWtJpS7.8yOwNsC$10.rNtJsHpP2.L2pKO.pDsUwQyL5yOtVpN4.pS7yOyBuXqC$23.rXwDxUyGyL3yOxEsMqOqMrXuU6yOyLwQsR$24.pAvBwXxPyD4yOyBxHyB4yO2yLyGxCuFpU$26.pDtDxEyGyL5yOyIyBxRxPxMxEwIuKqW$28.sKxMyDyBxUxPxEwQwAvOvGvEuXuItBqO$28.VtTwSwFuUtTtBsKrXrQ2rNrDqHQ$30.pUqCT!"}),[0,0],0)

lenia.world.params['b'] = [float(i) for i in lenia.world.params['b']]


lenia.world.params

print("GNNNNN", lenia.world.params['gn'])

lenia.nb_k = 3
kernels = np.array([lenia.calc_kernel(lenia.world.params) for i in range(lenia.nb_k)])
lenia.world.params['kn'] = 2
kernels[1] = lenia.calc_kernel(lenia.world.params)
lenia.world.params['R'] = 60
lenia.world.params['b'] = [1, 1, 1]
lenia.world.params['s'] = 0.07
kernels[2] = lenia.calc_kernel(lenia.world.params)

lenia.world.params['fK'] = kernels
lenia.world.params['h'] = jnp.array(1.0)

# Convert lenia.world.channels to a JAX array
lenia.world.channels = jnp.array(lenia.world.channels)

random_array = np.random.rand(1, SIZEX, SIZEY)

# 将数组值缩放到 [0, 1.0] 范围内
scaled_array = random_array / (np.max(random_array) * 5)


lenia.world.channels = jnp.array([lenia.kernel] + scaled_array)

# Convert lenia.world.params values to JAX arrays if they are not already
lenia.world.params = {key: jnp.array(value) if isinstance(value, np.ndarray) else value for key, value in lenia.world.params.items()}

# Ensure that lenia.world.params['fK'] is a JAX array
# lenia.world.params['fK'] = jnp.array([lenia.world.params['fK'],lenia.world.params['fK']])
print(lenia.world.params['fK'].shape)
for key, value in lenia.world.params.items():
    print("Key: {",key,"}, Value: {",type(value),"}")

lenia.nb_k = 3

step_fn = lenia.build_system(True, SIZEX, SIZEY, lenia.nb_k, 1, dd = 5, dt = (1.0 / lenia.world.params['T']), sigma = 0.55, theta_A=0.88)

lenia.world.P = jnp.array(np.random.rand(3, SIZEX, SIZEY))
sm = lenia.world.P.sum(axis = 0)
lenia.world.P = lenia.world.P / sm

# lenia.world.P = jnp.array([jnp.full((128, 128), 0.3), jnp.full((128,128),0.3), jnp.full((128, 128), 0.4)])


# Print the types
print(type(lenia.world.channels), type(lenia.world.params))

# step_fn(lenia.world.channels, lenia.world.params)

# lenia.world.cells=np.random.rand(*lenia.world.cells.shape)

# lenia.calc_kernel()

# print(lenia.world.params)

# print(lenia.world.cells.shape)

# print(lenia.world.cells)

fig, ax = plt.subplots()
im = ax.imshow(lenia.world.channels[0], cmap='jet', interpolation='none')


print(lenia.world.channels.sum())

def update(frame):
    lenia.world.channels, lenia.world.P = step_fn(lenia.world.channels, lenia.world.P, lenia.world.params)
    # lenia.calc_once()
    im.set_array(lenia.world.P[2])
    print(lenia.world.channels.sum())
    return im,

update(1)

# 设置动画
# ani = Func
ani = FuncAnimation(fig, update, frames=200, interval=1, blit=True)

plt.show()
"""