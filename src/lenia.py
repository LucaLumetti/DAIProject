import numpy as np
import scipy.signal
import matplotlib.pylab as plt
import matplotlib.animation
import scipy

from video import Video, Image
from aquarium import Aquarium

gaussian = lambda x, m, s: np.exp(-((x-m)/s)**2/2)
sigmoid = lambda x, a, b: 1/(1+np.exp(-a*(x-b)))

class LeniaParam:
    def __init__(self, size, R, k_mean, k_std, b, c, g_mean, g_std, T, h, r):
        self.size = size
        self.R = R
        self.k_mean = k_mean
        self.k_std = k_std
        self.b = np.asarray(b)
        self.c = c
        self.g_mean = g_mean
        self.g_std = g_std
        self.T = T
        self.h = h
        self.r = r
        self.g_mean = g_mean
        self.g_std = g_std
        self.h = h
        self.compute_kernel()

    def compute_kernel(self):
        mid = self.size //2
        self.dT = 1/self.T
        self.D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / self.R * len(self.b) / self.r
        idx = np.minimum(self.D.astype(int), len(self.b)-1)
        self.K = (self.D < len(self.b)) * self.b[idx] * gaussian(self.D % 1, self.k_mean, self.k_std)
        self.K /= np.sum(self.K)
        self.fK = np.fft.fft2(np.fft.fftshift(self.K))

    def get_params(self):
        return {
                'R': self.R,
                'k_mean': self.k_mean,
                'k_std': self.k_std,
                'b': self.b,
                'c': self.c,
                'g_mean': self.g_mean,
                'g_std': self.g_std,
                'T': self.T,
                'h': self.h,
                'r': self.r,
                }

    def growth(self, W):
        g = gaussian(W, self.g_mean, self.g_std)*2-1
        return self.dT*g

class Lenia:
    def __init__(self, size, scale):
        size *= scale
        aquarium = Aquarium.get()

        mid = size//2
        self.size = size
        self.world = np.zeros((3, size, size,))
        # self.world = np.random.rand(3, size, size)

        Cs = np.asarray(aquarium["cells"])
        Cs = scipy.ndimage.zoom(Cs, (1,scale,scale), order=0)

        self.world[:, mid:mid+Cs.shape[1], mid:mid+Cs.shape[2]] = Cs
        self.world[:, mid-20:mid-20+Cs.shape[1], mid-5:mid-5+Cs.shape[2]] = Cs
        self.world[:, mid-30:mid-30+Cs.shape[1], mid:mid+Cs.shape[2]] = Cs
        # self.world[:, mid+30:mid+30+Cs.shape[1], mid+30:mid+30+Cs.shape[2]] = Cs
        self.steps = 0
        self.params = [LeniaParam(size, scale*aquarium["R"], 0.5, 0.15, k["b"], [k["c0"],k["c1"]], k["m"], k["s"], aquarium["T"], k["h"], k["r"]) for k in aquarium["kernels"]]

        # self.params = [
        #     LeniaParam(size, R=13, k_mean=0.5, k_std=0.15, b=[1], c=[0,0], g_mean=0.3, g_std=0.04, T=10, h=1, r=1),
        #     LeniaParam(size, R=13, k_mean=0.5, k_std=0.15, b=[1], c=[0,1], g_mean=0.3, g_std=0.04, T=10, h=0.7, r=1),
        #     LeniaParam(size, R=13, k_mean=0.5, k_std=0.15, b=[1], c=[1,1], g_mean=0.3, g_std=0.04, T=10, h=1, r=1),
        #     LeniaParam(size, R=13, k_mean=0.5, k_std=0.15, b=[1], c=[1,0], g_mean=0.3, g_std=0.04, T=10, h=0.7, r=1),
        # ]

    def set_R(self, R):
        self.params[0].R = R
        for p in self.params:
            p.compute_kernel()

    def set_k_mean(self, k_mean):
        self.params[0].k_mean = k_mean
        for p in self.params:
            p.compute_kernel()

    def set_k_std(self, k_std):
        self.params[0].k_std = k_std
        for p in self.params:
            p.compute_kernel()

    def set_g_mean(self, g_mean):
        self.params[0].g_mean = g_mean
        for p in self.params:
            p.compute_kernel()

    def set_g_std(self, g_std):
        self.params[0].g_std = g_std
        for p in self.params:
            p.compute_kernel()

    def set_T(self, T):
        self.params[0].T = T
        for p in self.params:
            p.compute_kernel()

    def set_b(self, b):
        self.params[0].b = np.asarray(b)
        for p in self.params:
            p.compute_kernel()

    def step(self):
        U = [ np.real(np.fft.ifft2(p.fK * np.fft.fft2(self.world[p.c[0], :, :]))) for p in self.params]
        G = [ p.growth(u) for u,p in zip(U, self.params)]
        H = np.zeros((3, self.size, self.size,))
        for c in range(3):
            H[c] = sum(p.h*g for g,p in zip(G, self.params) if p.c[1] == c)
        # H = [ np.sum(p.h*g for g,p in zip(G, self.params) if p.c[1] == c) for c in range(3)]
        H = np.asarray(H)
        # for i,h in enumerate(H):
        #     print(f"{i}: {np.asarray(h).shape}")

        self.world = np.clip(self.world + H, 0, 1)
        # self.world[:,:,0] = np.clip(self.world[:,:,0] + H[0], 0, 1)
        # self.world[:,:,1] = np.clip(self.world[:,:,1] + H[1], 0, 1)
        # self.world[:,:,2] = np.clip(self.world[:,:,2] + H[2], 0, 1)

        # Video.imwrite(f"frames/frame_%04d.jpg"%self.steps, self.world)
    def reset(self):
        self.world = np.zeros((3, self.size, self.size,))
        self.steps = 0

    def add_gaussian_noise(self, channels=[0,1,2], strong=1):
        for c in channels:
            self.world[c] += np.random.normal(0, 0.1, self.world[c].shape)*strong/10
        self.world = np.clip(self.world, 0, 1)

    def randomize_1D_param(self):
        R = np.random.randint(5, 20)
        k_mean = np.random.rand()
        k_std = np.random.rand()*0.1
        b = [1]
        c = [0,0]
        g_mean = np.random.rand()
        g_std = np.random.rand()*0.1
        T = np.random.randint(5, 15)
        h = 1
        r = 1
        self.params = [LeniaParam(self.size, R, k_mean, k_std, b, c, g_mean, g_std, T, h, r)]


    def get_world_img(self):
        a = self.world.copy()
        a = (a[:3, :, :]*255).astype(np.uint8)
        a = a.swapaxes(0,2).swapaxes(0,1)
        return a

    def get_kernel_img(self):
        a = self.params[0].K.copy()
        max_val = np.max(a)
        m = 255/max_val
        a = (a*m).astype(np.uint8)
        # a = np.dstack([a,a,a])
        return a

    def run(self):
        # for p in self.params:
        #     Image.figure_asset(p.R, p.K, p.growth)
        total_steps = 200

        for i in range(total_steps):
            print(f"step: {i}/{total_steps}")
            self.step()
            self.steps += 1

        # Video.compose('gol.mp4')

if __name__ == "__main__":
    lenia = Lenia(128, 1)
    lenia.run()


