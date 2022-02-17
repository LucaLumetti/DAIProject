import PIL.Image, PIL.ImageDraw
import numpy as np
import glob
import moviepy.editor as mvp
import matplotlib.pylab as plt


class Image:
    def __init__(self):
        pass
    def figure_asset(name, R, K, growth, cmap='viridis', K_sum=1, bar_K=False):
        K_size = K.shape[0]
        K_mid = K_size // 2
        fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,1,2]})
        ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
        ax[0].title.set_text('kernel K')
        if bar_K:
            ax[1].bar(range(K_size), K[K_mid,:], width=1)
        else:
            ax[1].plot(range(K_size), K[K_mid,:])
        ax[1].title.set_text('K cross-section')
        ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
        if K_sum <= 1:
            x = np.linspace(0, K_sum, 1000)
            ax[2].plot(x, growth(x))
        else:
            x = np.arange(K_sum + 1)
            ax[2].step(x, growth(x))
        ax[2].axhline(y=0, color='grey', linestyle='dotted')
        ax[2].title.set_text('growth G')
        fig.savefig('figure_asset_%04d.png'%name)


class Video:
    def __init__(self):
        pass

    def arr2rgb(a):
        if len(a.shape) == 2:
            a = np.dstack((a, a, a))
            return a
        if len(a.shape) == 3:
            return a

    def np2pil(a):
        if a.dtype in [np.float32, np.float64]:
            a = np.uint8(np.clip(a, 0, 1)*255)
        return PIL.Image.fromarray(a)

    def imwrite(f, a, fmt=None):
        a = a[:3, :, :]
        a = (a*255).astype(np.uint8)
        a = a.swapaxes(0,2)
        a = a.swapaxes(0,1)
        if isinstance(f, str):
            fmt = f.rsplit('.', 1)[-1].lower()
            if fmt == 'jpg':
                fmt = 'jpeg'
            f = open(f, 'wb')
        Video.np2pil(a).save(f, fmt, quality=95)

    def compose(f):
        frames = sorted(glob.glob('frames/frame_*.jpg'))
        print(f"frames: {len(frames)}")
        mvp.ImageSequenceClip(frames, fps=30.0).write_videofile(f)


if __name__ == '__main__':
    # size = 256
    # for i in range(100):
    #     a = np.random.randint(2, size=(size, size)).astype(np.uint8)
    #     print(a)
    #     a = Video.arr2rgb(a)
    #     Video.imwrite('frames/test_%04d.jpg'%i, a)
    Video.compose('out.mp4')
