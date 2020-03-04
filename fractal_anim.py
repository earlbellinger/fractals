## Animate fractals from an iterated function system 
## by Earl Patrick Bellinger
## earlbellinger@gmail.com 
## earlbellinger.com 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from easing import easing
from PIL import ImageColor # for rgb
from numba import jit
from joblib import Parallel, delayed
import os

n_jobs = int(os.environ['OMP_NUM_THREADS'])

# maybe_sub.sh -p 20 python3 fractal_anim.py

N = 10**5 # number of points to generate 
n_frames = 100 # number of frames in between each keyframe 
width = 6 # inches wide 
n = 3 # degree of the easing polynomial 
n_first = 3 # in case we want a different easing for the first keyframe 
#ns = [1, 1, 3, 3, 3, 3, 3, 3, 3, 1] # in case we want a list of easings 

randints = np.random.randint(0, 2, N)

keyframes = [ # parameters alpha, beta, gamma, delta 
    [0.5    - 0.5j,    0,           0.5   - 0.5j,   0],               # dragon 
    [0.4614 + 0.4614j, 0,           0.622 - 0.196j, 0],               # shell 
    [0.5    - 0.5j,    0,           0.5   + 0.5j,   0],               # flex 
    [0.7    - 0.4614j, 0,           0,              0    - 0.5j],     # bush 
    [0.7    - 0.4614j, 0,           0,              0    + 0.45j],    # static
    [0,                0.5 + 0.5j,  0,              -0.5 + 0.5j],     # stars 
    [0,                0.3 + 0.3j,  0,              0.82],            # leaves 
    [0,                0.5 + 0.5j,  0.5,            0],               # sierp.
    [0,                0.5 + 0.5j,  0,              0.4    - 0.4j],   # tri.
    [0.4614 + 0.4614j, 0,           0,              0.2896 - 0.585j], # static2 
]

colors = [
    ["#0B86A7", "#183E56"], # blues 
    ["#003049", "#D62828"], # blue and red 
    ["#0B86A7", "#183E56"], # blues 
    ["#679436", "#3C4600"], # foresty colors 
    ["#D62828", "#003049"], # red and blue 
    ["#353c16", "#562512"], # foliage
    ["#679436", "#3C4600"], # foresty colors 
    ["#003049", "#D62828"], # blue and red 
    ["#0B86A7", "#183E56"], # blues 
    ["#D62828", "#003049"], # red and blue 
]

keyframes = np.array([
        np.array([[x.real, x.imag] for x in keyframe]).flatten()
    for keyframe in keyframes])

keyframes = np.vstack((keyframes, keyframes[0])) # for looping 

@jit(nopython=True)
def F1(z, alpha, beta):
    return alpha*z + beta*np.conjugate(z)

@jit(nopython=True)
def F2(z, gamma, delta):
    return gamma*(z-1) + delta*(np.conjugate(z)-1) + 1

def contractions(alpha_re=0.4614, alpha_im=0.4614,
                 beta_re=0, beta_im=0,
                 gamma_re=0, gamma_im=0,
                 delta_re=0.2896, delta_im=-0.585):
    
    alpha = alpha_re + alpha_im*1j
    beta  = beta_re  + beta_im*1j
    gamma = gamma_re + gamma_im*1j
    delta = delta_re + delta_im*1j
    
    x = np.empty(N, dtype=complex)
    x[0] = 0. #1. #np.random.random() #0.1 
    
    #randints = np.random.randint(0, 2, N)
    for ii in range(1, N):
        if randints[ii]:
            x[ii] = F1(x[ii-1], alpha, beta)
        else:
            x[ii] = F2(x[ii-1], gamma, delta)
    
    return x

def save_frame(frame,
               filename='test.png',
               cols=colors[0],
               xlim=None,
               ylim=None):
    
    alpha_re, alpha_im, beta_re, beta_im, \
        gamma_re, gamma_im, delta_re, delta_im = frame 
    
    alpha = alpha_re + alpha_im*1j
    beta  = beta_re  + beta_im*1j
    gamma = gamma_re + gamma_im*1j
    delta = delta_re + delta_im*1j
    
    x = contractions(*frame)
    
    fig = plt.figure()
    fig.set_size_inches(int(width * (1 + 5 ** 0.5) / 2), width)
    
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    #plt.plot(x.real, x.imag, 'k.', alpha=0.444, ms=1, zorder=1)
    plt.scatter(x.real[1:], x.imag[1:], alpha=0.8, s=2.5, marker='.',
        c=[cols[r] for r in randints[:-1]])
        #c='k')
    
    mpl.rcParams['font.size'] = 18
    plt.annotate(s=r'$\{ z \rightarrow \alpha z + \beta \bar z,\quad' + \
            r'z \rightarrow \gamma (z-1) + \delta (\bar z - 1) + 1 \}$',
        xy=(192, 572),
        xycoords='figure pixels',
        c='#1669BA', alpha=0.3, zorder=0)
    
    mpl.rcParams['font.size'] = 22
    plt.annotate(s=r'Earl Patrick Bellinger', 
        #xy=(627, 10),
        xy=(577, 10),
        xycoords='figure pixels', 
        c='#1669BA', alpha=0.3, zorder=0)
    
    mpl.rcParams['font.size'] = 18
    plt.annotate(s=r'$\alpha = ' + f'{alpha:.2f}' + ',$' \
                 + r'$\beta  = ' + f'{beta:.2f}'  + ',$\n' \
                 + r'$\gamma = ' + f'{gamma:.2f}' + ',$' \
                 + r'$\delta = ' + f'{delta:.2f}' + '$',
        xy=(5, 10),
        xycoords='figure pixels',
        c='#1669BA', zorder=0, alpha=0.3)
    
    plt.savefig(filename)
    plt.close()

#save_frame(keyframes[5], xlim=[-0.5, 1.5], ylim=[-1, 1], cols=colors[5])
#quit()

k = 0
first = None
prev = None
for ii in tqdm(range(len(keyframes)-1)):
    a = keyframes[ii]
    b = keyframes[ii+1]
    
    col_a = colors[ii     % len(colors)]
    col_b = colors[(ii+1) % len(colors)]
    
    n_ = n
    if prev is None:
        a_frame = contractions(*a)
        first = a_frame
        n_ = n_first
    else:
        a_frame = prev
    if ii < len(keyframes) - 1:
        b_frame = contractions(*b)
    else:
        b_frame = first 
        n_ = n_first
        col_b = colors[0]
    prev = b_frame
    #n_ = ns[ii]
    
    #frames = np.linspace(a, b, num=n_frames)
    frames = easing.Eased(np.vstack((a,b))).power_ease(n=n_, 
        smoothness=n_frames)[:n_frames-1]
    
    # plot limits 
    a_xlim  = [min(a_frame.real)-0.02, max(a_frame.real)+0.02]
    b_xlim  = [min(b_frame.real)-0.02, max(b_frame.real)+0.02]
    a_ylim  = [min(a_frame.imag)-0.19, max(a_frame.imag)+0.10]
    b_ylim  = [min(b_frame.imag)-0.19, max(b_frame.imag)+0.10]
    xlims   = easing.Eased(np.vstack((a_xlim, b_xlim))).power_ease(n=n_,
        smoothness=n_frames)[:n_frames-1]
    ylims   = easing.Eased(np.vstack((a_ylim, b_ylim))).power_ease(n=n_,
        smoothness=n_frames)[:n_frames-1]
    
    # colors 
    rgb_a = np.array([ImageColor.getrgb(col) for col in col_a]).flatten()
    rgb_b = np.array([ImageColor.getrgb(col) for col in col_b]).flatten()
    rgb_cols = easing.Eased(np.vstack((rgb_a, rgb_b))).power_ease(n=n_,
        smoothness=n_frames)[:n_frames-1]
    
    hexs = []
    for jj in range(len(rgb_cols)):
        rgb = np.array(rgb_cols[jj], dtype=int)
        rgb = [tuple(rgb[a:a+3]) for a in range(0, len(rgb), 3)]
        hexs += [['#%02x%02x%02x' % a for a in rgb]]
    
    # now calculate! 
    k = ii * len(frames)
    
    Parallel(n_jobs=n_jobs)(delayed(save_frame)(frame,
            filename='plots/'+str(k+jj).zfill(5)+'.png',
            cols=hexs[jj],
            xlim=xlims[jj],
            ylim=ylims[jj]) 
        for jj, frame in enumerate(frames))
    
    """
    for jj, frame in enumerate(frames):
        save_frame(frame,
            filename='plots/'+str(k+jj).zfill(5)+'.png',
            cols=hexs[jj],
            xlim=xlims[jj],
            ylim=ylims[jj])
    """

# ffmpeg -y -framerate 55 -i plots/%05d.png -ab 128k -r 30 -vcodec libx264 -crf 18 -preset veryslow fractals.avi

# maybe_sub.sh -p 1 ffmpeg -y -framerate 30 -i plots/%05d.png -ab 128k -r 30 -vcodec libx264 -crf 18 -preset veryslow fractals.avi
