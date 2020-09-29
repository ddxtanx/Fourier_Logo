import numpy
from sage.all import *

def fourier_coef_dft(pts, n):
    N = len(pts)
    return sum([pt * numpy.exp(-2*numpy.pi*1j*n*i/N) for i,pt in enumerate(pts)])
def coefs_svg_map(tupl):
    i,pts = tupl
    return (i, fourier_coef_dft(pts, i))

def ptt(p): return (p.real(), p.imag())
def circles_anim(tv, cfs, res):
    circs = []
    lines = []
    current_sum = cfs[0][1]/res
    for i, tp in enumerate(cfs[1:]):
        freq, coef = tp
        alpha = 1/(i+1.6)
        addend = coef/res * numpy.exp(2*numpy.pi*1j*freq*tv)
        next_sum = current_sum + addend
        r = abs(addend)
        color = "#00AAFF"
        center = ptt(current_sum)
        circ = circle(center, r, alpha=alpha, thickness=3, edgecolor=color)
        linear = line([ptt(current_sum), ptt(next_sum)], alpha=alpha, thickness=3, color=color)
        current_sum = next_sum
        circs.append(circ)
        lines.append(linear)
    obj = sum(circs+lines)
    return obj

def send_to_circles(tupl):
    tv, cfs, pfn, res = tupl
    return (circles_anim(tv, cfs, res) + pfn)