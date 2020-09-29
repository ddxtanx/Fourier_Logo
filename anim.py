#!/usr/bin/env sage -python
import re
from scipy.integrate import quadrature
import numpy
from math import floor
from sage.all import *
from numpy import ndarray, exp
import multiprocessing
from pool_funcs import *
try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default
set_verbose(-2)
print(cpus)
pool = multiprocessing.Pool(processes=cpus)
t = var('t')
origin = (55 + 820*I)
x_scale = 900
y_scale = 750
def split_svg(svg_text):
    return re.findall(r"[M,C].[0-9\.+]*", svg_text)
def c_bezier(sp, cp1, cp2, ep): 
    return sp*(1-t)**3 + cp1*3*(1-t)**2*t+cp2*3*(1-t)*t**2+ep*t**3
def parse_pts_to_comp_num(pt1,pt2):
    tp = (float(pt1) + I*float(pt2)) - origin
    return (tp.real()/x_scale) - I*(tp.imag()/y_scale)
def parse_svg_text_to_funcs(svgs):
    current_start = 0
    funcs = []
    points = []
    translated = ""
    for svg_command in svgs:
        start_char = svg_command[0]
        pts = (svg_command[1:]).split("+")
        translated_command = start_char + ""
        if start_char == 'M':
            pt1 = pts[0]
            pt2 = pts[1]
            current_start = parse_pts_to_comp_num(pt1,pt2)
            translated_command += f"{current_start.real()}+{current_start.imag()}"
        if start_char == 'C':
            offset = len(funcs)
            cp1 = parse_pts_to_comp_num(pts[0], pts[1])
            translated_command += f"{cp1.real()}+{cp1.imag()}"
            cp2 = parse_pts_to_comp_num(pts[2], pts[3])
            translated_command += f"+{cp2.real()}+{cp2.imag()}"
            ep = parse_pts_to_comp_num(pts[4], pts[5])
            translated_command += f"+{ep.real()}+{ep.imag()}"
            fn = c_bezier(current_start, cp1, cp2, ep)
            funcs.append(fn(t-offset))
            current_start = ep
        translated+=translated_command
    return funcs
def make_interval(i):
    if i%2 == 0:
        return [i,i+1]
    else:
        return (i,i+1)
with open("logo_svg_in_text.txt", "r") as f:
    txt = f.read()
    txt_to_list = split_svg(txt)
    svg_funcs = parse_svg_text_to_funcs(txt_to_list)
    end = len(svg_funcs)
def svg_func(t0):
    mint = floor(t0)
    mint = min(mint, end-1)
    func = svg_funcs[mint]
    return (func(t0))
def svg_func_x(t): return svg_func(t).real()
def svg_func_y(t): return svg_func(t).imag()
def integrate(func, startv, endv):
    return quadrature(func,startv,endv,vec_func=False, tol=80)

def coefs_svg(n):
    lst = [(i,svg_points) for i in range(-int(n/2), int(n/2)+1)]
    return pool.map(coefs_svg_map, lst)

def fourier_approx_svg(t):
    coef_outer = 1/resolution
    return coef_outer*sum([coef * exp(2*pi*I*i*t) for i,coef in coefs_80_svg])

resolution = 200
svg_points = [svg_func(end*t/resolution) for t in range(0,resolution)]
coefs_80_svg = coefs_svg(80)
coefs_80_sorted_svg = sorted(
    coefs_80_svg, 
    key=lambda p: 500 if p[0] == 0 else abs(p[1]), 
    reverse=True
)
if __name__ == "__main__":
    parametric_plot((svg_func_x,svg_func_y), (t,0,end), axes=True)
    
    pfunc_svg = parametric_plot(
        (fourier_approx_svg(t).real(), fourier_approx_svg(t).imag()), 
        (t,0,1), 
        thickness=3
    )
    pfunc_svg.save("pic.png")
    num_frames = 100
    times = [
        (t/num_frames, coefs_80_sorted_svg, pfunc_svg, resolution) 
        for t in range(0,num_frames)
    ]
    circ_anim = pool.map(send_to_circles, times)
    anim = animate(circ_anim, xmin=-.2, xmax=1.2, ymin=-.2, ymax=1.2, axes=False, figsize=(2.37,2.37), dpi=100)
    anim.save(filename="garrett_logo_small.gif", delay=1/60)
