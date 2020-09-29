from anim import *
from pool_funcs import *
print(coefs_svg_map((1, svg_points)))
pfunc_svg = parametric_plot(
    (fourier_approx_svg(t).real(), fourier_approx_svg(t).imag()), 
    (t,0,1), 
    thickness=3
)
frame = send_to_circles((.1, coefs_80_sorted_svg, pfunc_svg, resolution))
frame.save("frame.png")