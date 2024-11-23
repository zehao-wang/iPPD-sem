import numpy as np
from scipy import interpolate

def scaler(bound, target_range, value):
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]


def B_spline(points):
    x = points[:-1]
    y = points[1:]
    tck, *rest = interpolate.splprep([x, y])
    u = np.linspace(0, 1, num=10)
    smoothed = interpolate.splev(u, tck)
    return smoothed


def get_surrounding_point_rel_pos_with_radius(radius, num_split, sorted_type='ascending'):
    """
    Args:
        radius: max radius of point relative to center
        num_split: number of splits on one axis
        sorted_type: from centor to bound use 'ascending'
    """
    x = np.linspace(-radius, radius, num_split)
    y = np.linspace(-radius, radius, num_split)
    xv, yv = np.meshgrid(x, y)

    coord_pairs = list(zip(xv.flatten(), yv.flatten()))
    
    if sorted_type == 'ascending':
        out_pairs = sorted(coord_pairs, key=lambda z: z[0]**2 + z[1]**2 )
        out_pairs = coord_pairs[:len(coord_pairs)//2] \
            + [pair for pair in coord_pairs[len(coord_pairs)//2:] if pair[1]**2 + pair[0]**2 <= radius**2 ]
    else:
        raise NotImplementedError()
    return out_pairs