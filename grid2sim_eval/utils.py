import numpy as np

def get_surrounding_point_rel_pos_with_radius3d(radius, num_splits, sorted_type='ascending'):
    """
    Args:
        radius: max radius of point relative to center
        num_split: number of splits on one axis
        sorted_type: from centor to bound use 'ascending'
    """
    x = np.linspace(-radius, radius, num_splits)
    y = np.linspace(-radius, radius, num_splits)
    z = np.linspace(-2, 2, 41)
    xv, yv, zv = np.meshgrid(x, y, z)

    coord_pairs = list(zip(xv.flatten(), yv.flatten(), zv.flatten()))
    
    if sorted_type == 'ascending':
        out_pairs = sorted(coord_pairs, key=lambda z: z[0]**2 + z[1]**2 + z[2]**2 )
        out_pairs = coord_pairs[:len(coord_pairs)//2]
    else:
        raise NotImplementedError()
    return out_pairs

def process_end():
    pass