import numpy as np
from scipy.ndimage import maximum_filter
from VoxelData import VoxelData
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def draw_map_with_pts(scene_id, nav_map, points, search_space=None, downsample_scale = 4, dump_format="{scene_id}-astar.html"):
    if search_space is not None:
        for (i,j,k) in search_space:
            nav_map[i][j][k] = 3
    nav_map = maximum_filter(nav_map, size=downsample_scale, mode='constant')
    nav_map = nav_map[::downsample_scale, ::downsample_scale, ::downsample_scale]
    # Visualize
    color_palette = np.array( # 0 empty, 1 navigable, 2 is obstacle, 3 is search space
        [[1., 1., 1., 0.], [0.5, 0.5, 0.5, 1.], [0.8, 0.8, 0.8, 0.1], [0.95,0.82,0.42,1.]],  dtype=np.float32
    )
    facecolors = color_palette[nav_map.astype(int)]

    # Voxels = VoxelData(nav_map!=0, facecolors) # show obstacle
    Voxels = VoxelData(nav_map>0, facecolors)  # show only navigable area

    color = np.array(Voxels.colors)
    color[:, :3] *= 255
    color = [f"rgba({int(rgba[0])}, {int(rgba[1])}, {int(rgba[2])}, {1})" for rgba in color]
    data = []
    data.append(
        go.Mesh3d(
            x=Voxels.vertices[0],
            y=Voxels.vertices[1],
            z=Voxels.vertices[2],
            i=Voxels.triangles[0],
            j=Voxels.triangles[1],
            k=Voxels.triangles[2],
            facecolor=color,
            opacity=0.8,
        )
    )

    nodes_plot = go.Scatter3d(
        x=points[:, 0]//downsample_scale, 
        y=points[:, 1]//downsample_scale, 
        z=points[:, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color='#3d6ec9',
            #color=z,                # set color to an array/list of desired values
            #colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(nodes_plot)

    start_pt = go.Scatter3d(
        x=points[:1, 0]//downsample_scale, 
        y=points[:1, 1]//downsample_scale, 
        z=points[:1, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color='#0dab62',
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(start_pt)

    end_pt = go.Scatter3d(
        x=points[-1:, 0]//downsample_scale, 
        y=points[-1:, 1]//downsample_scale, 
        z=points[-1:, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color="#e04a3f",
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(end_pt)

    layout = go.Layout(
                scene=dict(
                    aspectmode='data'
            )) 

    fig = go.Figure(data=data,layout=layout)
    fig.write_html(dump_format.format(scene_id=scene_id))

def draw_heat_map(scene_id, heat_map, points, downsample_scale = 4, dump_format="{scene_id}-astar.html"):
    heat_map = maximum_filter(heat_map, size=downsample_scale, mode='constant')
    heat_map = heat_map[::downsample_scale, ::downsample_scale, ::downsample_scale]
    # Visualize
    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, 101))
    heat_map = (heat_map*100).astype(int)
    
    facecolors = color_palette[heat_map]

    # Voxels = VoxelData(nav_map!=0, facecolors) # show obstacle
    Voxels = VoxelData(heat_map>0, facecolors)  # show only navigable area

    color = np.array(Voxels.colors)
    color[:, :3] *= 255
    color = [f"rgba({int(rgba[0])}, {int(rgba[1])}, {int(rgba[2])}, {1})" for rgba in color]
    data = []
    data.append(
        go.Mesh3d(
            x=Voxels.vertices[0],
            y=Voxels.vertices[1],
            z=Voxels.vertices[2],
            i=Voxels.triangles[0],
            j=Voxels.triangles[1],
            k=Voxels.triangles[2],
            facecolor=color,
            opacity=0.8,
        )
    )

    nodes_plot = go.Scatter3d(
        x=points[:, 0]//downsample_scale, 
        y=points[:, 1]//downsample_scale, 
        z=points[:, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color='#3d6ec9',
            #color=z,                # set color to an array/list of desired values
            #colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(nodes_plot)

    start_pt = go.Scatter3d(
        x=points[:1, 0]//downsample_scale, 
        y=points[:1, 1]//downsample_scale, 
        z=points[:1, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color='#0dab62',
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(start_pt)

    end_pt = go.Scatter3d(
        x=points[-1:, 0]//downsample_scale, 
        y=points[-1:, 1]//downsample_scale, 
        z=points[-1:, 2]//downsample_scale,                          
        mode='markers', 
        marker=dict(
            size=10,
            color="#e04a3f",
            opacity=0.8
        ),
        showlegend=False
    )
    data.append(end_pt)

    layout = go.Layout(
                scene=dict(
                    aspectmode='data'
            )) 

    fig = go.Figure(data=data,layout=layout)
    fig.write_html(dump_format.format(scene_id=scene_id))