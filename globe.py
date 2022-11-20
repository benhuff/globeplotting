import numpy as np
import pandas as pd
import itertools
import plotly.graph_objs as go

def d2r(degree):
    return degree*np.pi/180
  
def coords_to_sphere(lon, lat, radius=1):
    lon = np.array(lon, dtype=np.float64)
    lat = np.array(lat, dtype=np.float64)
    lon = d2r(lon)
    lat = d2r(lat)
    xs = radius*np.cos(lon)*np.cos(lat)
    ys = radius*np.sin(lon)*np.cos(lat)
    zs = radius*np.sin(lat)
    return xs, ys, zs

def base_sphere(size=1, clr='#00001f', dist=0): 
    theta = np.linspace(0,2*np.pi,100)
    phi = np.linspace(0,np.pi,100)

    x0 = dist + size * np.outer(np.cos(theta),np.sin(phi))
    y0 = size * np.outer(np.sin(theta),np.sin(phi))
    z0 = size * np.outer(np.ones(100),np.cos(phi))

    trace= go.Surface(x=x0, y=y0, z=z0, colorscale=[[0,clr], [1,clr]])
    trace.update(showscale=False)

    return trace

def base_layout(title=None, width=800, height=800):
    noaxis = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks='',
        title='',
        zeroline=False
    )

    titlecolor = 'white'
    bgcolor = 'black'

    layout = go.Layout(
        autosize=False, 
        width=width, 
        height=height,
        title=title,
        titlefont=dict(
            family='Courier New', 
            color=titlecolor
        ),
        showlegend=False,
        scene=dict(
            xaxis=noaxis,
            yaxis=noaxis,
            zaxis=noaxis
        ),
        paper_bgcolor=bgcolor,
        plot_bgcolor=bgcolor
    )

    return layout

def add_height(row, i:str, multiplier=.002):
    multi = []
    if i == 'z':
        multi.append(row.z[0])
        for c, v in enumerate(range(int(row.hex_count/6))):  # change to any int column
            ci = c+1
            ratio = 1+ci*multiplier
            multi.append(row.z[0]*ratio)
        return multi
    elif i == 'x':
        multi.append(row.x[0])
        for c, v in enumerate(range(int(row.hex_count/6))):
            ci = c+1
            ratio = 1+ci*multiplier
            multi.append(row.x[0]*ratio)
        return multi
    else: 
        multi.append(row.y[0])
        for c, v in enumerate(range(int(row.hex_count/6))):
            ci = c+1
            ratio = 1+ci*multiplier
            multi.append(row.y[0]*ratio)
        return multi

def point_to_sphere(point):

    x, y = np.array(point.xy, dtype=np.float64)
    xi, yi, zi = coords_to_sphere(x, y, 1.01)

    return xi, yi, zi

def point_dataframe_to_sphere(dataframe):

    dataframe['xyz'] = dataframe.geometry.apply(point_to_sphere)
    dataframe[['x','y','z']] = pd.DataFrame(dataframe["xyz"].to_list(), index=dataframe.index)
    
    dataframe['x'] = dataframe.apply(add_height, args=['x'], axis=1)
    dataframe['y'] = dataframe.apply(add_height, args=['y'], axis=1)
    dataframe['z'] = dataframe.apply(add_height, args=['z'], axis=1)
    
    x = list(itertools.chain.from_iterable(dataframe.x.to_list()))
    y = list(itertools.chain.from_iterable(dataframe.y.to_list()))
    z = list(itertools.chain.from_iterable(dataframe.z.to_list())) 
    
    points = dict(
        type='scatter3d',
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(color='slateblue', size=1),
        opacity=.4
    )
    return points

def polygon_to_sphere(polygon):

    x, y = np.array(polygon.exterior.coords.xy, dtype=np.float64)
    xi, yi, zi = coords_to_sphere(x, y, 1.01)

    return xi, yi, zi

def polygon_dataframe_to_sphere(dataframe):

    dataframe['xyz'] = dataframe.geometry.apply(polygon_to_sphere)
    dataframe[['x','y','z']] = pd.DataFrame(dataframe["xyz"].to_list(), index=dataframe.index)
    
    xm = dataframe.x.to_list()
    ym = dataframe.y.to_list()
    zm = dataframe.z.to_list()
    
    polygon_dicts = []

    for i in range(len(xm)):
        dicth = dict(
            type='scatter3d',
            x=xm[i],
            y=ym[i],
            z=zm[i],
            mode='lines',
            line=dict(color='grey', width=2)
        )
        data = go.Scatter3d(dicth)
        polygon_dicts.append(data)

    return polygon_dicts