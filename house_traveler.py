import json
import numpy as np
import matplotlib.pyplot as plt
from module.LQR import LQRmain
import random
from polylabel import polylabel
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import geopandas as gpd

def drawPolygon(nodes, node, obj_list, RobotHeight, CarpetHeight):
    polygons = []
    for obj_idx in obj_list:
        obj = nodes[obj_idx]
        # 1. not colide
        if obj['bbox']['min'][1] > RobotHeight or obj['bbox']['max'][1] < CarpetHeight:
            continue
        else:
            polygons.append(Polygon([(obj['bbox']['min'][0], obj['bbox']['min'][2]),
                                    (obj['bbox']['min'][0], obj['bbox']['max'][2]),
                                    (obj['bbox']['max'][0], obj['bbox']['max'][2]),
                                    (obj['bbox']['max'][0], obj['bbox']['min'][2])]))
    #2. union the all object polygons
    u = cascaded_union(polygons)


    room_line = Polygon([(node['bbox']['min'][0], node['bbox']['min'][2]),
                        (node['bbox']['min'][0], node['bbox']['max'][2]),
                        (node['bbox']['max'][0], node['bbox']['max'][2]),
                        (node['bbox']['max'][0], node['bbox']['min'][2])])
    #3. make room - objects
    boundary_room = gpd.GeoSeries(room_line).__sub__(gpd.GeoSeries(u))
    res_poly = boundary_room.unary_union

    #4. calculate polylabel
    extForPrint = []
    ext = []
    # if res_poly have seperate space, so res_poly is multipolygon, choose biggest polygon
    if res_poly.geom_type == 'MultiPolygon':
        max_idx = 0
        max_size = -1
        for idx, p in enumerate(res_poly):
            if p.area > max_size:
                max_size = p.area
                max_idx = idx
        res_poly = res_poly[max_idx]
    for i in res_poly.exterior.coords:
        extForPrint.append((i[0], i[1]))
        ext.append([i[0], i[1]])

    interForPrint = []
    inter = []

    for i in res_poly.interiors:
        interInterForPrint = []
        interInter = []
        for j in i.coords:
            interInterForPrint.append((j[0], j[1]))
            interInter.append([j[0], j[1]])
        interForPrint.append(interInterForPrint)
        inter.append(interInter)


    ext = [ext]
    for i in inter:
        ext.append(i)
    res, dist = polylabel(ext, with_distance=True) # find the Pole of Inaccessibiliy(PIA)

    return res, dist

def drawPolyLabel(file, RobotHeight=0.75, CarpetHeight=0.15):
    resList = []
    with open(file) as json_file:
        json_data = json.load(json_file)

        for node_idx, node in enumerate(json_data):
            if node['type'] == 'Room':
                try:
                    try:
                        obj_list = node['nodeIndices']
                    except:
                        obj_list = []

                    for obj_idx in obj_list:
                        obj = json_data[obj_idx]
                        if obj['bbox']['min'][1] > RobotHeight or obj['bbox']['max'][1] < CarpetHeight:  # not colide
                            continue

                    res, dist = drawPolygon(json_data, node, obj_list, RobotHeight, CarpetHeight)
                    resList.append([res, dist, node['roomTypes']])
                except Exception as e:
                    print('here is in drawPolyLabel',e)
    return resList


def make_right_travel_with_smooth(np_map, start_point, min, max,
                                  visited, topview_map,
                                  threshold=180, show_fig=False, show_fig_detail=False, step_length=180):
    neighbors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    full_path = []
    yaw = 0

    # calculate proper parameters
    startX, startY = int(start_point[0]*10), int(start_point[1]*10)
    maxX, maxY = int(max[0] * 10), int(max[1] * 10)
    minX, minY = int(min[0] * 10), int(min[1] * 10)

    padding = 4 # (padding < 0.4 m)

    if np_map[startY][startX] == 0: # start point is not walkable
        return full_path, visited


    is_break = False
    currentX, currentY = startX, startY
    is_First = True
    total_count = 0

    while not is_break:
        total_count +=1
        count = 0
        path = []
        if is_First:
            is_First = False
            is_front = False
            while is_front is False:
                if count >= threshold:
                    break
                if currentX > maxX or currentY > maxY or currentX < minX or currentY < minY:
                    is_break = True
                    break
                if visited[currentY][currentX]:
                    is_break = True
                    break
                # current process
                path.append((currentX, currentY))
                count += 1
                visited[currentY][currentX] = True

                # next process
                currentX, currentY = currentX + neighbors[yaw][0], currentY + neighbors[yaw][1]
                checkFrontX, checkFrontY = currentX + neighbors[yaw][0] * padding, \
                                           currentY + neighbors[yaw][1] * padding
                ####################################

                # if turn left == face wall
                right_yaw = ((yaw + 1)) % 4
                for x in range(-padding, padding + 1):
                    check_padding_FrontX, check_padding_FrontY = checkFrontX + neighbors[right_yaw][0] * x,\
                                                                 checkFrontY + neighbors[right_yaw][1] * x
                    if np_map[check_padding_FrontY][check_padding_FrontX] == 0:  # if wall is in front?
                        yaw = (yaw - 1 + 4) % 4  # turn left
                        is_front = True
                        break
        while len(path) < threshold:
            if currentX > maxX or currentY> maxY or currentX < minX or currentY <  minY:
                is_break = True
                break
            if visited[currentY][currentX]:
                is_break = True
                break
            # current process
            path.append((currentX, currentY))
            count += 1
            visited[currentY][currentX] = True

            # next process
            currentX, currentY = currentX + neighbors[yaw][0], currentY + neighbors[yaw][1]

            checkFrontX, checkFrontY = currentX + neighbors[yaw][0] * padding, \
                                       currentY + neighbors[yaw][1] * padding

            ####################################
            # if turn left == face wall
            is_front = False
            right_yaw = (yaw + 1) % 4
            for x in range(-padding-1, padding):
                check_padding_FrontX, check_padding_FrontY = checkFrontX+ neighbors[right_yaw][0] * x, \
                                                             checkFrontY + neighbors[right_yaw][1] * x

                if np_map[check_padding_FrontY][check_padding_FrontX] == 0: # check is wall in front
                    yaw = (yaw - 1 + 4) % 4 # turn left
                    is_front = True
                    break
            ####################################

            ####################################
            # if turn right == have no wall in around
            if not is_front:
                check_rightX, check_rightY = currentX + neighbors[right_yaw][0] * padding, \
                                             currentY + neighbors[right_yaw][1] * padding
                have_wall = False
                for x in range(-padding+1, padding):
                    check_padding_rightX, check_padding_rightY = check_rightX + neighbors[yaw][0] * x, \
                                                                 check_rightY + neighbors[yaw][1] * x

                    if np_map[check_padding_rightY][check_padding_rightX] == 0:  # check is wall in right
                        have_wall = True
                # have no wall -> turn right
                if have_wall is False:
                    yaw = (yaw + 1) % 4  # turn right
            ####################################

        # one path end process
        if not is_break:
            if len(path) >10:
                if show_fig and show_fig_detail:
                    plt.imshow(topview_map)
                    _x, _y = [], []
                    for idx, (x,y) in enumerate(path):
                        if idx % 2 == 0:  # even
                            _x.append(x - min[0] * 10)
                            _y.append(y - min[1] * 10)
                    plt.scatter(_x, _y, s=2,color='#4477aa')
                    plt.axis('off')
                    plt.show()

                gox, goy = [], []
                backx, backy = [], []
                for idx, (x, y) in enumerate(path):
                    if idx % 3 == 0:  # even
                        if idx % 9 == 0:
                            gox.append(x + random.gauss(0.0, 0.2))
                            goy.append(y + random.gauss(0.0, 0.2))
                        else:
                            gox.append(x)
                            goy.append(y)
                for idx, (x, y) in enumerate(reversed(path)):
                    if idx % 3 == 0:  # back even
                        if idx % 9 == 0:
                            backx.append(x + random.gauss(0.0, 0.2))
                            backy.append(y + random.gauss(0.0, 0.2))
                        else:
                            backx.append(x)
                            backy.append(y)

                if show_fig and show_fig_detail:
                    plt.imshow(topview_map)
                    _x, _y = [], []
                    for (x,y) in zip(gox,goy):
                        _x.append(x-min[0]* 10)
                        _y.append(y-min[1]* 10)
                    plt.scatter(_x, _y, s=2,color='#4477aa')
                    plt.axis('off')
                    plt.show()


                res_go = LQRmain(gox, goy, min,0, topview_map, show_fig=show_fig,
                                 show_fig_detail=show_fig_detail)
                res_back = LQRmain(backx, backy, min,1, topview_map)


                if res_go == False or res_back == False:
                    continue

                origin_go_path_len = len(res_go)
                origin_back_path_len = len(res_back)+2

                step_size_go = origin_go_path_len // step_length
                step_size_back = origin_back_path_len // step_length

                sampled_path = []
                res_go.reverse()
                go_final = []

                for i in range(step_length):
                    sampled_path.append(res_go[step_size_go * i])
                    go_final.append(res_go[step_size_go * i])

                sampled_path.reverse()
                back_final = []
                for i in range(step_length+2):
                    sampled_path.append(res_back[step_size_back * i])
                    back_final.append(res_back[step_size_back * i])

                full_path.append(sampled_path)
                if show_fig:
                    plt.imshow(topview_map)
                    _x, _y = [], []
                    for x,y in go_final:
                        _x.append(x-min[0]* 10)
                        _y.append(y-min[1]* 10)
                    plt.scatter(_x, _y, s=2,color='#4477aa')
                    plt.axis('off')
                    plt.show()
        else:
            for (i,j) in path:
                visited[j][i] = True

    return full_path, visited

def get_config():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--json_file_name', type=str, default='sample_house')  # example file
    parser.add_argument('--show_fig', type=bool, default=True)  # show map figure for each generation procedure
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_config()
    json_file = 'data/'+args.json_file_name+'.json'

    # 1. make map
    print('It is pre-calculated map for sample house')

    house_map = np.load('data/house_map.npy')
    topview_map = np.load('data/topview_map.npy')
    min_in_map = [37.31249906600063, 31.652499192511492]
    max_in_map = [57.574998713098466, 43.08999915405094]
    plt.imshow(topview_map)
    plt.axis('off')
    plt.show()

    # 2. find PIAs which is a point where the largest circle can be drawn in the polygon
    pia_list = drawPolyLabel(json_file, RobotHeight=0.75, CarpetHeight=0.15)

    # 3. extract PIAs where the circle radius is greater than 40 cm inside each room
    padding_radius = 0.4
    try:
        for idx, (point, dist, roomtype) in enumerate(pia_list):
            if dist < padding_radius:
                del pia_list[idx]
    except:
        pass

    visited = [[False for i in range(1000)] for j in range(1000)]
    plt.imshow(topview_map)

    # 4. generate the trajectory for each start point
    for pia, dist, roomtype in pia_list:
        _path, visited = make_right_travel_with_smooth(house_map, pia, min_in_map, max_in_map,
                                                       visited, topview_map, threshold=180,
                                                       show_fig=args.show_fig, show_fig_detail=False,
                                                       step_length=180)  # threshold=180 : 1 path = 180m
        print(_path)
    plt.axis('off')
    plt.show()


