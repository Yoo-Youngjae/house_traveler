import json
import csv
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
    # if res_poly have seperate space, so res_poly is multipolygon,
    # choose biggest polygon
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
    res,dist = polylabel(ext, with_distance=True)

    return res, dist

def drawPolyLabel(id, RobotHeight=0.75, CarpetHeight=0.15):
    """
       Attributes
       ----------
       id : House id

    """
    resList = []
    with open('data/'+id+'/house.json') as json_file:
        json_data = json.load(json_file)
        first_floor = json_data['levels'][0]

        for node_idx, node in enumerate(first_floor['nodes']):
            if node['type'] == 'Room':
                try:
                    try:
                        obj_list = node['nodeIndices']
                    except:
                        obj_list = []

                    for obj_idx in obj_list:
                        obj = first_floor['nodes'][obj_idx]
                        if obj['bbox']['min'][1] > RobotHeight or obj['bbox']['max'][1] < CarpetHeight:  # not colide
                            continue

                    res, dist = drawPolygon(first_floor['nodes'], node, obj_list, RobotHeight, CarpetHeight)
                    resList.append([res, dist, node['roomTypes']])
                except Exception as e:
                    print('here is in drawPolyLabel',e)
    return resList


def parse_walls(objFile, lower_bound = 1.0):
    def create_box(vers):
        if len(vers) == 0:
            return None
        v_max = [-1e20, -1e20, -1e20]
        v_min = [1e20, 1e20, 1e20]
        for v in vers:
            for i in range(3):
                if v[i] < v_min[i]: v_min[i] = v[i]
                if v[i] > v_max[i]: v_max[i] = v[i]
        obj = {}
        obj['bbox'] = {}
        obj['bbox']['min']=v_min
        obj['bbox']['max']=v_max
        if v_min[1] < lower_bound:
            return obj
        return None
    walls = []
    try:
        with open(objFile, 'r') as file:
            vers = []
            for line in file.readlines():
                if len(line) < 2: continue
                if line[0] == 'g':
                    if (vers is not None) and (len(vers) > 0): walls.append(create_box(vers))
                    if ('Wall' in line):
                        vers = []
                    else:
                        vers = None
                if (vers is not None) and (line[0] == 'v') and (line[1] == ' '):
                    vals = line[2:]
                    coor =[float(v) for v in vals.split(' ') if len(v)>0]
                    if len(coor) != 3:
                        print('line = {}'.format(line))
                        print('coor = {}'.format(coor))
                        assert(False)
                    vers.append(coor)
            if (vers is not None) and (len(vers) > 0):
                walls.append(create_box(vers))
            ret_walls = [w for w in walls if w is not None]
            return ret_walls
    except:
        return False


def makeDoorAndColide(all_obj):
    target_match_class = 'nyuv2_40class'
    target_door_labels = ['door', 'fence', 'arch']
    door_ids = set()
    window_ids = set()
    fine_grained_class = 'fine_grained_class'
    ignored_labels = ['person', 'umbrella', 'curtain']
    person_ids = set()
    CarpetHeight = 0.15
    RobotHeight = 0.75
    with open('data/ModelCategoryMapping.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[target_match_class] in target_door_labels:
                door_ids.add(row['model_id'])
            if row[target_match_class] == 'window':
                window_ids.add(row['model_id'])
            if row[fine_grained_class] in ignored_labels:
                person_ids.add(row['model_id'])
    def is_door(obj):
        if obj['modelId'] in door_ids:
            return True
        if (obj['modelId'] in window_ids) and (obj['bbox']['min'][1] < CarpetHeight):
            return True
        return False

    solid_obj = [obj for obj in all_obj if
                 (not is_door(obj)) and (obj['modelId'] not in person_ids)]  # ignore person
    door_obj = [obj for obj in all_obj if is_door(obj)]
    colide_obj = [obj for obj in solid_obj if
                  obj['bbox']['min'][1] < RobotHeight and obj['bbox']['max'][1] > CarpetHeight]
    return door_obj, colide_obj


def makeHouseMap(id):
    RobotHeight = 0.75
    file = 'data/'+id+'/house.json'
    fileObj = 'data/'+id+'/house.obj'

    with open(file) as json_file:
        json_data = json.load(json_file)

        # 1. make walkable house map
        map = [[0 for i in range(1000)] for j in range(1000)]
        min = [100, 100]
        max = [10, 10]

        # 1-1. fill 1 to room space
        for node in json_data['levels'][0]['nodes']:
            if node['type'] == 'Room':
                try:
                    if node['bbox']['min'][0] < min[0]:
                        min[0] = node['bbox']['min'][0]
                    if node['bbox']['min'][2] < min[1]:
                        min[1] = node['bbox']['min'][2]
                    if node['bbox']['max'][0] > max[0]:
                        max[0] = node['bbox']['max'][0]
                    if node['bbox']['max'][2] > max[1]:
                        max[1] = node['bbox']['max'][2]
                    for i in range(int(node['bbox']['min'][0] * 10), int(node['bbox']['max'][0] * 10)):
                        for j in range(int(node['bbox']['min'][2] * 10), int(node['bbox']['max'][2] * 10)):
                            map[j][i] = 1
                except Exception as e:
                    print(e)

        all_walls = parse_walls(fileObj, RobotHeight)
        if all_walls == False:
            return False, False, False
        for wall in all_walls:
            try:
                for i in range(int(wall['bbox']['min'][0] * 10), int(wall['bbox']['max'][0] * 10)):
                    for j in range(int(wall['bbox']['min'][2] * 10), int(wall['bbox']['max'][2] * 10)):
                        map[j][i] = 0
            except:
                print('walls error')
                pass
        all_obj = [node for node in json_data['levels'][0]['nodes'] if node['type'].lower() == 'object']

        door_obj, colide_obj = makeDoorAndColide(all_obj)
        # 1-2. fill 0 to express colide object
        for obj in colide_obj:
            try:
                for i in range(int(obj['bbox']['min'][0] * 10), int(obj['bbox']['max'][0] * 10)):
                    for j in range(int(obj['bbox']['min'][2] * 10), int(obj['bbox']['max'][2] * 10)):
                        map[j][i] = 0
            except:
                print('colide obj error')
                pass

        # 1.3 fill 2 to door object
        for obj in door_obj:
            try:
                for i in range(int(obj['bbox']['min'][0] * 10), int(obj['bbox']['max'][0] * 10)):
                    for j in range(int(obj['bbox']['min'][2] * 10), int(obj['bbox']['max'][2] * 10)):
                        map[j][i] = 2
            except:
                print('door error')
                pass

    return np.array(map), min, max

def make2dMap(id):
    RobotHeight = 0.75

    file = 'data/'+id+'/house.json'
    fileObj = 'data/'+id+'/house.obj'

    with open(file) as json_file:
        json_data = json.load(json_file)
        first_floor = json_data['levels'][0]

        # 1. make walkable house map
        map = [[(255,255,255) for i in range(1000)] for j in range(1000)] # white
        min = [100, 100]
        max = [10, 10]

        # 1-1. fill 1 to room space
        for node in json_data['levels'][0]['nodes']:
            if node['type'] == 'Room':
                try:
                    if node['bbox']['min'][0] < min[0]:
                        min[0] = node['bbox']['min'][0]
                    if node['bbox']['min'][2] < min[1]:
                        min[1] = node['bbox']['min'][2]
                    if node['bbox']['max'][0] > max[0]:
                        max[0] = node['bbox']['max'][0]
                    if node['bbox']['max'][2] > max[1]:
                        max[1] = node['bbox']['max'][2]
                    for i in range(int(node['bbox']['min'][0] * 10), int(node['bbox']['max'][0] * 10)):
                        for j in range(int(node['bbox']['min'][2] * 10), int(node['bbox']['max'][2] * 10)):
                            map[j][i] = (255, 255, 255) #  light gray (211, 211, 211)
                except Exception as e:
                    print(e)

        all_walls = parse_walls(fileObj, RobotHeight)
        if all_walls == False:
            return False, False, False
        for wall in all_walls:
            try:
                for i in range(int(wall['bbox']['min'][0] * 10), int(wall['bbox']['max'][0] * 10)):
                    for j in range(int(wall['bbox']['min'][2] * 10), int(wall['bbox']['max'][2] * 10)):
                        map[j][i] = (0,0,0) #black
            except:
                pass
        all_obj = [node for node in json_data['levels'][0]['nodes'] if node['type'].lower() == 'object']

        door_obj, colide_obj = makeDoorAndColide(all_obj)
        # 1-2. fill 0 to express colide object
        for obj in colide_obj:
            try:
                for i in range(int(obj['bbox']['min'][0] * 10), int(obj['bbox']['max'][0] * 10)):
                    for j in range(int(obj['bbox']['min'][2] * 10), int(obj['bbox']['max'][2] * 10)):
                        map[j][i] = (105, 105, 105) # dark gray
            except:
                pass

        # 1.3 fill 1 to door object
        for obj in door_obj:
            try:
                for i in range(int(obj['bbox']['min'][0] * 10), int(obj['bbox']['max'][0] * 10)):
                    for j in range(int(obj['bbox']['min'][2] * 10), int(obj['bbox']['max'][2] * 10)):
                        map[j][i] = (170, 170, 170)
            except:
                pass

        data = np.array(map)
    return data

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

    parser.add_argument('--with_suncg', type=bool, default=False)
    parser.add_argument('--house_id', type=str, default='2986fc10adbc33689803254b3541faff') # just for example
    parser.add_argument('--show_fig', type=bool, default=True)  # just for example
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_config()
    house_id = args.house_id
    # 1. make map
    if args.with_suncg:
        house_map, min_in_map, max_in_map = makeHouseMap(house_id)
        topview_map = make2dMap(house_id)
        topview_map = topview_map[int(min_in_map[1] * 10): int(max_in_map[1] * 10),
                      int(min_in_map[0] * 10): int(max_in_map[0] * 10)]
    else:
        print('It is pre-calculated for house', house_id)
        min_in_map = [37.31249906600063, 31.652499192511492]
        max_in_map = [57.574998713098466, 43.08999915405094]
        house_map = np.load('data/house_map.npy')
        topview_map = np.load('data/topview_map.npy')


    # 2. find start point in house
    if args.with_suncg:
        start_point_list = drawPolyLabel(id=house_id, RobotHeight=0.75, CarpetHeight=0.15)
    else:
        start_point_list = [[[53.87932367783033, 37.78257584739472], 1.1871773392811877, ['Bathroom']],
                            [[53.77999984792331, 33.805028426797925], 1.3949707864207639, ['Kitchen']],
                            [[48.28308190028118, 34.83750711214133], 2.4202552102860917, []],
                            [[38.77868314455699, 38.69144691838187], 0.9764477613778126, ['Garage']],
                            [[48.059683137598746, 38.990310133869734], 1.5953109697131183, ['Lobby']]]


    # 3. delete narrow point (padding < 0.4 m)
    padding_radius = 0.4
    try:
        for idx, (point, dist, roomtype) in enumerate(start_point_list):
            if dist < padding_radius:
                del start_point_list[idx]
    except:
        pass

    visited = [[False for i in range(1000)] for j in range(1000)]
    plt.imshow(topview_map)

    # 4. find the path for each start point
    for start_point, dist, roomtype in start_point_list:
        _path, visited = make_right_travel_with_smooth(house_map, start_point,min_in_map, max_in_map,
                                                       visited, topview_map, threshold=180,
                                                       show_fig=args.show_fig, show_fig_detail=False, step_length=180)  # threshold=180 : 1 path = 180m
        print(_path)
    plt.axis('off')
    plt.show()


