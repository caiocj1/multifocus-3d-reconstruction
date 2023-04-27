"""
All credits for this file go to Yamaguchi
"""

import numpy as np
import math
import torch

def default_transmittance(flag, img):
    """
    default_transmittance
    flag:0 Set based on the luminance value of the teacher data
        :1 Initial value of transmittance = 0.75
    img:size[nv]
    :return: transmittance
    """
    if flag == 0:
        a = torch.pow(img, 1/img.size()[0])
        a = a.to(torch.float32)

    elif flag == 1:
        a = torch.ones_like(img)
        a = a * 0.75
        a = a.to(torch.float32)

    return a


def set_diameter(NA, layer_num, depth_resolution):
    ratio = NA/math.sqrt(1-math.pow(NA, 2))
    diameter = ratio * (2*layer_num-1) * depth_resolution
    return diameter


def set_incidental_light(diameter, num_div):
    num_div = int(num_div)
    radius = diameter/2
    center = diameter/2
    offset = diameter/num_div

    a = np.zeros((num_div, num_div))
    ray_check = []
    Nr = 0

    for i in range(num_div):
        for j in range(num_div):
            x = pow(offset * (0.5 + i) - center, 2)
            y = pow(offset * (0.5 + j) - center, 2)
            if(x+y) <= pow(radius, 2):
                a[i][j] = Nr
                ray_check.append(Nr)
                Nr += 1
            else:
                a[i][j] = -1
                ray_check.append(-1)

    return Nr, ray_check


def set_areasize(diameter):
    if math.ceil(diameter)%2==0:
        areasize = math.ceil(diameter)+1
    else:
        areasize = math.ceil(diameter)
    return areasize


def boundary_decision(pre, now):
    """
    boundary_decision
    Boundary determination in the xy-plane
    return: Bool value
    """
    judge = (math.floor(pre) != math.floor(now)) and (now != math.floor(now)) and (pre != math.floor(pre))
    return judge


def range_matrix_generation(ray_number, ray_check, layer_num, diameter, apa_size, resolution, depth_resolution):
    """
    range_matrix_generation
    """
    voxel_depth_num = 2 * layer_num - 1
    depth = voxel_depth_num * depth_resolution

    areasize = diameter / resolution
    voxel_side_num = set_areasize(areasize)

    center = voxel_side_num * resolution / 2
    offset = diameter / apa_size

    p = [0]
    sign_x = sign_y = 0
    check_line_x = check_line_y = 0
    gap = (voxel_side_num * resolution - diameter) / 2

    light_path = np.zeros((ray_number, voxel_depth_num, voxel_side_num, voxel_side_num))

    for n in range(0, math.ceil(apa_size / 2), 1):
        # print('n =', n)
        for m in range(0, math.ceil(apa_size / 2), 1):
            # print('m =', m)
            count = 0
            vox_iter2 = -1

            if ray_check[n * apa_size + m] != -1:
                # print('ray =', ray_check[n * apa_size + m])
                ray_iter = ray_check[n * apa_size + m]
                ray_sym1 = ray_check[n * apa_size + (apa_size - 1 - m)]
                ray_sym2 = ray_check[(apa_size - 1 - n) * apa_size + m]
                ray_sym3 = ray_check[(apa_size - 1 - n) * apa_size + (apa_size - 1 - m)]
                # print(ray_iter, ray_sym1, ray_sym2, ray_sym3)
                for z_iter in range(0, layer_num + 1, 1):
                    z1 = depth_resolution * (2 * layer_num - 1 - z_iter)
                    k1 = 2 * z1 / depth - 1
                    y1 = center + k1 * ((n + 0.5) * offset + gap - center)
                    x1 = center + k1 * ((m + 0.5) * offset + gap - center)
                    # print("(x1,y1,z1) =", x1, y1, z1)

                    check = 0
                    x2 = y2 = x3 = y3 = 0
                    z2 = -100
                    z3 = -200

                    if count == 0:
                        p[0] = x1
                        p.append(y1)
                        p.append(z1)
                        count += 1

                    elif count > 0:
                        pre_x = p[(count - 1) * 3]
                        pre_y = p[(count - 1) * 3 + 1]
                        pre_z = p[(count - 1) * 3 + 2]
                        i = j = 0
                        nx = ny = 0
                        if boundary_decision(pre_x / resolution, x1 / resolution):
                            nx = math.floor(x1 / resolution) - math.floor(pre_x / resolution)
                            sign_x = resolution
                            check_line_x = resolution
                            check = 1
                        if boundary_decision(pre_y / resolution, y1 / resolution):
                            ny = math.floor(y1 / resolution) - math.floor(pre_y / resolution)
                            sign_y = resolution
                            check_line_y = resolution
                            if check != 1:
                                check = 2
                            else:
                                check = 3

                        while (i < nx) or (j < ny):
                            if ((check == 1) or (check == 3)) and (nx > 0):
                                x2 = math.floor(pre_x / resolution) * resolution + sign_x * i + check_line_x
                                if (m + 0.5) * offset + gap - center != 0:
                                    k2 = (x2 - center) / ((m + 0.5) * offset + gap - center)
                                else:
                                    k2 = 0
                                y2 = center + k2 * ((n + 0.5) * offset + gap - center)
                                z2 = (k2 + 1) * depth / 2

                            if ((check == 2) or (check == 3)) and (ny > 0):
                                y3 = math.floor(pre_y / resolution) * resolution + sign_y * j + check_line_y
                                if (n + 0.5) * offset + gap - center != 0:
                                    k3 = (y3 - center) / ((n + 0.5) * offset + gap - center)
                                else:
                                    k3 = 0
                                x3 = center + k3 * ((m + 0.5) * offset + gap - center)
                                z3 = (k3 + 1) * depth / 2

                            if z2 > z3:
                                if (z2 <= depth) and (z2 >= 0) and (z2 != pre_z):
                                    p.append(x2)
                                    p.append(y2)
                                    p.append(z2)
                                    count += 1
                                i += 1
                                check = 1

                            elif z2 < z3:
                                if (z3 <= depth) and (z3 >= 0) and (z3 != pre_z):
                                    p.append(x3)
                                    p.append(y3)
                                    p.append(z3)
                                    count += 1
                                j += 1
                                check = 2

                            else:
                                if (z2 <= depth) and (z2 >= 0) and (z2 != pre_z) and (z3 <= depth) and (z3 >= 0) and (
                                        z3 != pre_z):
                                    p.append(x2)
                                    p.append(y2)
                                    p.append(z2)
                                    count += 1
                                i += 1
                                j += 1
                                check = 3

                        p.append(x1)
                        p.append(y1)
                        p.append(z1)
                        count += 1
                        for k in range(0, count - 1, 1):
                            X1 = p[k * 3]
                            Y1 = p[k * 3 + 1]
                            Z1 = p[k * 3 + 2]
                            X2 = p[(k + 1) * 3]
                            Y2 = p[(k + 1) * 3 + 1]
                            Z2 = p[(k + 1) * 3 + 2]

                            d = math.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2 + (Z1 - Z2) ** 2)
                            # print('z_iter', z_iter, 'distance', d)

                            z_tmp = Z1

                            z_point = z_iter - 1
                            z_point_sym = voxel_depth_num - z_point - 1
                            y_point = math.floor(Y1 / resolution)
                            y_point_sym = voxel_side_num - 1 - math.floor(Y1 / resolution)
                            x_point = math.floor(X1 / resolution)
                            x_point_sym = voxel_side_num - 1 - math.floor(X1 / resolution)

                            vox_iter = z_point * (voxel_side_num ** 2) + y_point * voxel_side_num + x_point

                            if 0 <= math.floor(X1 / resolution) < voxel_side_num and 0 <= math.floor(
                                    Y1 / resolution) < voxel_side_num:
                                if z_tmp > 0 and d != 0:
                                    if vox_iter != vox_iter2:
                                        light_path[ray_iter, z_point, y_point, x_point] = d
                                        light_path[ray_sym1, z_point, y_point, x_point_sym] = d
                                        light_path[ray_sym2, z_point, y_point_sym, x_point] = d
                                        light_path[ray_sym3, z_point, y_point_sym, x_point_sym] = d
                                        if z_iter < layer_num + 1:
                                            light_path[ray_sym3, z_point_sym, y_point, x_point] = d
                                            light_path[ray_sym2, z_point_sym, y_point, x_point_sym] = d
                                            light_path[ray_sym1, z_point_sym, y_point_sym, x_point] = d
                                            light_path[ray_iter, z_point_sym, y_point_sym, x_point_sym] = d
                                    vox_iter2 = vox_iter

                        p[0] = p[(count - 1) * 3]
                        p[1] = p[(count - 1) * 3 + 1]
                        p[2] = p[(count - 1) * 3 + 2]
                        del p[2:-1]
                        count = 1
                del p[0:-1]
    return light_path