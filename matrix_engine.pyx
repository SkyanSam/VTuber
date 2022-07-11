import numpy as np
cimport numpy as np
import cv2
import copy
import time
import math
import itertools as its
import numba as nb

from joblib.numpy_pickle_utils import xrange
from skimage.draw import polygon
from scipy.ndimage import map_coordinates

cpdef create_blank(width, height, rgb_color=(0, 0, 0)):
    cdef np.ndarray image = np.zeros((height, width, 3), np.uint8)  # Create black blank image
    image[:] = tuple(reversed(rgb_color)) # Fill image with color (reversed because BGR)
    return image

cpdef round_pt(pt):
    return (round(pt[0]), round(pt[1]))

cpdef create_face_matrix_object(A, B, C):
    return (round_pt(B), np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]]), np.linalg.inv(np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]])))

cpdef to_unit_pts(face_matrix_object, pts):
    cdef new_pts = copy.deepcopy(pts)
    new_pts[0] -= face_matrix_object[0][0]
    new_pts[1] -= face_matrix_object[0][1]
    return np.matmul(face_matrix_object[2], new_pts)

cpdef from_unit_pts(tuple face_matrix_object, np.ndarray pts):
    cdef np.ndarray new_pts = np.matmul(face_matrix_object[1], pts)
    new_pts[0] += face_matrix_object[0][0]
    new_pts[1] += face_matrix_object[0][1]
    return new_pts

cpdef get_pts_in_triangle(A, B, C):
    cdef np.ndarray xx
    cdef np.ndarray yy
    xx, yy = polygon(np.array([A[0], B[0], C[0]]), np.array([A[1], B[1], C[1]]))
    return np.stack((xx, yy))

cpdef get_pts_in_faces(faces, prev_landmarks):
    cdef list faces_pts = []
    for f in faces:
        faces_pts.append(get_pts_in_triangle(prev_landmarks[f[0]], prev_landmarks[f[1]], prev_landmarks[f[2]]))
    return np.vstack(faces_pts)

cpdef draw_triangle(prev_pts, next_pts, prev_img, next_img):
    cdef list base_color = [0, 0, 0]
    cdef int prev_x
    cdef int prev_y
    cdef int next_x
    cdef int next_y
    for i in xrange(len(prev_pts[0])):
        prev_x = round(prev_pts[0][i])
        prev_y = round(prev_pts[1][i])
        next_x = next_pts[0][i]
        next_y = next_pts[1][i]
        if 0 <= prev_y < prev_img.shape[0] and 0 <= prev_x < prev_img.shape[1] and 0 <= next_y < next_img.shape[0] and 0 <= next_x < next_img.shape[1]:
            next_img[next_y][next_x] = prev_img[prev_y][prev_x]
    return next_img

cpdef create_face_matrices(faces, prev_landmarks):
    cdef list face_matrices = []
    for f in faces:
        face_matrices.append(create_face_matrix_object(prev_landmarks[f[0]], prev_landmarks[f[1]], prev_landmarks[f[2]]))
    return face_matrices

cpdef get_unit_pts_list(faces, faces_pts, prev_matrices, prev_landmarks):
    unit_pts_list = []
    for i, f in enumerate(faces):
        unit_pts_list.append(to_unit_pts(prev_matrices[i], faces_pts[i]))
    return unit_pts_list

@nb.njit()
cpdef draw_texture_2(list faces, list faces_pts, list unit_pts_list, list next_landmarks, np.ndarray prev_img, int export_width, int export_height):
    cdef tuple prev_matrix
    cdef tuple next_matrix
    cdef np.ndarray prev_pts
    cdef np.ndarray unit_pts
    cdef np.ndarray next_pts
    cdef np.ndarray prev_pts_t
    cdef np.ndarray next_pts_t
    cdef np.ndarray next_pt
    cdef np.ndarray prev_pt
    cdef int i
    cdef int p
    cdef tuple f
    cdef tuple ini_coord
    cdef np.ndarray out_coord
    cdef double tt
    mesh_x, mesh_y = np.arange(export_width), np.arange(export_height)
    #ini_coord = np.meshgrid(mesh_x, mesh_y)
    out_coord = np.array(np.meshgrid(mesh_x, mesh_y))
    #out_coord = [np.zeros(prev_img.shape, dtype=np.int32), np.zeros(prev_img.shape, dtype=np.int32)]
    cdef int len_faces = len(faces)
    for i in xrange(len_faces):
        f = faces[i]
        next_matrix = create_face_matrix_object(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        #unit_pts = to_unit_pts(prev_matrix, prev_pts)
        prev_pts = faces_pts[i]
        next_pts = from_unit_pts(next_matrix, unit_pts_list[i])
        prev_pts_t = prev_pts.T
        next_pts_t = next_pts.T.astype(int)
        for p in xrange(prev_pts_t.shape[0]):
            prev_pt = prev_pts_t[p]
            next_pt = next_pts_t[p]
            out_coord[0][next_pt[1]][next_pt[0]] = prev_pt[0]
            out_coord[1][next_pt[1]][next_pt[0]] = prev_pt[1]
    """
    next_img = np.zeros(prev_img.shape)
    print(list(out_coord))
    print(prev_img.ndim)
    next_img = map_coordinates(prev_img, out_coord, mode='nearest')
    return next_img
    """
    nrows = export_height
    ncols = export_width
    tt = time.process_time()
    r = map_coordinates(prev_img[:, :, 0], out_coord, order=1, mode='nearest')
    g = map_coordinates(prev_img[:, :, 1], out_coord, order=1, mode='nearest')
    b = map_coordinates(prev_img[:, :, 2], out_coord, order=1, mode='nearest')
    rgb = np.array([r, g, b])
    rgb = rgb.reshape(3, nrows, ncols).transpose(1, 2, 0)
    #print(f'tt : {time.process_time() - tt}')
    return rgb


cpdef draw_texture(faces, prev_matrices, next_landmarks, prev_img, next_img):
    cdef tuple f
    cdef tuple prev_matrix
    cdef tuple next_matrix
    cdef np.ndarray next_pts
    cdef np.ndarray unit_pts
    cdef np.ndarray prev_pts
    cdef double last_time
    cdef int count0 = 0
    for i in xrange(len(faces)):
        f = faces[i]
        prev_matrix = prev_matrices[i]

        #print("{")

        last_time = time.time()
        next_matrix = create_face_matrix_object(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        #print(f'Time to create face Matrix Object : {time.time() - last_time}')

        last_time = time.time()
        next_pts = get_pts_in_triangle(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        #print(f'Time to get pts in triangle : {time.time() - last_time}')

        if next_pts.shape[0] is not 0:
            last_time = time.time()
            unit_pts = to_unit_pts(next_matrix, next_pts)
            #print(f'Getting Unit Pts: {time.time() - last_time}')

            last_time = time.time()
            prev_pts = from_unit_pts(prev_matrix, unit_pts)
            #print(f'Getting Prev Pts: {time.time() - last_time}')

            last_time = time.time()
            next_img = draw_triangle(prev_pts, next_pts, prev_img, next_img)
            #print(f'Draw Triangle: {time.time() - last_time}')
        else:
            count0 += 1

        #print("}")
    print("count 0 " + str(count0))
    return next_img

