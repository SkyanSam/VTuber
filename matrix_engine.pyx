import numpy as np
cimport numpy as np
import cv2
import copy
import time
import math
import itertools as its
from skimage.draw import polygon

cpdef create_blank(width, height, rgb_color=(0, 0, 0)):
    cdef np.ndarray image = np.zeros((height, width, 3), np.uint8)  # Create black blank image
    image[:] = tuple(reversed(rgb_color)) # Fill image with color (reversed because BGR)
    return image

cpdef round_pt(pt):
    return (round(pt[0]), round(pt[1]))

cpdef create_face_matrix_object(A, B, C):
    return (round_pt(B), np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]]))

cpdef to_unit_pts(face_matrix_object, pts):
    cdef new_pts = copy.deepcopy(pts)
    new_pts[0] -= face_matrix_object[0][0]
    new_pts[1] -= face_matrix_object[0][1]
    return np.matmul(np.linalg.inv(face_matrix_object[1]), new_pts)

cpdef from_unit_pts(face_matrix_object, pts):
    cdef np.ndarray new_pts = np.matmul(face_matrix_object[1], pts)
    new_pts[0] += face_matrix_object[0][0]
    new_pts[1] += face_matrix_object[0][1]
    return new_pts

cpdef get_pts_in_triangle(A, B, C):
    cdef np.ndarray xx
    cdef np.ndarray yy
    xx, yy = polygon(np.array([A[0], B[0], C[0]]), np.array([A[1], B[1], C[1]]))
    return np.stack((xx, yy))

cpdef draw_triangle(prev_pts, next_pts, prev_img, next_img):
    cdef list base_color = [0, 0, 0]
    cdef int prev_x
    cdef int prev_y
    cdef int next_x
    cdef int next_y
    for i in range(0, len(prev_pts[0])):
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

cpdef draw_texture(faces, prev_matrices, next_landmarks, prev_img, next_img):
    cdef tuple f
    cdef tuple prev_matrix
    cdef tuple next_matrix
    cdef np.ndarray next_pts
    cdef np.ndarray unit_pts
    cdef np.ndarray prev_pts
    cdef double last_time
    cdef int count0 = 0
    for i in range(0, len(faces)):
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

