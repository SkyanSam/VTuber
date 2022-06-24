#cython: language_level=3

import numpy as np
cimport numpy as np
from skimage.draw import polygon
import cv2
import mediapipe as mp
import random
import math
import itertools as its
import os.path
import copy
import time

cdef create_blank(width, height, rgb_color=(0, 0, 0)):
    cdef np.ndarray image = np.zeros((height, width, 3), np.uint8)  # Create black blank image
    image[:] = tuple(reversed(rgb_color)) # Fill image with color (reversed because BGR)
    return image

cdef round_pt(pt):
    return (round(pt[0]), round(pt[1]))


cdef create_face_matrix_object(A, B, C):
    return (round_pt(B), np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]]))

cdef to_unit_pts(face_matrix_object, pts):
    cdef new_pts = copy.deepcopy(pts)
    new_pts[0] -= face_matrix_object[0][0]
    new_pts[1] -= face_matrix_object[0][1]
    return np.matmul(np.linalg.inv(face_matrix_object[1]), new_pts)

cdef from_unit_pts(face_matrix_object, pts):
    cdef np.ndarray new_pts = np.matmul(face_matrix_object[1], pts)
    new_pts[0] += face_matrix_object[0][0]
    new_pts[1] += face_matrix_object[0][1]
    return new_pts

cdef get_pts_in_triangle(A, B, C):
    cdef np.ndarray xx
    cdef np.ndarray yy
    xx, yy = polygon(np.array([A[0], B[0], C[0]]), np.array([A[1], B[1], C[1]]))
    return np.stack((xx, yy))

cdef draw_triangle(prev_pts, next_pts, prev_img, next_img):
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

cdef create_face_matrices(faces, prev_landmarks):
    cdef list face_matrices = []
    for f in faces:
        face_matrices.append(create_face_matrix_object(prev_landmarks[f[0]], prev_landmarks[f[1]], prev_landmarks[f[2]]))
    return face_matrices

cdef draw_texture(faces, prev_matrices, next_landmarks, prev_img, next_img):
    cdef tuple f
    cdef tuple prev_matrix
    cdef tuple next_matrix
    cdef np.ndarray next_pts
    cdef np.ndarray unit_pts
    cdef np.ndarray prev_pts
    for i in range(0, len(faces)):
        f = faces[i]
        prev_matrix = prev_matrices[i]
        next_matrix = create_face_matrix_object(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        next_pts = get_pts_in_triangle(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        if next_pts.shape[0] is not 0:
            unit_pts = to_unit_pts(next_matrix, next_pts)
            prev_pts = from_unit_pts(prev_matrix, unit_pts)
            next_img = draw_triangle(prev_pts, next_pts, prev_img, next_img)
    return next_img

# Main VTuber Program
cdef add_edge_to_graph(v1, v2, graph):
    graph_len = len(graph)
    if graph_len <= v1:
        for x in range(0, v1 - graph_len + 1):
            graph.append([])
    graph[v1].append(v2)
    return graph

cdef export(edges, next_landmarks, width, height, m):
    cdef np.ndarray export_image = create_blank(int(width * m), int(width * m), rgb_color=(255, 255, 255))
    cdef list export_coords = []
    for c in range(0, len(next_landmarks)):
        export_coords.append([next_landmarks[c][0] * m, next_landmarks[c][1] * m])
    for e in edges:
        cv2.arrowedLine(export_image, (int(export_coords[e[0]][0]), int(export_coords[e[0]][1])),
                        (int(export_coords[e[1]][0]), int(export_coords[e[1]][1])), (0, 0, 0), 2)
    cv2.imwrite("texture.jpg", export_image)
    np.save("coords.npy", np.array(export_coords))

cpdef main():
    cdef int texture_export_magnification = 2
    cdef int texture_import_magnification = 2
    cdef int video_export_width = 800
    cdef int video_export_height = 800
    cdef int video_export_x_offset = 0
    cdef int video_export_y_offset = 0
    cdef object mp_drawing = mp.solutions.drawing_utils
    cdef object mp_holistic = mp.solutions.holistic
    cdef object cap = cv2.VideoCapture(0)  # Video capture number match with webcam
    cdef list faces = []
    cdef list edges = list(mp_holistic.FACEMESH_TESSELATION)
    cdef int num_of_edges = len(edges)
    cdef list face_matrices = []
    cdef list prev_landmarks = []
    cdef list next_landmarks = []
    cdef list graph = [[]]
    cdef np.ndarray reference_texture = None

    if os.path.exists("texture.png"):
        reference_texture = cv2.imread("texture.png", cv2.IMREAD_COLOR)
        cv2.imshow("texture", reference_texture)
        cv2.waitKey(0)

    for e in edges:
        graph = add_edge_to_graph(e[0], e[1], graph)
        graph = add_edge_to_graph(e[1], e[0], graph)

    num_of_vertices = len(graph)

    print("# of Vertices : " + str(num_of_vertices))
    print("# of Edges : " + str(num_of_edges))

    cdef int neighbor
    cdef int neighbor2
    cdef list face_list
    cdef tuple face_tuple
    for v in range(0, num_of_vertices):
        for n in range(0, len(graph[v])):
            neighbor = graph[v][n]
            for n2 in range(0, len(graph[neighbor])):
                neighbor2 = graph[neighbor][n2]
                if neighbor2 in graph[v]:
                    face_list = [v, neighbor, neighbor2]
                    face_list.sort()
                    face_tuple = tuple(face_list)
                    if face_tuple not in faces:
                        faces.append(face_tuple)

    print(f'# of Faces : {len(faces)}')
    cdef np.ndarray arr
    if os.path.exists("coords.npy"):
        arr = np.load("coords.npy")
        for x in range(0, arr.shape[0]):
            prev_landmarks.append((arr[x][0], arr[x][1]))
        prev_matrices = create_face_matrices(faces, prev_landmarks)
        print(f'# Prev Matrices : {len(prev_matrices)}')
        print(f'# Prev Landmarks : {len(prev_landmarks)}')

    cdef double frame_start
    cdef bint ret
    cdef np.ndarray frame
    cdef np.ndarray image
    cdef int height
    cdef int width
    cdef object results

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            frame_start = time.time()
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Recolor Feed
            height, width = image.shape[:2]
            virtual_image = create_blank(video_export_width, video_export_height, rgb_color=(100, 100, 100))
            results = holistic.process(image) # Make Detection
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Recolor image back to BGR for rendering

            if results.face_landmarks is not None:
                next_landmarks = []
                for landmark in results.face_landmarks.landmark:
                    next_landmarks.append((landmark.x * image.shape[1], landmark.y * image.shape[0]))

                if prev_landmarks is []:
                    prev_landmarks = next_landmarks
                    prev_matrices = create_face_matrices(faces, prev_landmarks)

                if reference_texture is not None:
                    virtual_image = draw_texture(faces, prev_matrices, next_landmarks, reference_texture, virtual_image)

            cv2.imshow('Webcam', image)
            cv2.imshow('VTuber', virtual_image)
            print(time.time() - frame_start)
            if cv2.waitKey(33) == ord('q'):
                input_ans = input("Export? [Y/N]")
                if input_ans == "Y":
                    export(edges, next_landmarks, width, height, texture_export_magnification)
                break

        cap.release()
        cv2.destroyAllWindows()