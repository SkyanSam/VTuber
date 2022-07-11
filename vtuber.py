# TEST MP HOLISTIC VS MP FACEMASK
# PRINT OUT THE TESSELATION
# DOES THE TESSELATION / COORDS CHANGE!??!?!?!
# IM GOING TO GO INSANE

# make fucking C library because doing shit in Cython is prob slow af
import json

from joblib.numpy_pickle_utils import xrange
from numba import njit, prange
from numba import jit
from scipy.ndimage import map_coordinates

texture_export_magnification = 2
texture_import_magnification = 2
video_export_width = 800
video_export_height = 800
video_export_x_offset = 0
video_export_y_offset = 0


import cv2
import numpy as np
import mediapipe as mp
import random
import math
import os.path
import copy
import time
#import pyximport
#pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
#import vtuber_utils
import matrix_engine

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


cap = cv2.VideoCapture(0)  # Video capture number match with webcam
faces = []
edges = list(mp_holistic.FACEMESH_TESSELATION)
num_of_edges = len(edges)
face_matrices = []
prev_landmarks = []
next_landmarks = []
graph = [[]]
reference_texture = None

if os.path.exists("texture.png"):
    reference_texture = cv2.imread("texture.png", cv2.IMREAD_COLOR)
    cv2.imshow("texture", reference_texture)
    cv2.waitKey(0)

def add_edge_to_graph(v1, v2):
    graph_len = len(graph)
    if graph_len <= v1:
        for x in range(0, v1 - graph_len + 1):
            graph.append([])
    graph[v1].append(v2)

@njit(parallel=True)
def draw_texture_2_helper(faces, faces_pts, unit_pts_list, next_landmarks, prev_img, export_width, export_height, out_coord, len_faces):
    for i in prange(len_faces):
        f = faces[i]
        #next_matrix = matrix_engine.create_face_matrix_object(next_landmarks[f[0]], next_landmarks[f[1]], next_landmarks[f[2]])
        #unit_pts = to_unit_pts(prev_matrix, prev_pts)
        prev_pts = faces_pts[i]
        #next_pts = matrix_engine.from_unit_pts(next_matrix, unit_pts_list[i])
        A = next_landmarks[f[0]]
        B = next_landmarks[f[1]]
        C = next_landmarks[f[1]]
        #next_pts = np.matmul(np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]]), unit_pts_list[i])
        next_pts = np.array([[A[0]- B[0], C[0] - B[0]], [A[1]- B[1], C[1] - B[1]]]) @ unit_pts_list[i]
        next_pts[0] += next_landmarks[f[1]][0] #face_matrix_object[0][0]
        next_pts[1] += next_landmarks[f[1]][1] #face_matrix_object[0][1]
        prev_pts_t = prev_pts.T
        next_pts_t = next_pts.T
        for p in prange(prev_pts_t.shape[0]):
            prev_pt = prev_pts_t[p]
            next_pt = next_pts_t[p]
            out_coord[0][round(next_pt[1])][round(next_pt[0])] = prev_pt[0]
            out_coord[1][round(next_pt[1])][round(next_pt[0])] = prev_pt[1]
    return out_coord

def draw_texture_2(faces, faces_pts, unit_pts_list, next_landmarks, prev_img, export_width, export_height):
    #faces = np.array(faces)
    #unit_pts_list = np.array(unit_pts_list)
    #next_landmarks = np.array(next_landmarks)
    mesh_x, mesh_y = np.arange(export_width), np.arange(export_height)
    #ini_coord = np.meshgrid(mesh_x, mesh_y)
    out_coord = np.array(np.meshgrid(mesh_x, mesh_y))
    #out_coord = [np.zeros(prev_img.shape, dtype=np.int32), np.zeros(prev_img.shape, dtype=np.int32)]
    len_faces = len(faces)
    out_coord = draw_texture_2_helper(np.array(faces), faces_pts, unit_pts_list, next_landmarks, prev_img, export_width, export_height, out_coord, len_faces)
    nrows = export_height
    ncols = export_width
    tt = time.process_time()
    r = map_coordinates(prev_img[:, :, 0], out_coord, order=1, mode='nearest')
    g = map_coordinates(prev_img[:, :, 1], out_coord, order=1, mode='nearest')
    b = map_coordinates(prev_img[:, :, 2], out_coord, order=1, mode='nearest')
    rgb = np.array([r, g, b])
    rgb = rgb.reshape(3, nrows, ncols).transpose(1, 2, 0)
    # print(f'tt : {time.process_time() - tt}')
    return rgb

for e in edges:
    add_edge_to_graph(e[0], e[1])
    add_edge_to_graph(e[1], e[0])

num_of_vertices = len(graph)

print("# of Vertices : " + str(num_of_vertices))
print("# of Edges : " + str(num_of_edges))

for v in range(0, num_of_vertices):
    for n in range(0, len(graph[v])):
        neighbor = graph[v][n]
        for n2 in range(0, len(graph[neighbor])):
            neighbor2 = graph[neighbor][n2]
            if neighbor2 in graph[v]:
                face = [v, neighbor, neighbor2]
                face.sort()
                face = tuple(face)
                if face not in faces:
                    faces.append(face)

print(f'# of Faces : {len(faces)}')

uv_map_coords = []
with open("uv_map.json") as json_file:
    data = json.load(json_file)
    for i in range(0,468):
        uv_map_coords.append((float(data["u"][str(i)]) * reference_texture.shape[1], float(data["v"][str(i)]) * reference_texture.shape[0], 0))
prev_landmarks = uv_map_coords
prev_matrices = matrix_engine.create_face_matrices(faces, prev_landmarks)
faces_pts = matrix_engine.get_pts_in_faces(faces, prev_landmarks)
unit_pts_list = matrix_engine.get_unit_pts_list(faces, faces_pts, prev_matrices, prev_landmarks)
"""       
if os.path.exists("coords.npy"):
    arr = np.load("coords.npy")
    prev_landmarks = []
    for x in range(0, arr.shape[0]):
        prev_landmarks.append((arr[x][0], arr[x][1]))
    prev_matrices = matrix_engine.create_face_matrices(faces, prev_landmarks)
    print(f'# Prev Matrices : {len(prev_matrices)}')
    print(f'# Prev Landmarks : {len(prev_landmarks)}')
"""

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        frame_start = time.time()

        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Recolor Feed
        height, width = image.shape[:2]
        virtual_image = matrix_engine.create_blank(video_export_width, video_export_height, rgb_color=(100, 100, 100))
        results = holistic.process(image) # Make Detection

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Recolor image back to BGR for rendering

        if results.face_landmarks is not None:
            next_landmarks = []
            for landmark in results.face_landmarks.landmark:
                next_landmarks.append((landmark.x * image.shape[1], landmark.y * image.shape[0]))

            if prev_landmarks is []:
                prev_landmarks = next_landmarks
                prev_matrices = matrix_engine.create_face_matrices(faces, prev_landmarks)

            if reference_texture is not None:
                time_check = time.time()
                #print("before draw texture")
                virtual_image = draw_texture_2(faces, faces_pts, unit_pts_list, next_landmarks, reference_texture, image.shape[1], image.shape[0])
                #virtual_image = matrix_engine.draw_texture(faces, prev_matrices, next_landmarks, reference_texture, virtual_image)
                #print(f'After draw texture: {time.time() - time_check}')
        """
        def export():
            export_image = matrix_engine.create_blank(int(width * texture_export_magnification),
                                                     int(height * texture_export_magnification),
                                                     rgb_color=(255, 255, 255))
            export_coords = []
            for c in range(0, len(next_landmarks)):
                export_coords.append([next_landmarks[c][0] * texture_export_magnification, next_landmarks[c][1] * texture_export_magnification])
            for e in edges:
                cv2.arrowedLine(export_image, (int(export_coords[e[0]][0]), int(export_coords[e[0]][1])),
                                (int(export_coords[e[1]][0]), int(export_coords[e[1]][1])), (0, 0, 0), 2)
            cv2.imwrite("texture.jpg", export_image)
            np.save("coords.npy", np.array(export_coords))
            np.save("edges.npy", np.array(edges))
        """

        #cv2.imshow('Webcam', image)
        cv2.imshow('VTuber', virtual_image)
        print(time.time() - frame_start)
        if cv2.waitKey(33) == ord('q'):
            """
            input_ans = input("Export? [Y/N]")
            if input_ans == "Y":
                export()
            """
            break



    cap.release()
    cv2.destroyAllWindows()