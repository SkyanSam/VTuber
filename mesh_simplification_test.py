import json
import random

import quad_mesh_simplify
import numpy as np
import mediapipe as mp
import cv2
from skimage.draw import polygon

from quad_mesh_simplify import simplify_mesh

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import matrix_engine

def rpt(pt):
    return matrix_engine.round_pt(pt)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

positions = []
#for p in np.load("coords.npy"):
    #positions.append((p[0], p[1], 0))
#positions = np.array(positions)
#print(positions)


((1,2,3), (2,3,4))

edges = list(mp_holistic.FACEMESH_TESSELATION)
num_of_edges = len(edges)
graph = [[]]
faces = []

def create_blank(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)  # Create black blank image
    image[:] = tuple(reversed(rgb_color)) # Fill image with color (reversed because BGR)
    return image

def add_edge_to_graph(v1, v2):
    graph_len = len(graph)
    if graph_len <= v1:
        for x in range(0, v1 - graph_len + 1):
            graph.append([])
    graph[v1].append(v2)

def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

def is_inside(pt, pt1, pt2, pt3):
    x, y, z = pt
    x1, y1, z1 = pt1
    x2, y2, z2 = pt2
    x3, y3, z3 = pt3
    A = triangle_area(x1, y1, x2, y2, x3, y3)
    A1 = triangle_area(x, y, x2, y2, x3, y3)
    A2 = triangle_area(x1, y1, x, y, x3, y3)
    A3 = triangle_area(x1, y1, x2, y2, x, y)
    return abs(A - (A1 + A2 + A3)) < 0.5

def is_inside_2(pt, pt1, pt2, pt3):
    img = np.zeros((400, 400), dtype=np.uint8)
    r = np.array([round(pt1[0]), round(pt2[0]), round(pt3[0])])
    c = np.array([round(pt1[1]), round(pt2[1]), round(pt3[1])])
    rr, cc = polygon(r, c)
    img[rr, cc] = 1
    return img[round(pt[1]), round(pt[0])] == 1

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

print(len(faces))

uv_map_coords = []
with open("uv_map.json") as json_file:
    data = json.load(json_file)
    for i in range(0,468):
        uv_map_coords.append((float(data["u"][str(i)]) * 400, float(data["v"][str(i)]) * 400, 0))

uv_img = create_blank(400, 400, rgb_color=(255, 255, 255))
for f in faces:
    pt1 = [uv_map_coords[f[0]][0], uv_map_coords[f[0]][1]]
    pt2 = [uv_map_coords[f[1]][0], uv_map_coords[f[1]][1]]
    pt3 = [uv_map_coords[f[2]][0], uv_map_coords[f[2]][1]]
    _color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    points = np.array([pt1, pt2, pt3])
    uv_img = cv2.fillPoly(uv_img, pts=[np.int32(points)], color=_color)
cv2.imshow("UV IMG", uv_img)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Recolor Feed
        height, width = image.shape[:2]
        results = holistic.process(image) # Make Detection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Recolor image back to BGR for rendering


        if results.face_landmarks is not None:
            next_landmarks = []
            positions = []
            for i, landmark in enumerate(results.face_landmarks.landmark):
                z = landmark.z * ((image.shape[0] + image.shape[1]) / 2)
                positions.append([landmark.x * image.shape[1], landmark.y * image.shape[0], z])
                uv_map_coords[i] = (uv_map_coords[i][0], uv_map_coords[i][1], z)

            v_image = create_blank(image.shape[1], image.shape[0], rgb_color=(0,0,0))
            mp_drawing.draw_landmarks(v_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            cv2.imshow("cam", v_image)
            if cv2.waitKey(33) == ord('q'):
                new_positions, new_face = simplify_mesh(np.array(uv_map_coords), np.array(faces, dtype=np.uint32), 150)
                img = create_blank(image.shape[1], image.shape[0], rgb_color=(255, 255, 255))
                for f in new_face:
                    pt1 = [new_positions[f[0]][0], new_positions[f[0]][1]]
                    pt2 = [new_positions[f[1]][0], new_positions[f[1]][1]]
                    pt3 = [new_positions[f[2]][0], new_positions[f[2]][1]]
                    _color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    points = np.array([pt1, pt2, pt3])
                    img = cv2.fillPoly(img, pts=[np.int32(points)], color=_color)

                upos = []
                uv_map_coords = np.array(uv_map_coords)
                for ni, n in enumerate(new_positions):
                    for pi, p in enumerate(uv_map_coords):
                        #print(f'comarping {n} with {p}')
                        if n[0] == p[0] and n[1] == p[1] and n[2] == p[2]:
                            upos.append(("pos", pi))
                            break
                    if not ni < len(upos):
                        for fi, f in enumerate(faces):
                            if is_inside(n, uv_map_coords[f[0]], uv_map_coords[f[1]], uv_map_coords[f[2]]):
                                upos.append(("face", fi, ni))
                    if not ni < len(upos):
                        upos.append(("pos", 0))
                        print("FAIL!!!!")
                print(f'upos : {upos}')

                new_landmarks = []
                prev_matrices = matrix_engine.create_face_matrices(faces, uv_map_coords)
                next_matrices = matrix_engine.create_face_matrices(faces, positions)
                for i in upos:
                    if i[0] == "pos":
                        new_landmarks.append(positions[i[1]])
                    elif i[0] == "face":
                        _faceIndex = i[1]
                        f = faces[_faceIndex]
                        _newPositionsIndex = i[2]
                        l = new_positions[i[2]]
                        ul = matrix_engine.to_unit_pts(prev_matrices[i[1]], np.array([[l[0]],[l[1]]]))
                        nl = matrix_engine.from_unit_pts(next_matrices[i[1]], ul)
                        new_landmarks.append((nl[0][0], nl[1][0], 0))
                        tri_img = create_blank(1000, 1000, rgb_color=(255, 255, 255))
                        #print(uv_map_coords[f[0]])
                        #print(rpt(uv_map_coords[f[0]]))
                        tri_img = cv2.fillPoly(tri_img, pts=[np.array([rpt(uv_map_coords[f[0]]), rpt(uv_map_coords[f[1]]), rpt(uv_map_coords[f[2]])], dtype=np.int32)], color=(0,255,0))
                        tri_img = cv2.fillPoly(tri_img, pts=[np.array([rpt(positions[f[0]]), rpt(positions[f[1]]), rpt(positions[f[2]])], dtype=np.int32)], color=(255,0,0))
                        tri_img = cv2.circle(tri_img, (int(nl[0][0]), int(nl[1][0])), 3, (0, 0, 255), -1)
                        tri_img = cv2.circle(tri_img, (int(l[0]), int(l[1])), 3, (0, 0, 0), -1)
                        #cv2.imshow("triangle img", tri_img)
                        #cv2.waitKey(0)
                    else:
                        pass
                        #print("ERR")
                print(new_landmarks)

                new_img = create_blank(width, height, rgb_color=(255, 255, 255))
                for f in new_face:
                    pt1 = [new_landmarks[f[0]][0], new_landmarks[f[0]][1]]
                    pt2 = [new_landmarks[f[1]][0], new_landmarks[f[1]][1]]
                    pt3 = [new_landmarks[f[2]][0], new_landmarks[f[2]][1]]
                    _color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    points = np.array([pt1, pt2, pt3])
                    new_img = cv2.fillPoly(new_img, pts=[np.int32(points)], color=_color)
                cv2.imshow("NEW IMG", new_img)
                new_img_2 = create_blank(width, height, rgb_color=(255, 255, 255))
                for l in new_landmarks:
                    new_img_2 = cv2.circle(new_img_2, (int(l[0]), int(l[1])), 3, (255,0,0), -1)
                cv2.imshow("NEW IMG 2", new_img_2)

                uv_img_2 = create_blank(400, 400, rgb_color=(255, 255, 255))
                for f in faces:
                    pt1 = [uv_map_coords[f[0]][0], uv_map_coords[f[0]][1]]
                    pt2 = [uv_map_coords[f[1]][0], uv_map_coords[f[1]][1]]
                    pt3 = [uv_map_coords[f[2]][0], uv_map_coords[f[2]][1]]
                    _color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    points = np.array([pt1, pt2, pt3])
                    uv_img_2 = cv2.fillPoly(uv_img_2, pts=[np.int32(points)], color=_color)
                for l in new_positions:
                    uv_img_2_2 = cv2.circle(uv_img_2, (int(l[0]), int(l[1])), 3, (255,0,0), -1)
                cv2.imshow("UV IMG 2", uv_img_2)
                """
                new_face_update = []
                for f in new_face:
                    new_face_update.append((upos[f[0]], upos[f[1]], upos[f[2]]))
                """

                print("done")
                print("FACES LENGTH")
                print(len(new_face))

                cv2.imshow("image", img)
                cv2.waitKey(0)
                break
    cap.release()
    cv2.destroyAllWindows()

    #cv2.arrowedLine(image, new_positions[f[0]], int(export_coords[e[0]][1])),
                    #(int(export_coords[e[1]][0]), int(export_coords[e[1]][1])), (0, 0, 0), 2)
#print(new_positions)
#print(new_face)
#print(len(new_positions))
