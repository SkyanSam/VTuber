import random

import quad_mesh_simplify
import numpy as np
import mediapipe as mp
import cv2

from quad_mesh_simplify import simplify_mesh

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
            for landmark in results.face_landmarks.landmark:
                z = landmark.z * ((image.shape[0] + image.shape[1]) / 2)
                positions.append([landmark.x * image.shape[1], landmark.y * image.shape[0], z])
            print(len(positions))

            v_image = create_blank(image.shape[1], image.shape[0], rgb_color=(0,0,0))
            mp_drawing.draw_landmarks(v_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            cv2.imshow("cam", v_image)
            if cv2.waitKey(33) == ord('q'):
                positions = np.array(positions)
                new_positions, new_face = simplify_mesh(positions, np.array(faces, dtype=np.uint32), 150)
                img = create_blank(image.shape[1], image.shape[0], rgb_color=(255, 255, 255))
                for f in new_face:
                    pt1 = [new_positions[f[0]][0], new_positions[f[0]][1]]
                    pt2 = [new_positions[f[1]][0], new_positions[f[1]][1]]
                    pt3 = [new_positions[f[2]][0], new_positions[f[2]][1]]
                    _color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    points = np.array([pt1, pt2, pt3])
                    img = cv2.fillPoly(img, pts=[np.int32(points)], color=_color)

                upos = []
                for ni, n in enumerate(new_positions):
                    for pi, p in enumerate(positions):
                        if n[0] == p[0] and n[1] == p[1] and n[2] == p[2]:
                            upos.append(pi)
                            print("can find")
                            break
                        if n[0] == p[0] and n[1] == p[1] and n[2] == p[2]:
                            print("SUS")
                    print("Cant find")
                print(f'upos : {upos}')

                new_face_update = []
                for f in new_face:
                    new_face_update.append((upos[f[0]], upos[f[1]], upos[f[2]]))

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
