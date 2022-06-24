import mediapipe as mp
import cv2
import numpy as np
import random

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# REALTIME WEBCAM FEED

cap = cv2.VideoCapture(0)  # Video capture number match with webcam

faces = [[]]
edges = list(mp_holistic.FACEMESH_TESSELATION)
num_of_edges = len(edges)
graph = [[]]


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
faces = []
for v in range(0, num_of_vertices):
    for n in range(0, len(graph[v])):
        neighbor = graph[v][n]
        for n2 in range(0, len(graph[neighbor])):
            neighbor2 = graph[neighbor][n2]
            if (neighbor2 in graph[v]):
                face = [v, neighbor, neighbor2]
                face.sort()
                # print((v, neighbor) in edges or (neighbor, v) in edges)
                # print((neighbor, neighbor2) in edges or (neighbor2, neighbor) in edges)
                # print((neighbor2, v) in edges or (v, neighbor2) in edges)
                if not face in faces:
                    faces.append(face)

faceColors = []
for x in faces:
    faceColors.append([random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)])
print("EDGES")
print(edges)
print("FACES")
print(faces)
print("GRAPH")
print(graph)
print("Faces Computed with Graph Theory : " + str(2 - num_of_vertices + (len(edges) / 2)))
print("Faces Computed with My Formula : " + str(len(faces)))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    coords = []
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detection
        results = holistic.process(image)
        # print(results.face_landmarks)
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if not results.face_landmarks.landmark is None:
            coords = []
            for landmark in results.face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                coords.append([relative_x, relative_y])


        #
        # for face in results.multi_face_landmarks:
        #    for landmark in face.landmark:
        #
        #
        def draw(face, color):
            pt0 = coords[face[0]]
            pt1 = coords[face[1]]
            pt2 = coords[face[2]]
            pts = np.array([[pt0, pt1], [pt1, pt2], [pt2, pt0]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, pts=[pts], color=(color[0], color[1], color[2]))


        for f in range(0, len(faces)):
            draw(faces[f], faceColors[f])
        # Recolor
        # print(results)
        cv2.imshow('Webcam', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# mp_holistic.FACE_CONNECTIONS -> FACEMESH_TESSELATION
# mp_holistic.POSE_CONNECTIONS
# mp_holistic.HAND_CONNECTIONS
# FACEMESH_CONTOURS outline of face
