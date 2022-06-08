import mediapipe as mp
import cv2
import numpy as np
import random
import math



mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# REALTIME WEBCAM FEED

cap = cv2.VideoCapture(0) # Video capture number match with webcam

faces = [[]]
edges = list(mp_holistic.FACEMESH_TESSELATION)
num_of_edges = len(edges)
graph = [[]]

def lerp(i1, i2, t):
    return ((i2 - il) / 2) + il

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
                #print((v, neighbor) in edges or (neighbor, v) in edges)
                #print((neighbor, neighbor2) in edges or (neighbor2, neighbor) in edges)
                #print((neighbor2, v) in edges or (v, neighbor2) in edges)
                if not face in faces:
                    faces.append(face)
                    

def create_face_matrix(face_index, coords):
    face = faces[face_index]
    face_matrix = []
    face_matrix.append(face_index) # FACE INDEX
    face_matrix.append(coords[face[0]][0] - coords[face[1]][0]) # LENGTH X / A
    face_matrix.append(coords[face[0]][1] - coords[face[1]][1]) # LENGTH Y / B
    face_matrix.append(coords[face[2]][0] - coords[face[1]][0]) # WIDTH X / C
    face_matrix.append(coords[face[2]][1] - coords[face[1]][1]) # WIDTH Y / D
    return face_matrix
        
def transform(coord, face_matrix, coords): # Assuming the plane is centered around (0,0)
    x = coord[0]
    y = coord[1]
    face_index = face_matrix[0]
    a = face_matrix[1]
    b = face_matrix[2]
    c = face_matrix[3]
    d = face_matrix[4]
    center_vertice = faces[face_index][1]
    relative_x = coords[center_vertice][0]
    relative_y = coords[center_vertice][1]
    new_coord = [(a*x) + (b*y), (c*x) + (d*y)]
    new_coord[0] += relative_x
    new_coord[1] += relative_y
    return new_coord
    
def transform_inverse(coord, face_matrix, coords): # Assuming the plane is centered around center vertice
    x = coord[0]
    y = coord[1]
    face_index = face_matrix[0]
    a = face_matrix[1]
    b = face_matrix[2]
    c = face_matrix[3]
    d = face_matrix[4]
    center_vertice = faces[face_index][1]
    relative_x = coords[center_vertice][0]
    relative_y = coords[center_vertice][1]
    x -= relative_x
    y -= relative_y
    inv_det = 1 / ((a*d) - (b*c)) # Inverse Determinant
    new_coord = [(inv_det * d * x) + (-1 * inv_det * b * x), (-1 * inv_det * c * y) + (inv_det * a * y)]
    return new_coord

def draw_face_matrix(face_matrix, image):
    face_index = face_matrix[0]
    a = face_matrix[1]
    b = face_matrix[2]
    c = face_matrix[3]
    d = face_matrix[4]
    center_vertice = faces[face_index][1]
    relative_x = coords[center_vertice][0]
    relative_y = coords[center_vertice][1]
    cv2.arrowedLine(image, (relative_x, relative_y), (relative_x + a, relative_y), (0,255,0), 2)
    cv2.arrowedLine(image, (relative_x + a, relative_y), (relative_x + a, relative_y + b), (0,255,0), 2)
    cv2.arrowedLine(image, (relative_x, relative_y), (relative_x + c, relative_y), (255,0,0), 2)
    cv2.arrowedLine(image, (relative_x + c, relative_y), (relative_x + c, relative_y + d), (255,0,0), 2)
    

faceColors = []
for x in faces:
    faceColors.append([random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)])
print("EDGES")
print(edges)
print("FACES")
print(faces)
print("GRAPH")
print(graph)
print("Faces Computed with Graph Theory : " + str(2 - num_of_vertices + (len(edges) / 2)))
print("Faces Computed with My Formula : " + str(len(faces)))
    

face_reference_objects = []
reference_coords = []
face_objects = []
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    coords = []
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Make Detection
        results = holistic.process(image)
        #print(results.face_landmarks)
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
        if not results.face_landmarks is None:
            coords = []
            for landmark in results.face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                shape = image.shape 
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                coords.append([relative_x, relative_y])
                
            if face_reference_objects == []:
                reference_coords = coords
                print("REFERENCE COORDS")
                print(reference_coords)
                for faceIndex in range(0, len(faces)):
                    face_reference_objects.append(create_face_matrix(faceIndex, coords))
            else:
                face_objects = []
                for i in range(0, len(faces)):
                    face_objects.append(create_face_matrix(i, coords))
        
        #
        #for face in results.multi_face_landmarks:
        #    for landmark in face.landmark:
        #        
        #
        #i
        #if face_reference_objects == [] and not results.face_landmarks.landmark is None:
            
        
        def draw(face, color):
            pt0 = coords[face[0]]
            pt1 = coords[face[1]]
            pt2 = coords[face[2]]
            pts = np.array([[pt0, pt1],[pt1, pt2],[pt2, pt0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(image,pts = [pts], color = (color[0],color[1],color[2]))
        
        #for f in range(0, len(faces)):
            #draw(faces[f], faceColors[f])

        if len(face_objects) > 0:
            draw_face_matrix(face_objects[0], image)
            #draw(faces[0], (0,0,0))
            #print("--")
            #print(reference_coords[faces[0][0]])
            #print(reference_coords[faces[0][1]])
            #print(reference_coords[faces[0][2]])

            ex_coord_x = reference_coords[faces[0][0]][0] + reference_coords[faces[0][1]][0] + reference_coords[faces[0][2]][0]      
            ex_coord_x /= 3
            ex_coord_y = reference_coords[faces[0][0]][1] + reference_coords[faces[0][1]][1] + reference_coords[faces[0][2]][1]      
            ex_coord_y /= 3                                                                
            ex_coord_1 = [ex_coord_x, ex_coord_y]
            ex_coord_2 = transform_inverse(ex_coord_1, face_reference_objects[0], reference_coords)
            ex_coord_3 = transform(ex_coord_2, face_objects[0], coords)
            #print(ex_coord_3)
            cv2.circle(image, (int(ex_coord_3[0]), int(ex_coord_3[1])), 1, (255, 255, 0), thickness=-1)
            height, width = image.shape[:2]
            cv2.circle(image, (int(width / 2), int(height / 2)), 1, (255, 0, 255), thickness=-1)
            
        
        #Recolor
        #print(results)
        cv2.imshow('Webcam', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#mp_holistic.FACE_CONNECTIONS -> FACEMESH_TESSELATION
#mp_holistic.POSE_CONNECTIONS
#mp_holistic.HAND_CONNECTIONS
# FACEMESH_CONTOURS outline of face
