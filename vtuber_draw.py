import numpy as np
import cv2
import random

def draw_face_matrix(face_matrix, image, faces, coords):
    face_index = face_matrix[0]
    a = face_matrix[1]
    b = face_matrix[2]
    c = face_matrix[3]
    d = face_matrix[4]
    center_vertice = faces[face_index][1]
    relative_x = coords[center_vertice][0]
    relative_y = coords[center_vertice][1]
    cv2.arrowedLine(image, (relative_x, relative_y), (relative_x + a, relative_y), (0, 255, 0), 2)
    cv2.arrowedLine(image, (relative_x + a, relative_y), (relative_x + a, relative_y + b), (0, 255, 0), 2)
    cv2.arrowedLine(image, (relative_x, relative_y), (relative_x + c, relative_y), (255, 0, 0), 2)
    cv2.arrowedLine(image, (relative_x + c, relative_y), (relative_x + c, relative_y + d), (255, 0, 0), 2)


def draw(image, face, color, coords, texture_import_magnification):
    pt0 = coords[face[0]] / texture_import_magnification
    pt1 = coords[face[1]] / texture_import_magnification
    pt2 = coords[face[2]] / texture_import_magnification
    pts = np.array([[pt0, pt1], [pt1, pt2], [pt2, pt0]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, pts=[pts], color=(color[0], color[1], color[2]))


faceColors = []


def init_face_colors(faces):
    for x in faces:
        faceColors.append([random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)])
