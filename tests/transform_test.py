from tkinter import Y
import numpy as np

def transform(x,y):
    #cdef int x = copy.deepcopy(coord[0])
    #cdef int y = copy.deepcopy(coord[1])
    #cdef int face_index = copy.deepcopy(face_matrix[0])
    #cdef int a = copy.deepcopy(face_matrix[1])
    #cdef int b = copy.deepcopy(face_matrix[2])
    #cdef int c = copy.deepcopy(face_matrix[3])
    #cdef int d = copy.deepcopy(face_matrix[4])
    #cdef int relative_x = copy.deepcopy(coords[center_vertice][0])
    #cdef int relative_y = copy.deepcopy(coords[center_vertice][1])
    #cdef list new_coord = [(a*x) + (b*y), (c*x) + (d*y)]
    a,b,c,d=0 #TEMP!
    return np.dot([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]],[[x], [y]])

def transform_inv(x,y,matrix):
    a,b,c,d=0 #TEMP!
    return np.dot(np.linalg.inv([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]]),[[x],[y]])

def transform_face(face_matrix, coords):
    pass

def sus(coords):
    # xx - coords[:, 0]
    # yy - coords[:, 1]
    return np.dot([[1,0],[1,1]], np.vstack([coords[:, 0].ravel(), coords[:, 1].ravel()])).T

arr = []
for x in range(0,1000):
    arr.append([x,x])

print(sus(np.array(arr)))
#print(transform_new(1,1))
#print(transform_inv_new(2,1))