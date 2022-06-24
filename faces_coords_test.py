import cv2
import numpy as np
import mediapipe as mp
import time
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
import matrix_engine

start = time.time()
print(matrix_engine.get_pts_in_triangle_3((100,100),(100,150),(150,150)))
print(time.time() - start)
"""
def grid(image):
    img = image
    x = 0
    y = 0
    while(x < 399):
        img = cv2.line(img, (x,0), (x,399), (0,0,0), 1)
        x += 100
    while (y < 399):
        img = cv2.line(img, (0, y), (399, y), (0, 0, 0), 1)
        y += 100
    return img

faces = [[0, 1, 2]]
reference_coords = [[100, 100], [100, 150], [150, 150]]
coords = [[100+100+100, 100+100], [100+100, 150+100], [150+100, 150+100]]
virtual_image = vtuber_utils.create_blank(400,400,rgb_color=(100,100,100))
reference_texture = vtuber_utils.create_blank(400,400,rgb_color=(100,100,100))
reference_texture = grid(reference_texture)
virtual_image = grid(virtual_image)
reference_texture = cv2.fillPoly(reference_texture, pts=[np.array([(100, 100), (100, 150), (150, 150)])], color=(255, 0, 0))
reference_texture = cv2.circle(reference_texture, (100,100), 50, (0,255,0), 5)

face_reference_objects = [vtuber_utils.create_face_matrix(0,reference_coords,faces)]
face_objects = [vtuber_utils.create_face_matrix(0,coords,faces)]
old_faces_coords = vtuber_utils.get_faces_coords(faces, reference_texture.shape[1], reference_texture.shape[0], reference_coords)
unit_faces_coords = vtuber_utils.get_unit_faces_coords(old_faces_coords, face_reference_objects, reference_coords, faces)
new_faces_coords = vtuber_utils.get_new_faces_coords(unit_faces_coords, face_objects, coords, faces)
virtual_image = vtuber_utils.draw_texture(old_faces_coords, new_faces_coords, reference_texture, virtual_image)
virtual_image = cv2.circle(virtual_image, (200,300), 1, (0,0,255),2)
virtual_image = cv2.circle(virtual_image, (250,200), 1, (0,0,255),2)
virtual_image = cv2.circle(virtual_image, (250,250), 2, (0,0,255),2)
virtual_image = cv2.circle(virtual_image, (300,200), 1, (0,255,255),2)
virtual_image = cv2.circle(virtual_image, (200,250), 1, (0,255,255),2)
virtual_image = cv2.circle(virtual_image, (250,250), 1, (0,255,255),2)
print(face_reference_objects)
print(face_objects)
print(len(old_faces_coords[0]))
print(len(unit_faces_coords[0]))
print(len(new_faces_coords[0]))
print(np.dot([[0,50],[-50,0]],[[1,0,0],[0,0,1]]))

# I HAVE A THEORY
# NEW POINTS IS = (MATRIX)*OLD_POINTS or (MATRIX)*(MATRIX)*UNIT_POINTS
# We need to get down to the bottom of this
# Another theory is that the transform is adding too much
# Maybe center_vertice > 1000 or something stupid like that or it may not be adding properly
# Check if any real world coords are effected globally and not locally (similar to image getting smaller and smaller erorr)
cv2.imshow("Image", reference_texture)
cv2.waitKey(0)
cv2.imshow("New Image", virtual_image)
#cv2.waitKey(0)
"""