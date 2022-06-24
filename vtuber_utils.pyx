import numpy as np
cimport numpy as np
import cv2

cpdef lerp(i1, i2, t):
    return ((i2 - i1) / 2) + i1

def create_blank(width, height, rgb_color=(0, 0, 0)):
    cdef np.ndarray image = np.zeros((height, width, 3), np.uint8)  # Create black blank image
    cdef tuple color = tuple(reversed(rgb_color))  # Since OpenCV uses BGR, convert the color first
    image[:] = color  # Fill image with color
    return image

cpdef create_face_matrix(face_index, coords, faces):
    cdef list face = faces[face_index]
    cdef list face_matrix = []
    face_matrix.append(face_index) # FACE INDEX
    # i hat - length
    # j hat - width
    face_matrix.append(coords[face[0]][0] - coords[face[1]][0]) # LENGTH X / A i hat x
    face_matrix.append(coords[face[2]][0] - coords[face[1]][0])  # WIDTH X / C j hat x
    face_matrix.append(coords[face[0]][1] - coords[face[1]][1]) # LENGTH Y / B i hat y
    face_matrix.append(coords[face[2]][1] - coords[face[1]][1]) # WIDTH Y / D j hat y
    #face_matrix[2] *= -1
    #face_matrix[1] *= -1
    #face_matrix[3] *= -1
    #face_matrix[4] *= -1
    return face_matrix

cpdef transform_inverse(coord, face_matrix, coords, faces):
    cdef double x = coord[0] - coords[faces[face_matrix[0]][1]][0]
    cdef double y = coord[1] - coords[faces[face_matrix[0]][1]][1]
    return np.dot(np.linalg.inv([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]]),[[x], [y]]).T

cpdef transform(coord, face_matrix, coords, faces): # Assuming the plane is centered around (0,0)
    cdef double x = coord[0] + coords[faces[face_matrix[0]][1]][0]
    cdef double y = coord[1] + coords[faces[face_matrix[0]][1]][1]
    return np.dot([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]], [[x], [y]]).T

cpdef transform_inverse_multiple(face_coords, face_matrix, coords, faces):
    cdef np.ndarray np_coords = np.asarray(face_coords).T
    cdef np.ndarray xx = np_coords[0] - coords[faces[face_matrix[0]][1]][0]
    cdef np.ndarray yy = np_coords[1] - coords[faces[face_matrix[0]][1]][1]
    return np.matmul(np.linalg.inv([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]]), [xx,yy]).T

cpdef transform_multiple(face_coords, face_matrix, coords, faces): # Assuming the plane is centered around (0,0)
    cdef np.ndarray xx
    cdef np.ndarray yy
    #xx, yy = np.dot([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]],np.vstack([coords[:,0].ravel(), coords[:,1].ravel()]))
    #print("--")
    #print("step 1 xx yy")
    #print(np.asarray(face_coords).T[0])
    #print(np.asarray(face_coords).T[1])
    xx, yy = np.matmul([[face_matrix[1], face_matrix[2]], [face_matrix[3], face_matrix[4]]], np.asarray(face_coords).T)
    #print("step 2 xx yy")
    #print(xx)
    #print(yy)
    xx = xx + coords[faces[face_matrix[0]][1]][0]
    yy = yy + coords[faces[face_matrix[0]][1]][1]
    #print("CENTER COORD")
    #print(coords[faces[face_matrix[0]][1]])
    #print("Step 3 xx yy")
    #print(xx)
    #print(yy)
    #print("--")
    return np.array([xx,yy]).T

cpdef round_coord(coord):
    return (round(coord[0]), round(coord[1]))

cpdef get_faces_coords(faces, width, height, reference_coords):
    cdef list faces_coords = []
    cdef np.ndarray img
    cdef np.ndarray triangle_cnt
    cdef np.ndarray indices
    for f in range(0, len(faces)):
        img = create_blank(width,height)
        triangle_cnt = np.array([round_coord(tuple(reference_coords[faces[f][0]])), round_coord(tuple(reference_coords[faces[f][1]])), round_coord(tuple(reference_coords[faces[f][2]]))])
        img = cv2.drawContours(img, [triangle_cnt], 0, (255,0,0), -1)
        indices = np.asarray(np.where(img == 255))
        indices = np.delete(indices, 2, 0)
        faces_coords.append(indices.T.tolist())
    return faces_coords


cpdef get_unit_faces_coords(faces_coords, face_reference_objects, coords, faces):
    cdef list unit_faces = []
    for f in range(0, len(faces)):
        unit_faces.append(transform_inverse_multiple(faces_coords[f], face_reference_objects[f], coords, faces).tolist())
    return unit_faces

cpdef get_new_faces_coords(faces_coords, face_reference_objects, coords, faces):
    cdef list new_faces = []
    for f in range(0, len(faces)):
        new_faces.append(transform_multiple(faces_coords[f], face_reference_objects[f], coords, faces).tolist())
    return new_faces

cpdef draw_texture(old_faces_coords, new_faces_coords, reference_texture, virtual_image):
    cdef int new_x
    cdef int old_x
    cdef int new_y
    cdef int old_y
    for i in range(0, len(new_faces_coords)):
        for j in range(0, len(new_faces_coords[i])):
            #if 0 < i < len(old_faces_coords) and 0 < j < len(old_faces_coords[i]) and 0 < i < len(new_faces_coords) and 0 < j < len(new_faces_coords[i]):
            new_x = round(new_faces_coords[i][j][0])
            new_y = round(new_faces_coords[i][j][1])
            old_x = round(old_faces_coords[i][j][0])
            old_y = round(old_faces_coords[i][j][1])
            #if 0 < new_x < virtual_image.shape[1] and 0 < new_y < virtual_image.shape[0] and 0 < old_x < reference_texture.shape[1] and 0 < old_y < reference_texture.shape[0]:
            virtual_image[new_x][new_y] = reference_texture[old_x][old_y]
            print(f'{old_x},{old_y} -> {new_x},{new_y} with {reference_texture[old_x][old_y]}')
    return virtual_image
