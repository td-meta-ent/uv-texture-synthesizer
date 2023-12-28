# Copyright (c) 2023 Metaverse Entertainment Inc. - All Rights Reserved.

import argparse
import copy
import glob
import time

import cv2
import h5py
import numpy as np
import open3d as o3d

from bilinear_interpolator import BilinearInterpolator


def compute_interpolated_texture_image(images, weights):
    # Initialize the weighted sum array
    weighted_pixel_sum = np.zeros_like(images[0].astype(np.float32))
    weight_sum = np.zeros_like(images[0].astype(np.float32))

    # Get the shape of the images
    height, width, _ = images[0].shape

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Gather weights for this pixel across all images
            pixel_weights = np.array([weight[y, x] for weight in weights])

            # Check if all weights are non-positive
            if np.all(pixel_weights <= 0):
                effective_weights = pixel_weights
            else:
                # Otherwise, ignore negative weights
                effective_weights = np.clip(pixel_weights, 0, None)

            # Compute weighted sum for this pixel
            for img, weight in zip(images, effective_weights):
                weight = weight**2
                weighted_pixel_sum[y, x] += img[y, x] * weight
                weight_sum[y, x] += weight

    for channel in range(3):  # Loop over each color channel (R, G, B)
        channel_mask = (weight_sum[:, :, channel] == 0)
        weighted_pixel_sum[:, :, channel][
            channel_mask] = 255  # Assign white to the channel

    # Normalize the weighted sum by the sum of weights
    weight_sum[weight_sum == 0] = 1  # Avoid division by zero
    interpolated_texture_image = weighted_pixel_sum / weight_sum

    return interpolated_texture_image


def barycentric_coords(point, tri):
    """Calculate the barycentric coordinates of a point with respect to a triangle."""
    # Triangle vertices
    v0, v1, v2 = tri
    # Areas of subtriangles
    a0 = triangle_area(point, v1, v2)
    a1 = triangle_area(point, v0, v2)
    a2 = triangle_area(point, v0, v1)
    total_area = a0 + a1 + a2

    # Check for zero area triangle
    if total_area == 0:
        return 0, 0, 0

    return a0 / total_area, a1 / total_area, a2 / total_area


def map_point_and_interpolate_cosine(point, texture_triangle,
                                     projected_triangle,
                                     texture_triangle_cosine):
    """
    Map a point from the texture triangle to its corresponding position in the projected triangle
    and interpolate cosine values based on barycentric coordinates.
    """
    # Get barycentric coordinates in the texture triangle
    bary_coords = barycentric_coords(point, texture_triangle)

    # Calculate the corresponding point in the projected triangle
    projected_point = (bary_coords[0] * projected_triangle[0][0] +
                       bary_coords[1] * projected_triangle[1][0] +
                       bary_coords[2] * projected_triangle[2][0],
                       bary_coords[0] * projected_triangle[0][1] +
                       bary_coords[1] * projected_triangle[1][1] +
                       bary_coords[2] * projected_triangle[2][1])

    # Interpolate the cosine values based on barycentric coordinates
    point_cosine = (bary_coords[0] * texture_triangle_cosine[0] +
                    bary_coords[1] * texture_triangle_cosine[1] +
                    bary_coords[2] * texture_triangle_cosine[2])

    return projected_point, point_cosine, bary_coords


def is_point_inside_or_on_edge_of_triangle(pt, tri):
    """Check if point pt (x, y) is inside or on the edge of the triangle tri [(x1, y1), (x2, y2), (x3, y3)]."""

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] -
                                                                      p3[1])

    b1 = sign(pt, tri[0], tri[1]) <= 0.0
    b2 = sign(pt, tri[1], tri[2]) <= 0.0
    b3 = sign(pt, tri[2], tri[0]) <= 0.0

    return ((b1 == b2) and (b2 == b3))


def triangle_area(p1, p2, p3):
    """ Calculate the area of a triangle given its vertices """
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] *
                (p1[1] - p2[1])) / 2.0)


def parse_obj_file(file_path):
    vertices = []
    textures = []
    faces = []
    vertex_to_texture = {}  # Mapping from vertex index to texture index

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex position
                parts = line.strip().split()
                vertices.append(
                    [float(parts[1]),
                     float(parts[2]),
                     float(parts[3])])
            elif line.startswith('vt '):  # Texture coordinate
                parts = line.strip().split()
                textures.append([float(parts[1]), float(parts[2])])
            elif line.startswith(
                    'f '):  # Face (could be triangle, quadrilateral, etc.)
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    p_split = p.split('/')
                    vert_index = int(p_split[0]) - 1 if p_split[0] else 0
                    tex_index = int(p_split[1]) - 1 if len(
                        p_split) > 1 and p_split[1] else 0
                    face.append(vert_index)
                    vertex_to_texture[vert_index] = tex_index
                faces.append(face)

    return np.array(vertices), np.array(textures), np.array(
        faces), vertex_to_texture


def change_of_coordinate(points, transformation):

    points = np.array(points)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = points @ transformation.T
    return points[:, :3]


def change_of_coordinate_of_transformation(transformation,
                                           coordinate_change_matrix):
    rot_z_m90_4d = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    coordinate_change_matrix = rot_z_m90_4d
    res = coordinate_change_matrix @ transformation @ np.linalg.inv(
        coordinate_change_matrix)
    return res


def change_of_coordinate_of_camera_matrix_with_padding(camera_matrix, padding):
    if camera_matrix.shape == (3, 3):
        res = copy.deepcopy(camera_matrix)
        res[0][0] = camera_matrix[1][1]
        res[1][1] = camera_matrix[0][0]
        res[0][2] = 5120.0 - camera_matrix[1][2] + padding
        res[1][2] = camera_matrix[0][2]
        return res
    elif camera_matrix.shape == (3, 4):
        res = copy.deepcopy(camera_matrix)
        res[0][0] = camera_matrix[1][1]
        res[1][1] = camera_matrix[0][0]
        res[0][2] = 5120.0 - camera_matrix[1][2] + padding
        res[1][2] = camera_matrix[0][2]
        return res
    else:
        return False


def draw_camera(camera_marix, image_sensor_size, transformation, camera_num,
                camera_colors):
    fx = camera_marix[0][0]
    fy = camera_marix[1][1]
    (px, py) = (camera_marix[0][2], camera_marix[1][2])
    (h, w) = image_sensor_size
    pixel_to_mm = 22.5 / 4096
    pixel_to_m = pixel_to_mm / 1000
    fx *= pixel_to_m
    px *= pixel_to_m
    py *= pixel_to_m
    w *= pixel_to_m
    h *= pixel_to_m

    points = [
        [0, 0, 0],
        [-px, -py, fx],
        [w - px, -py, fx],
        [w - px, h - py, fx],
        [-px, h - py, fx],
    ]

    points = change_of_coordinate(points, transformation)

    lines = [[0, 1], [0, 2], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    colors = [camera_colors[i] for i in range(len(lines))]
    colors[-4] = [1, 0, 0]
    colors[-1] = [0, 1, 0]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def prepare_mesh_and_frame(vertices, m_to_cm):
    rot_z_m90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertices = o3d.utility.Vector3dVector(
        np.matmul(np.asarray(mesh.vertices), rot_z_m90.T))
    mesh.vertices = o3d.utility.Vector3dVector(m_to_cm *
                                               np.asarray(mesh.vertices))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01,
                                                              origin=[0, 0, 0])

    return mesh, frame


def process_camera_data(path_to_image, path_to_camera, number_of_cameras,
                        m_to_cm):
    images = []
    camera_matrices = []
    rectified_camera_matrices = []
    rectifications = []
    transformations_from_0 = []

    rot_z_m90_4d = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    for i in range(number_of_cameras):
        images.append(cv2.imread(path_to_image[i]))
        f = cv2.FileStorage(path_to_camera[i], cv2.FILE_STORAGE_READ)
        camera_matrices.append(f.getNode('camera_matrix_stereo').mat())
        rectified_camera_matrices.append(f.getNode('P').mat())
        rectifications.append(f.getNode('R').mat())
        transformations_from_0.append(f.getNode(f'T_0_{i}').mat())
        f.release()

        rectified_camera_matrices[
            i] = change_of_coordinate_of_camera_matrix_with_padding(
                rectified_camera_matrices[i], 0)
        transformations_from_0[i][0, 3] *= m_to_cm
        transformations_from_0[i][1, 3] *= m_to_cm
        transformations_from_0[i][2, 3] *= m_to_cm
        transformations_from_0[i] = change_of_coordinate_of_transformation(
            transformations_from_0[i], rot_z_m90_4d)

        rect_i = rectifications[i]
        rect_i = np.hstack((rect_i, np.array([[0., 0., 0.]]).T))
        rect_i = np.vstack((rect_i, np.array([0., 0., 0., 1.])))
        rectifications[i] = change_of_coordinate_of_transformation(
            rect_i, rot_z_m90_4d)

    return images, camera_matrices, rectified_camera_matrices, rectifications, transformations_from_0


def compute_projection_indices(vertex, intrinsics, extrinsics):
    f = intrinsics[0, 0]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    transformed_point = change_of_coordinate([vertex], extrinsics)[0]
    z = transformed_point[2]

    if z <= 0:  # Skip points behind the camera
        return None

    fac = f / z
    x = transformed_point[0] * fac + cx
    y = transformed_point[1] * fac + cy

    return x, y


def project_pcd(mesh, intrinsics, extrinsics, image):
    pts = np.asarray(mesh.vertices)

    for p in pts:
        indices = compute_projection_indices(p, intrinsics,
                                             np.linalg.inv(extrinsics))
        if indices is not None:
            x, y = round(indices[0]), round(indices[1])
            (h, w) = image.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                image[y, x] = [0, 255, 0]
    cv2.imshow('Projected Image', image)
    cv2.waitKey(0)
    return None


def display_camera_views_and_reprojection(frame, mesh, number_of_cameras,
                                          rectifications,
                                          transformations_from_0,
                                          rectified_camera_matrices, images,
                                          visualize):
    visualization = [frame, mesh] if visualize else []

    camera_colors = np.zeros(([number_of_cameras, 3]), dtype=np.float32)
    for i in range(number_of_cameras):
        camera_colors[i] = [
            i / number_of_cameras, 1 - i / number_of_cameras,
            i / number_of_cameras
        ]

    for camera_num in range(number_of_cameras):
        t_0_2 = rectifications[8] @ transformations_from_0[8]
        t_0_i = transformations_from_0[camera_num]
        rect_i = rectifications[camera_num]
        transformation = t_0_2 @ np.linalg.inv(t_0_i) @ np.linalg.inv(rect_i)

        if visualize:
            visualization.append(
                draw_camera(rectified_camera_matrices[camera_num],
                            images[camera_num].shape[:2], transformation,
                            camera_num, camera_colors))

            mesh_for_proj = copy.deepcopy(mesh)
            project_pcd(mesh_for_proj, rectified_camera_matrices[camera_num],
                        transformation, images[camera_num])
        else:
            pts = np.asarray(mesh.vertices)
            for p in pts:
                indices = compute_projection_indices(
                    p, rectified_camera_matrices[camera_num], transformation)
                if indices is not None:
                    x, y = round(indices[0]), round(indices[1])
                    (h, w) = images[camera_num].shape[:2]
                    if 0 <= y < h and 0 <= x < w:
                        images[camera_num][y, x] = camera_colors[camera_num]

    if visualize:
        o3d.visualization.draw_geometries(visualization)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        '3D Model Texture Synthesis using Multi-Camera RGB Data and UV Mapping. This tool integrates colors from multiple views into textures for realistic 3D rendering.'
    )
    parser.add_argument('--root_path',
                        type=str,
                        required=True,
                        help='Root path to the data directory')
    parser.add_argument('--project_name',
                        type=str,
                        required=True,
                        help='Name of the project')
    parser.add_argument('--date',
                        type=str,
                        required=True,
                        help='Date of the project data')

    parser.add_argument('--actor_name',
                        type=str,
                        required=True,
                        help='Name of the actor')
    parser.add_argument('--cut_number',
                        type=str,
                        default='00',
                        help='Cut number')
    parser.add_argument('--frame_number',
                        type=str,
                        default='0000',
                        help='Frame number')
    parser.add_argument('--time_stamp',
                        type=str,
                        default='*',
                        help='Timestamp for file selection')
    parser.add_argument('--number_of_cameras',
                        type=int,
                        required=True,
                        help='Number of cameras')
    args = parser.parse_args()

    start_time = time.time()  # Start time measurement

    number_of_cameras = args.number_of_cameras
    ROOT_PATH = args.root_path
    project_name = args.project_name
    date = args.date
    actor_name = args.actor_name
    cut_number = args.cut_number
    frame_number = args.frame_number
    time_stamp = args.time_stamp

    path_to_image = []
    path_to_camera = []
    for cam_num in range(number_of_cameras):
        path_to_image.append(
            glob.glob(
                '{}/rectification/rectification_{}_{}_{}_cut{}_{}_{}_{}.tiff'.
                format(ROOT_PATH, project_name, date, actor_name, cut_number,
                       cam_num, frame_number, time_stamp))[0])
        path_to_camera.append(
            '{}/calibration/rectification_{}_test_{}_{}.yml'.format(
                ROOT_PATH, date, cut_number, cam_num))

    m_to_cm = 1
    images, camera_matrices, rectified_camera_matrices, rectifications, transformations_from_0 = process_camera_data(
        path_to_image, path_to_camera, number_of_cameras, m_to_cm)

    obj_file_path = '{}/mesh/wrap_tri.obj'.format(ROOT_PATH)

    vertices, textures, faces, vertex_to_texture = parse_obj_file(obj_file_path)

    mesh, frame = prepare_mesh_and_frame(vertices, m_to_cm)

    # display_camera_views_and_reprojection(frame,
    #                                       mesh,
    #                                       number_of_cameras,
    #                                       rectifications,
    #                                       transformations_from_0,
    #                                       rectified_camera_matrices,
    #                                       images,
    #                                       visualize=True)

    vertices = mesh.vertices
    camera_normals = [[0, 0, 0] for _ in range(number_of_cameras)]
    cosine_images = [[] for _ in range(number_of_cameras)]
    texture_images = [[] for _ in range(number_of_cameras)]

    # Define the data type for the structured array
    dtype = [('int', '<i4'), ('float1', '<f8'), ('float2', '<f8'),
             ('float3', '<f8')]

    # Create a single element with the desired values
    single_element = np.array((-1, 0.0, 0.0, 0.0), dtype=dtype)

    # Create an empty array
    texture_pixel_info = np.empty((4096, 4096), dtype=dtype)

    # Fill the array
    texture_pixel_info[:] = single_element

    for cam_index in range(number_of_cameras):
        # if cam_index != 4:
        #     continue

        t_0_2 = rectifications[8] @ transformations_from_0[8]
        t_0_i = transformations_from_0[cam_index]
        rect_i = rectifications[cam_index]
        transformation = t_0_2 @ np.linalg.inv(t_0_i) @ np.linalg.inv(rect_i)
        camera_normals[cam_index] = -1 * (
            transformation[:3, :3] @ np.array([0, 0, 1]).reshape(3, 1))

        # Create a blank 5120x5120 white image
        texture_image = np.ones((4096, 4096, 3), dtype=np.uint8) * 255

        projection_triangles = []
        bilinear_interpolator = BilinearInterpolator(images[cam_index])

        # image_directory = "/Users/ethan/Projects/3d-face-reconstruction/cpp/surface-refinement/data/input/v6.0.0/231102/texture"
        # file_name = f"Texture_{project_name}_{date}_{actor_name}_{cam_index}.png"
        # file_path = os.path.join(image_directory, file_name)
        # images[cam_index] = cv2.imread(file_path)
        # cv2.imshow('Texture Image', images[cam_index])
        # cv2.waitKey(0)

        # mesh.vertices = vertices
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Normalize camera normal for the current camera
        cam_normal = camera_normals[cam_index].flatten()
        cam_normal = cam_normal / np.linalg.norm(cam_normal)

        # Compute per-vertex normals and centroid
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        epsilon = 0.01

        front_epsilon = 0.1
        if (cam_index == 4 or cam_index == 5):
            centroid = np.mean(np.asarray(mesh.vertices),
                               axis=0) - front_epsilon * cam_normal
        else:
            centroid = np.mean(np.asarray(mesh.vertices),
                               axis=0) + epsilon * cam_normal

        # Calculate the angle between the normals and the camera direction
        cosines = np.dot(normals, cam_normal)
        angles = np.arccos(cosines)

        # Angle mask: True for vertices where the angle is larger than the filter
        angle_filter = 180 if cam_index == 4 or cam_index == 5 else 90
        angle_mask = angles > np.deg2rad(angle_filter)

        # Compute direction vectors from the centroid to each vertex
        directions = np.asarray(mesh.vertices) - centroid

        # Centroid mask: True for vertices behind the centroid relative to the camera
        centroid_mask = np.array(
            [np.dot(dir, cam_normal) < 0 for dir in directions])

        # Combine masks: True if either condition is met
        combined_mask = np.logical_or(angle_mask, centroid_mask)

        # # Remove vertices by the combined mask
        # mesh.remove_vertices_by_mask(combined_mask)

        # # Clean up the mesh
        # mesh = mesh.remove_degenerate_triangles()
        # mesh = mesh.remove_unreferenced_vertices()
        # mesh.compute_vertex_normals()

        # Create a small sphere at the centroid location for visualization
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        centroid_sphere.translate(centroid)
        centroid_sphere.paint_uniform_color([1, 0,
                                             0])  # Red color for the centroid

        # Add the mesh and the centroid sphere to the visualization
        o3d.visualization.draw_geometries([mesh, centroid_sphere])

        # Mark vertices as visible or not based on the masks
        vertex_visibility = np.logical_not(combined_mask)

        # Initialize a mask for faces
        face_visibility = np.ones(len(mesh.triangles), dtype=bool)

        # Check each face to determine if all its vertices are visible
        for i, face in enumerate(faces):
            # If any vertex of the face is not visible, mark the face as not visible
            if not all(vertex_visibility[vert_idx] for vert_idx in face):
                face_visibility[i] = False

        cosine_image = np.zeros(
            (images[cam_index].shape[0], images[cam_index].shape[1]),
            dtype=np.float32)

        for i, face in enumerate(faces):
            if not face_visibility[i]:
                continue

            projection_triangle = []
            texture_triangle = []
            texture_triangle_cosine = []
            color_sum = np.zeros(3, dtype=int)

            for vertex_index in face:
                vertex = mesh.vertices[vertex_index]
                projection_col_idx, projection_row_idx = compute_projection_indices(
                    vertex, rectified_camera_matrices[cam_index],
                    np.linalg.inv(transformation))
                if projection_col_idx < 0 or projection_col_idx >= images[cam_index].shape[
                    0] or projection_row_idx < 0 or projection_row_idx >= \
                        images[cam_index].shape[1]:
                    continue
                texture_color = images[cam_index][
                    int(projection_row_idx),
                    int(projection_col_idx)].astype(int)
                color_sum += texture_color

                u = textures[vertex_to_texture[vertex_index]][0]
                v = textures[vertex_to_texture[vertex_index]][1]
                texture_row_index = int((1 - v) * texture_image.shape[0])
                texture_col_index = int(u * texture_image.shape[1])
                texture_image[texture_row_index,
                              texture_col_index] = texture_color
                texture_triangle.append((texture_col_index, texture_row_index))
                texture_triangle_cosine.append(cosines[vertex_index])
                projection_triangle.append(
                    (projection_col_idx, projection_row_idx))

            if len(projection_triangle) == 3:
                projection_triangles.append(projection_triangle)

            if len(texture_triangle) == 3:

                current_area = triangle_area(texture_triangle[0],
                                             texture_triangle[1],
                                             texture_triangle[2])

                if current_area < 40000:
                    # Finding all points inside the triangle
                    points_inside_triangle = []

                    # Calculate bounding box of the triangle
                    min_x = min(p[0] for p in texture_triangle)
                    max_x = max(p[0] for p in texture_triangle)
                    min_y = min(p[1] for p in texture_triangle)
                    max_y = max(p[1] for p in texture_triangle)

                    # Iterate over each point in the bounding box
                    for x in range(int(min_x), int(max_x) + 1):
                        for y in range(int(min_y), int(max_y) + 1):
                            if is_point_inside_or_on_edge_of_triangle(
                                (x, y), texture_triangle):
                                points_inside_triangle.append((x, y))

                    # Dictionary to store the projected point of each point
                    texture_projected_map = {}

                    # Get color for each point inside the triangle
                    for point in points_inside_triangle:
                        projected_point, point_cosine, bary_coords = map_point_and_interpolate_cosine(
                            point, texture_triangle, projection_triangle,
                            texture_triangle_cosine)
                        texture_pixel_info[point[1], point[0]]['int'] = i
                        texture_pixel_info[point[1],
                                           point[0]]['float1'] = bary_coords[0]
                        texture_pixel_info[point[1],
                                           point[0]]['float2'] = bary_coords[1]
                        texture_pixel_info[point[1],
                                           point[0]]['float3'] = bary_coords[2]
                        texture_image[point[1], point[
                            0]] = bilinear_interpolator.bilinear_interpolation_at_point(
                                projected_point)
                        cosine_image[point[1], point[0]] = point_cosine

        # log with cam num
        print('finished cam {}'.format(cam_index))

        # Convert the array to a dictionary for YAML serialization
        # texture_pixel_info_dict = {"row_" + str(i): row.tolist() for i, row in enumerate(texture_pixel_info)}
        #
        # # Save the data as a YAML file
        # save_path = '/Users/ethan/Downloads/texture_pixel_info.yml'
        # with open(save_path, 'w') as file:
        #     yaml.dump(texture_pixel_info_dict, file)
        #
        # print(f"Array saved as YAML at: {save_path}")

        # texture_images[cam_index] = texture_image
        # cosine_images[cam_index] = cosine_image
        # # cv2.imshow('Texture Image', texture_image)
        # cv2.imwrite(
        #     "/Users/ethan/Downloads/Texture_{}_{}_{}_{}.png".format(
        #         project_name, date, actor_name, cam_index), texture_image)

        # for triangle in projection_triangles:
        #     cv2.polylines(images[cam_index], [np.array(triangle)],
        #                   isClosed=True,
        #                   color=(0, 0, 255),
        #                   thickness=1)

        # cv2.imshow('Projection Image', images[cam_index])
        # cv2.imwrite(
        #     "/Users/ethan/Downloads/Projection_{}_{}_{}_{}.png".format(
        #         project_name, date, actor_name, cam_index), images[cam_index])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Combine barycentric coordinates into one array
    triangle_indices = texture_pixel_info['int']
    barycentric_coordinates = np.stack(
        (texture_pixel_info['float1'], texture_pixel_info['float2'],
         texture_pixel_info['float3']),
        axis=-1)

    hdf5_file_path = '/Users/ethan/Projects/3d-face-reconstruction/cpp/surface-refinement/data/input/v6.0.0/231102/texture_pixel_info.h5'

    # Create and write to the HDF5 file
    with h5py.File(hdf5_file_path, 'w') as hdf:
        hdf.create_dataset('triangle_index', data=triangle_indices)
        hdf.create_dataset('barycentric_coordinates',
                           data=barycentric_coordinates)

    print(f'Data saved to HDF5 file at: {hdf5_file_path}')

    # save_path = '/Users/ethan/Projects/3d-face-reconstruction/cpp/surface-refinement/data/input/v6.0.0/231102/texture_pixel_info.npy'
    # np.save(save_path, texture_pixel_info)
    #
    # print(f"Array saved at: {save_path}")
    exit(1)

    texture_image = compute_interpolated_texture_image(texture_images,
                                                       cosine_images)

    # save to /Users/ethan/Downloads/Texture_Final.png
    cv2.imwrite('/Users/ethan/Downloads/Texture_Final.png', texture_image)

    end_time = time.time()  # End time measurement
    print('Processing time: {:.2f} seconds'.format(end_time - start_time))
