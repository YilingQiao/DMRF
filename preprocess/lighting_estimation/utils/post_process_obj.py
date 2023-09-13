import os
import json

import imageio.v2 as imageio
import numpy as np
import trimesh
from tqdm import tqdm


def tex_coord_to_px(tex_coord, img_width=1024, img_height=1024):
    u, v = tex_coord[0], tex_coord[1]
    px_x = (2 * img_width * u - 1) / 2
    px_y = (2 * img_height * (1 - v) - 2) / 2

    px_x = int(np.clip(px_x, 0, img_width - 1))
    px_y = int(np.clip(px_y, 0, img_height - 1))

    return px_y, px_x


def is_emitter(face_tex_indices, texture_coords, emission_texture, threshold=0.1):
    num_below_threshold = 0
    for tex_index in face_tex_indices:
        tex_coord = texture_coords[tex_index - 1]
        tex_coord_px = tex_coord_to_px(tex_coord)
        # print(tex_coord, tex_coord_px)
        if (emission_texture[tex_coord_px[0], tex_coord_px[1]] < threshold).all():
            num_below_threshold += 1
    return num_below_threshold < 3


def export_lighting_for_hybrid_nerf(scene):
    mesh = trimesh.load(scene.SCENE["local_scene"]["filename"])
    emission_texture = imageio.imread(scene.EMISSION_TEXTURE_FILEPATH)
    output_obj_path = os.path.join(os.path.dirname(scene.SCENE["local_scene"]["filename"]), "mesh.obj")

    with open(scene.SCENE["local_scene"]["filename"], 'r') as obj_file:
        obj_content = obj_file.read()

        vertices = []
        texture_coords = []
        faces = []

        for i, line in enumerate(obj_content.splitlines()):
            if line.startswith('v '):
                vertex = tuple(map(float, line[2:].split()))
                vertices.append(vertex)
            elif line.startswith('vt '):
                tex_coord = tuple(map(float, line[3:].split()))
                texture_coords.append(tex_coord)
            elif line.startswith('f '):
                tri = line[2:].split()
                faces.append(
                    {"vertex_indices": tuple(map(int, [each.split('/')[0] for each in tri])),
                     "tex_coord_indices": tuple(map(int, [each.split('/')[1] for each in tri]))}
                )

        vertices = np.array(vertices)
        texture_coords = np.array(texture_coords)

        # print(vertices.shape)
        # print(texture_coords.shape)
        # print(texture_coords.min(), texture_coords.max())
        # print(face_vertices.max(), face_vertices.min())
        # print(face_tex_coords.max(), face_tex_coords.min())
        # exit()

        # Print out the vertices and their texture coordinates
        # for i, vertex in enumerate(vertices):
        #     texture_coord = texture_coords[i]
        #     print(
        #         f"Vertices {i}: ({vertex[0]}, {vertex[1]}, {vertex[2]}), Texture Coordinates: ({texture_coord[0]}, {texture_coord[1]})")

        # for i, face_vertex_indices in enumerate(faces["vertex_indices"]):
        #     face_tex_coord_indices = faces["tex_coord_indices"][i]
        #     print(
        #         f"Face {i} vertices: ({face_vertex_indices[0]}, {face_vertex_indices[1]}, {face_vertex_indices[2]}), "
        #         f"Face texture Coordinates: ({face_tex_coord_indices[0]}, {face_tex_coord_indices[1]}, {face_tex_coord_indices[2]})")

        # kept_faces_indices = []
        # for i, face in enumerate(tqdm(faces)):
        #     if is_emitter(face["tex_coord_indices"], texture_coords, emission_texture):
        #         kept_faces_indices.append(i)
        # print(mesh.faces.shape)
        # mesh.faces = mesh.faces[kept_faces_indices]
        # print(mesh.faces.shape)

        monosdf_to_nerfstudio_transform = json.load(open(scene.METADATA_PATH))["worldtogt"]
        mesh.vertices = trimesh.transformations.transform_points(mesh.vertices, monosdf_to_nerfstudio_transform)

        mesh.export(output_obj_path, file_type="obj")

        # with open(output_obj_path, 'w') as new_obj_file:
        #     new_obj_file.write(str(mesh.))

        print(f"\nFinished pruning non-emitting faces.\nKept {len(kept_faces_indices)} / {len(faces)} emitter faces.")


def test_is_emitter():
    texture = np.array([[1, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0]])


if __name__ == "__main__":
    test_is_emitter()
