# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Humanoid
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################


import numpy as np

import warp as wp
from pxr import Usd, UsdGeom

import os
import trimesh

wp.init()


@wp.kernel
def simulate(positions: wp.array(dtype=wp.vec3),
            velocities: wp.array(dtype=wp.vec3),
            mesh: wp.uint64,
            restitution: float,
            margin: float,
            dt: float):
    
    
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v - v*0.1*dt
    xpred = x + v*dt

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 1.5
    
    if (wp.mesh_query_point(mesh, xpred, max_dist, sign, face_index, face_u, face_v)):
        
        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

        delta = xpred-p
        
        dist = wp.length(delta)*sign
        err = dist - margin

        # mesh collision
        if (err < 0.0):
            n = wp.normalize(delta)*sign
            xpred = xpred - n*err

    # pbd update
    v = (xpred - x)*(1.0/dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v


class HybridSim:

    frame_dt = 1.0 / (60.0)
    # frame_dt = 1.0 / (24.0)

    episode_duration = 3 #5.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 1
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def transfer_to_blender(self, x):
        return [x[0], -x[2], x[1]]

    def transfer_from_blender(self, x):
        return [x[0], x[2], -x[1]]

    def __init__(self, n_balls):
        self.num_particles = n_balls

        self.sim_steps = 500
        self.sim_dt = 1.0/60.0

        self.sim_time = 0.0
        self.sim_timers = {}

        self.sim_restitution = 0.0
        self.sim_margin = 0.05

        mesh_object = trimesh.load_mesh("./extra_data/game_kitchen/kitchen_adjusted_simplified.obj")

        verts = np.array([self.transfer_to_blender(x) for x in mesh_object.vertices])
        # create collision mesh
        self.mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3),
            indices=wp.array(mesh_object.faces.astype(np.int32).flatten(), dtype=int))
        # 0.5 -0.5 0

        # random particles
        x = [0.5, 0.8, 0.5]
        # x = [0., 0.6, 0.]
        init_pos = np.array([self.transfer_to_blender(x)])
        init_vel = np.random.rand(self.num_particles, 3)*0.0

        self.positions = wp.from_numpy(init_pos, dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel, dtype=wp.vec3)


    def call_step(self, c):


        body_qd = self.velocities.numpy() 
        if c == 'W':
            body_qd[0][-3] = 0.5
        elif c == 'S':
            body_qd[0][-3] = -0.5
        elif c == 'D':
            body_qd[0][-2] = -0.5
        elif c == 'A':
            body_qd[0][-2] = 0.5
        elif c == 'Z':
            body_qd[0][-1] = 0.5
        elif c == 'X':
            body_qd[0][-1] = -0.5
        elif c == ' ':
            body_qd[0][-3] = 0.
            body_qd[0][-2] = 0.
            body_qd[0][-1] = 0.


        self.velocities = wp.array(body_qd, dtype=wp.vec3)



        wp.launch(
            kernel=simulate, 
            dim=self.num_particles, 
            inputs=[self.positions, self.velocities, self.mesh.id, self.sim_restitution, self.sim_margin, self.sim_dt])


        return self.transfer_from_blender(self.positions.numpy()[0])



if __name__ == '__main__':
    robot = HybridSim(render=True)
    robot.run()
