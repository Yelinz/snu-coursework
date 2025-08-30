"""
    THis file is to visualize "FILTERED" results with final data structure
"""

import time
import sys
import argparse
from pathlib import Path

import numpy as onp
import tyro
from tqdm.auto import tqdm

import json
import trimesh
import viser
import viser.extras
import viser.transforms as tf
import matplotlib.cm as cm  # For colormap

from utils import get_bones, N_JNTS, create_bone_mesh



def get_bone_mesh(bones, skel, color):
    vertices = []
    faces = []
    n_verts = 0
    for part_idx, (u, v) in enumerate(bones):
        if u in skel and v in skel:
            A = skel[u].reshape(3)
            B = skel[v].reshape(3)
            bone_mesh = create_bone_mesh(A, B)
            
            vertices.append(bone_mesh.vertices)
            faces.append(bone_mesh.faces + n_verts)
            n_verts += bone_mesh.vertices.shape[0]
            
    bone_mesh = trimesh.Trimesh(
        vertices=onp.concatenate(vertices, axis=0),
        faces=onp.concatenate(faces, axis=0),
        vertex_colors=color,
    )
    
    return bone_mesh




def main(
    data_path: Path = Path("./demo_tmp/NULL"),
    downsample_factor: int = 1,
    max_frames: int = 1000,
    share: bool = False,
    conf_threshold: float = 1.0,
    foreground_conf_threshold: float = 0.1,
    point_size: float = 0.001,
    camera_frustum_scale: float = 0.02,
    no_mask: bool = False,
    xyzw: bool = True,
    axes_scale: float = 0.25,
    bg_downsample_factor: int = 1,
    init_conf: bool = False,
    cam_thickness: float = 1.5,
    port: int = None,
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported

    if port is not None:
        server = viser.ViserServer(port=port)
    else:
        server = viser.ViserServer()
    if share:
        server.request_share_url()

    server.scene.set_up_direction('-z')

    n_jnts = N_JNTS
    



    with server.gui.add_folder("HW Part2"):
        gui_viz_hw2_1 = server.gui.add_checkbox("Viz HW 2_1 (2-view)", True, disabled=True)
        gui_load_hw2_1 = server.gui.add_button("Load HW 2_1", disabled=False)
        gui_viz_hw2_2 = server.gui.add_checkbox("Viz HW 2_2 (n-view)", True, disabled=True)
        gui_load_hw2_2 = server.gui.add_button("Load HW 2_2", disabled=False)

    
    @gui_viz_hw2_1.on_update
    def _(_) -> None:
        hw2_1_node.visible = gui_viz_hw2_1.value
    
    @gui_viz_hw2_2.on_update
    def _(_) -> None:
        hw2_2_node.visible = gui_viz_hw2_2.value


    hw2_1_node = None
    hw2_2_node = None

    @gui_load_hw2_1.on_click
    def _(_) -> None:
        nonlocal hw2_1_node

        hw2_1_jnts_proposal = onp.load(data_path / 'two_view_triangulated.npy', allow_pickle=True)[()]

        color_rgba = cm.viridis(0)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components
        
        bone_mesh = get_bone_mesh(get_bones(), hw2_1_jnts_proposal, color_rgb)
        
        hw2_1_node = server.scene.add_mesh_trimesh(
            name=f"/hw2/2_1_bones",
            mesh=bone_mesh,
            wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
            position=(0.0, 0.0, 0.0),
        )
        gui_viz_hw2_1.disabled = False
    
    @gui_load_hw2_2.on_click
    def _(_) -> None:
        nonlocal hw2_2_node

        hw2_2_jnts_proposal = onp.load(data_path / 'N_view_triagulated.npy', allow_pickle=True)[()]

        color_rgba = cm.viridis(1)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components
        
        bone_mesh = get_bone_mesh(get_bones(), hw2_2_jnts_proposal, color_rgb)
        
        hw2_2_node = server.scene.add_mesh_trimesh(
            name=f"/hw2/2_2_bones",
            mesh=bone_mesh,
            wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
            position=(0.0, 0.0, 0.0),
        )

        gui_viz_hw2_2.disabled = False
    
    
    
    
    
    with server.gui.add_folder("HW Par3"):
        gui_viz_hw3_1 = server.gui.add_checkbox("Viz HW 3_1 (2-view)", True, disabled=True)
        gui_load_hw3_1 = server.gui.add_button("Load HW 3_1", disabled=False)
        gui_viz_hw3_2 = server.gui.add_checkbox("Viz HW 3_2 (n-view)", True, disabled=True)
        gui_load_hw3_2 = server.gui.add_button("Load HW 3_2", disabled=False)
    
    hw3_1_node = None
    hw3_2_node = None
    
    bones = get_bones()
    
    @gui_viz_hw3_1.on_update
    def _(_) -> None:
        hw3_1_node.visible = gui_viz_hw3_1.value
    
    @gui_viz_hw3_2.on_update
    def _(_) -> None:
        hw3_2_node.visible = gui_viz_hw3_2.value

    @gui_load_hw3_1.on_click
    def _(_) -> None:
        nonlocal hw3_1_node

        hw3_1_jnts_proposal = onp.load(data_path / 'two_view_brute.npy', allow_pickle=True)[()]
        

        # Add Joints
        jnts = []
        colors = []
        n_person = len(hw3_1_jnts_proposal)
        bone_meshes = []
        for pid in sorted(list(hw3_1_jnts_proposal.keys())):
            if n_person > 1:
                norm_i = pid / (n_person - 1)
            else:
                norm_i = 0.0
            
            color_rgba = cm.viridis(1)  # Get RGBA color from colormap
            color_rgb = color_rgba[:3]  # Use RGB components
            
            bone_mesh = get_bone_mesh(get_bones(), hw3_1_jnts_proposal[pid], color_rgb)
            bone_meshes.append(bone_mesh)
            
        bone_mesh = trimesh.util.concatenate(bone_meshes)
        
        hw3_1_node = server.scene.add_mesh_trimesh(
            name=f"/hw3/3_1_bones",
            mesh=bone_mesh,
            wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
            position=(0.0, 0.0, 0.0),
        )
        gui_viz_hw3_1.disabled = False
    
    @gui_load_hw3_2.on_click
    def _(_) -> None:
        nonlocal hw3_2_node

        hw3_2_jnts_proposal = onp.load(data_path / 'N_view_brute.npy', allow_pickle=True)[()]

        # Add Joints
        jnts = []
        colors = []
        n_person = len(hw3_2_jnts_proposal)
        bone_meshes = []
        for pid in sorted(list(hw3_2_jnts_proposal.keys())):
            if n_person > 1:
                norm_i = pid / (n_person - 1)
            else:
                norm_i = 0.0
            
            color_rgba = cm.viridis(1)  # Get RGBA color from colormap
            color_rgb = color_rgba[:3]  # Use RGB components
            
            bone_mesh = get_bone_mesh(get_bones(), hw3_2_jnts_proposal[pid], color_rgb)
            bone_meshes.append(bone_mesh)
            
        bone_mesh = trimesh.util.concatenate(bone_meshes)
        
        hw3_2_node = server.scene.add_mesh_trimesh(
            name=f"/hw3/3_2_bones",
            mesh=bone_mesh,
            wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
            position=(0.0, 0.0, 0.0),
        )

        gui_viz_hw3_2.disabled = False
    
    
    


    # Add playback UI.
    with server.gui.add_folder("Joints"):
        gui_show_joints = server.gui.add_checkbox("Turn On Joints Viz", True, disabled=True)
        gui_jnts = server.gui.add_slider(
            "Joints",
            min=0,
            max=n_jnts - 1,
            step=1,
            initial_value=0,
            disabled=False,
        )
        gui_next_jnts = server.gui.add_button("Next Jnt", disabled=True)
        gui_prev_jnts = server.gui.add_button("Prev Jnt", disabled=True)
        gui_show_all_joints = server.gui.add_checkbox("Show all Joints", False, disabled=True)
        gui_load_jnts = server.gui.add_button("Load Joints", disabled=False)
        
        

    # Frame step buttons.
    @gui_next_jnts.on_click
    def _(_) -> None:
        gui_jnts.value = (gui_jnts.value + 1) % n_jnts

    @gui_prev_jnts.on_click
    def _(_) -> None:
        gui_jnts.value = (gui_jnts.value - 1) % n_jnts

    # @gui_show_smpl.on_update
    # def _(_) -> None:
    #     smplx_node.visible = gui_show_smpl.value

    @gui_show_joints.on_update
    def _(_) -> None:
        nonlocal prev_jnt
        if gui_show_joints.value:
            gui_jnts.disabled = False
            gui_next_jnts.disabled = False
            gui_prev_jnts.disabled = False
            gui_show_all_joints.disabled = False
            
            current_jnt = gui_jnts.value
            if not gui_show_all_joints.value:
                with server.atomic():
                    frame_nodes[current_jnt].visible = True
                    frame_nodes[prev_jnt].visible = False
            else:
                with server.atomic():
                    for i, frame_node in enumerate(frame_nodes):
                        frame_node.visible = True

        else:
            gui_jnts.disabled = True
            gui_next_jnts.disabled = True
            gui_prev_jnts.disabled = True
            gui_show_all_joints.disabled = True
            
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = False

        server.flush()  # Optional!


    # Toggle frame visibility when the timestep slider changes.
    @gui_jnts.on_update
    def _(_) -> None:
        nonlocal prev_jnt
        current_jnt = gui_jnts.value
        if not gui_show_all_joints.value:
            with server.atomic():
                frame_nodes[current_jnt].visible = True
                frame_nodes[prev_jnt].visible = False
        prev_jnt = current_jnt
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_joints.on_update
    def _(_) -> None:
        if gui_show_all_joints.value:
            # Show frames with stride
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = True
            # Disable playback controls
            gui_jnts.disabled = True
            gui_next_jnts.disabled = True
            gui_prev_jnts.disabled = True
        else:
            # Show only the current frame
            current_jnt = gui_jnts.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = i == current_jnt
            # Re-enable playback controls
            gui_jnts.disabled = False
            gui_next_jnts.disabled = False
            gui_prev_jnts.disabled = False   
            

    @gui_load_jnts.on_click
    def _(_) -> None:
        nonlocal frame_nodes

        if True:
            # Add Bones
            
            jnts_proposal = onp.load(data_path / 'panoptic' / "node_proposals.npy", allow_pickle=True)[()]


            # Add Joints
            for i in sorted(list(jnts_proposal.keys())):
                jnt = jnts_proposal[i]
                jnt = jnt.reshape(-1, 3)
                jnt = onp.array(jnt, dtype=onp.float32)

                if n_jnts > 1:
                    norm_i = i / (n_jnts - 1)
                else:
                    norm_i = 0.0
                color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
                color_rgb = color_rgba[:3]  # Use RGB components

                jnt_node = server.scene.add_point_cloud(
                    f"/joints/f{i}",
                    points=jnt,
                    point_size=0.03,
                    colors=color_rgb,
                )
                frame_nodes.append(jnt_node)

                
            # Initialize frame visibility.
            for i, frame_node in enumerate(frame_nodes):
                if gui_show_all_joints.value:
                    frame_node.visible = True
                else:
                    frame_node.visible = i == gui_jnts.value
            prev_jnt = gui_bones.value

            gui_show_joints.disabled = False
            gui_jnts.disabled = False
            gui_next_jnts.disabled = False
            gui_prev_jnts.disabled = False
            gui_show_all_joints.disabled = False
            
            gui_show_joints.value = True

 


    n_bones = len(get_bones())
    with server.gui.add_folder("Bones"):
        gui_show_bones = server.gui.add_checkbox("Turn On Bone Viz", True, disabled=True)
        gui_bones = server.gui.add_slider(
            "Bones",
            min=0,
            max=n_bones - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_bone_viz_thrs = server.gui.add_slider(
            "Bone Viz Thrs",
            min=0,
            max=1,
            step=0.01,
            initial_value=0.05,
            disabled=False,
        )
        
        gui_next_bones = server.gui.add_button("Next Bone", disabled=True)
        gui_prev_bones = server.gui.add_button("Prev Bone", disabled=True)
        gui_show_all_bones = server.gui.add_checkbox("Show all Bones", False, disabled=True) 
        gui_load_bones = server.gui.add_button("Load Bones", disabled=False)

    # Frame step buttons.
    @gui_next_bones.on_click
    def _(_) -> None:
        gui_bones.value = (gui_bones.value + 1) % n_bones

    @gui_prev_bones.on_click
    def _(_) -> None:
        gui_bones.value = (gui_bones.value - 1) % n_bones
        
    # Toggle frame visibility when the timestep slider changes.
    @gui_bones.on_update
    def _(_) -> None:
        nonlocal prev_bone
        nonlocal bone_loaded
        current_bone = gui_bones.value
        if bone_loaded:
            with server.atomic():
                bone_nodes[current_bone].visible = True
                bone_nodes[prev_bone].visible = False
        prev_bone = current_bone
        server.flush()  # Optional!

    @gui_show_bones.on_update
    def _(_) -> None:
        nonlocal prev_bone
        nonlocal bone_loaded
        if bone_loaded:
            if gui_show_bones.value:
                gui_bones.disabled = False
                gui_next_bones.disabled = False
                gui_prev_bones.disabled = False
                gui_show_all_bones.disabled = False
                
                current_bone = gui_bones.value
                if not gui_show_all_bones.value:
                    with server.atomic():
                        bone_nodes[current_bone].visible = True
                        bone_nodes[prev_bone].visible = False
                else:
                    with server.atomic():
                        for i, bone_node in enumerate(bone_nodes):
                            bone_node.visible = True

            else:
                gui_bones.disabled = True
                gui_next_bones.disabled = True
                gui_prev_bones.disabled = True
                gui_show_all_bones.disabled = True
                
                with server.atomic():
                    for i, bone_node in enumerate(bone_nodes):
                        bone_node.visible = False
        server.flush()  # Optional!
        

    @gui_show_all_bones.on_update
    def _(_) -> None:
        if gui_show_all_bones.value:
            # Show frames with stride
            with server.atomic():
                for i, frame_node in enumerate(bone_nodes):
                    frame_node.visible = True
            # Disable playback controls
            gui_bones.disabled = True
            gui_next_bones.disabled = True
            gui_prev_bones.disabled = True
        else:
            # Show only the current frame
            current_bone = gui_bones.value
            with server.atomic():
                for i, frame_node in enumerate(bone_nodes):
                    frame_node.visible = i == current_bone
            # Re-enable playback controls
            gui_bones.disabled = False
            gui_next_bones.disabled = False
            gui_prev_bones.disabled = False   

    
    bone_loaded = False
    @gui_load_bones.on_click
    def _(_) -> None:
        nonlocal bone_loaded
        nonlocal bone_nodes

        if True:
            # Add Bones
            part_proposals = onp.load(data_path / 'panoptic' / "viz_part_proposals.npy", allow_pickle=True)[()]
            bone_nodes = []

            for i, part_key in enumerate(sorted(list(part_proposals.keys()))):
                parts = part_proposals[part_key]
                
                vertices = [onp.empty((0, 3), dtype=onp.float32)]
                faces = [onp.empty((0, 3), dtype=onp.int32)]
                n_verts = 0
                for part in parts:
                    part_score = part['score']
                    if part_score < gui_bone_viz_thrs.value:
                        continue
                    
                    vertices.append(part['verts'].astype(onp.float32))
                    faces.append(part['faces'].astype(onp.int32)+n_verts)
                    
                    n_verts += part['verts'].shape[0]
                    
                # merge meshes
                vertices = onp.concatenate(vertices, axis=0)
                faces = onp.concatenate(faces, axis=0)
                
                # Normalize color
                if n_bones > 1:
                    norm_i = i / (n_bones - 1)
                else:
                    norm_i = 0
                color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
                color_rgb = color_rgba[:3]  # Use RGB components
                    
                
                # add bones
                bone_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=color_rgb,
                )
            
                bone_node = server.scene.add_mesh_trimesh(
                    name=f"/bones/f{i}",
                    mesh=bone_mesh,
                    wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
                bone_nodes.append(bone_node)
                
                
            # Initialize frame visibility.
            for i, bone_node in enumerate(bone_nodes):
                if gui_show_all_bones.value:
                    bone_node.visible = True
                else:
                    bone_node.visible = i == gui_bones.value
            prev_bone = gui_bones.value
            bone_loaded = True

            
            gui_show_bones.disabled = False
            gui_bones.disabled = False
            gui_next_bones.disabled = False
            gui_prev_bones.disabled = False
            gui_show_all_bones.disabled = False
            
            gui_show_bones.value = True

                
    # For skeletons
    n_person = 10       # set it as max default
    with server.gui.add_folder("Skeletons"):
        gui_show_skels = server.gui.add_checkbox("Turn On Skels Viz", True, disabled=True)
        gui_skels = server.gui.add_slider(
            "Skeletons ID",
            min=0,
            max=n_person - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        
        gui_next_skels = server.gui.add_button("Next Skeleton", disabled=True)
        gui_prev_skels = server.gui.add_button("Prev Skeleton", disabled=True)
        gui_show_all_skels = server.gui.add_checkbox("Show all Skeletons", False, disabled=True) 
        gui_load_skels = server.gui.add_button("Load Skeletons", disabled=False)
        
    with server.gui.add_folder("Refined Skeletons"):
        gui_show_refined_skels = server.gui.add_checkbox("Show Refined Skeletons", True, disabled=True)
        gui_load_refined_skels = server.gui.add_button("Load Refined Skeletons", disabled=False)

    # Frame step buttons.
    @gui_next_skels.on_click
    def _(_) -> None:
        gui_skels.value = (gui_skels.value + 1) % n_person

    @gui_prev_skels.on_click
    def _(_) -> None:
        gui_skels.value = (gui_skels.value - 1) % n_person
        
    # Toggle frame visibility when the timestep slider changes.
    @gui_skels.on_update
    def _(_) -> None:
        nonlocal prev_skel
        nonlocal skel_loaded
        nonlocal n_person
        current_skel= gui_skels.value % n_person
        if gui_skels.value > n_person - 1:
            gui_skels.value = current_skel
        elif skel_loaded:
            with server.atomic():
                skel_nodes[current_skel].visible = True
                skel_nodes[prev_skel].visible = False
        prev_skel = current_skel
        server.flush()  # Optional!


    @gui_show_skels.on_update
    def _(_) -> None:
        nonlocal prev_skel
        nonlocal skel_loaded
        nonlocal n_person
        if skel_loaded:
            if gui_show_skels.value:
                gui_skels.disabled = False
                gui_next_skels.disabled = False
                gui_prev_skels.disabled = False
                gui_show_all_skels.disabled = False
                
                current_skel= gui_skels.value % n_person
                if not gui_show_all_skels.value:
                    with server.atomic():
                        skel_nodes[current_skel].visible = True
                        skel_nodes[prev_skel].visible = False
                else:
                    with server.atomic():
                        for i, frame_node in enumerate(skel_nodes):
                            frame_node.visible = True

            else:
                gui_skels.disabled = True
                gui_next_skels.disabled = True
                gui_prev_skels.disabled = True
                gui_show_all_skels.disabled = True
                
                with server.atomic():
                    for i, frame_node in enumerate(skel_nodes):
                        frame_node.visible = False
                        
        server.flush()  # Optional!
        

    @gui_show_all_skels.on_update
    def _(_) -> None:
        if gui_show_all_skels.value:
            # Show frames with stride
            with server.atomic():
                for i, frame_node in enumerate(skel_nodes):
                    frame_node.visible = True
            # Disable playback controls
            gui_skels.disabled = True
            gui_next_skels.disabled = True
            gui_prev_skels.disabled = True
        else:
            # Show only the current frame
            current_skel = gui_skels.value
            with server.atomic():
                for i, frame_node in enumerate(skel_nodes):
                    frame_node.visible = i == current_skel
            # Re-enable playback controls
            gui_skels.disabled = False
            gui_next_skels.disabled = False
            gui_prev_skels.disabled = False 

    skel_loaded = False
    @gui_load_skels.on_click
    def _(_) -> None:
        nonlocal skel_loaded
        nonlocal n_person
        nonlocal skel_nodes

        if not skel_loaded or True:
            # Add Bones
            skel_proposals = onp.load(data_path / 'panoptic' / "viz_skel_proposals.npy", allow_pickle=True)[()]
            skel_nodes = []

            # get max skeleton case
            n_person = max([len(parts) for parts in skel_proposals.values()])

            
            for pid in range(n_person):
                # Normalize color
                if n_person > 1:
                    norm_i = pid / (n_person - 1)
                else:
                    norm_i = 0.0
                color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
                color_rgb = color_rgba[:3]  # Use RGB components

                vertices = [onp.empty((0, 3), dtype=onp.float32)]
                faces = [onp.empty((0, 3), dtype=onp.int32)]
                n_verts = 0
                
                for i, part_key in enumerate(sorted(list(skel_proposals.keys()))):
                    parts = skel_proposals[part_key]

                    if pid not in parts:
                        # If the part does not exist for this person, skip
                        continue
                    
                    part = parts[pid]                    
                    vertices.append(part['verts'].astype(onp.float32))
                    faces.append(part['faces'].astype(onp.int32)+n_verts)
                    n_verts += part['verts'].shape[0]
                    
                # merge meshes
                vertices = onp.concatenate(vertices, axis=0)
                faces = onp.concatenate(faces, axis=0)
                

                # add bones
                skel_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=color_rgb,
                )
            
                skel_node = server.scene.add_mesh_trimesh(
                    name=f"/skels/f{pid}",
                    mesh=skel_mesh,
                    wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
                skel_nodes.append(skel_node)
                
                
            # Initialize frame visibility.
            for i, skel_node in enumerate(skel_nodes):
                if gui_show_all_bones.value:
                    skel_node.visible = True
                else:
                    skel_node.visible = i == gui_skels.value
            prev_skel = gui_skels.value
            skel_loaded = True

            gui_show_skels.disabled = False
            gui_skels.disabled = False
            gui_next_skels.disabled = False
            gui_prev_skels.disabled = False
            gui_show_all_skels.disabled = False
            
            gui_show_skels.value = True
            server.flush()  # Optional!

            
        

    @gui_show_refined_skels.on_update
    def _(_) -> None:
        nonlocal refined_skel_loaded
        nonlocal re_skel_nodes
        if refined_skel_loaded:
            for i, skel_node in enumerate(re_skel_nodes):
                skel_node.visible = gui_show_refined_skels.value
        server.flush()  # Optional!


    refined_skel_loaded = False
    @gui_load_refined_skels.on_click
    def _(_) -> None:
        nonlocal refined_skel_loaded
        nonlocal re_skel_nodes

        if not refined_skel_loaded:
            # Add Bones
            skel_proposals = onp.load(data_path / 'panoptic' / "refined_viz_skel_proposals.npy", allow_pickle=True)[()]
            re_skel_nodes = []

            # get max skeleton case
            n_person = max([len(parts) for parts in skel_proposals.values()])

            
            for pid in range(n_person):
                # Normalize color
                if n_person > 1:
                    norm_i = pid / (n_person - 1)
                else:
                    norm_i = 0.0
                color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
                color_rgb = color_rgba[:3]  # Use RGB components

                vertices = [onp.empty((0, 3), dtype=onp.float32)]
                faces = [onp.empty((0, 3), dtype=onp.int32)]
                n_verts = 0
                
                for i, part_key in enumerate(sorted(list(skel_proposals.keys()))):
                    parts = skel_proposals[part_key]

                    if pid not in parts:
                        # If the part does not exist for this person, skip
                        continue
                    
                    part = parts[pid]                    
                    vertices.append(part['verts'].astype(onp.float32))
                    faces.append(part['faces'].astype(onp.int32)+n_verts)
                    n_verts += part['verts'].shape[0]
                    
                # merge meshes
                vertices = onp.concatenate(vertices, axis=0)
                faces = onp.concatenate(faces, axis=0)
                

                # add bones
                skel_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=color_rgb,
                )
            
                skel_node = server.scene.add_mesh_trimesh(
                    name=f"/refined_skels/f{pid}",
                    mesh=skel_mesh,
                    wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
                re_skel_nodes.append(skel_node)
                
                
            # Initialize frame visibility.
            for i, skel_node in enumerate(re_skel_nodes):
                if gui_show_refined_skels.value:
                    skel_node.visible = True
                else:
                    skel_node.visible = False
            refined_skel_loaded = True

            gui_show_refined_skels.disabled = False
            gui_show_refined_skels.value = True
            server.flush()  # Optional!

    
    

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    bone_nodes: list[viser.FrameHandle] = []
    skel_nodes: list[viser.FrameHandle] = []
    re_skel_nodes: list[viser.FrameHandle] = []
    


    if False:
        smpl_path = data_path / "smplx_mesh.obj"
        if not smpl_path.exists():
            print(f"SMPLX mesh path {smpl_path} does not exist. Skipping SMPLX mesh loading.")
            smplx_node = None
        else:
            smplx_obj = trimesh.load(str(smpl_path))
            smplx_vertices = smplx_obj.vertices
            smplx_vertices = onp.array(smplx_vertices, dtype=onp.float32)
            smplx_faces = smplx_obj.faces
            smpl_color = onp.array([[0,255,0]], dtype=onp.uint8)

            smpl_mesh = trimesh.Trimesh(
                vertices=smplx_vertices.astype(onp.float32),
                faces=smplx_faces.astype(onp.int32),
                vertex_colors=smpl_color,
            )
            smplx_node = server.scene.add_mesh_trimesh(
                name=f"/smplx",
                mesh=smpl_mesh,
                wxyz=tf.SO3.from_x_radians(onp.pi * 0).wxyz,
                position=(0.0, 0.0, 0.0),
            )
            
    
    # load cameras
    cam_pth = data_path / "floored_scaled_transform.npz"
    intr_pth = data_path / "undistorted_intrinsics.json"
    
    cam_data = onp.load(cam_pth)
    new_cam_dict = dict()
    w2cs = []

    cid_to_idx = dict()
    for k in sorted(list(cam_data.keys())):
        v = cam_data[k]
        cam_id = int(k)
        c2w = v

        new_cam_dict[cam_id] = dict(
            c2w=c2w,
        )
        cid_to_idx[cam_id] = len(w2cs)
        w2cs.append(v)
    w2cs = onp.stack(w2cs, axis=0).astype(onp.float32)
    
    with open(intr_pth, 'r') as f:
        intrinsic_data = json.load(f)
    
    intrinsics = []
    for k in sorted(list(intrinsic_data.keys())):
        v = intrinsic_data[k]
        cam_id = int(k)
        if cam_id not in new_cam_dict:
            print(f"Camera ID {cam_id} not found in camera data.")
            continue
        new_cam_dict[cam_id]['height'] = v['height']
        new_cam_dict[cam_id]['width'] = v['width']
        intrinsics.append(onp.array(v['Intrinsics'], onp.float32).reshape(3,3))
    Ks = onp.stack(intrinsics, axis=0).reshape(-1, 3, 3)

    T_world_cameras = onp.array(w2cs, onp.float32).reshape(-1, 4, 4)
    T_world_cameras = T_world_cameras.astype(onp.float32)



    # Add camera frustums.
    n_camera = len(T_world_cameras)
    for i in range(n_camera):
        K = Ks[i]
        T_world_camera = T_world_cameras[i]
        cids = sorted(list(new_cam_dict.keys()))
        cid = cids[i]
        cam_height = new_cam_dict[cid]['height']
        cam_width = new_cam_dict[cid]['width']
        
        # Compute color for frustum based on frame index.
        if n_camera > 1:
            norm_i = i / (n_camera - 1)
        else:
            norm_i = 0.0
        color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
        color_rgb = color_rgba[:3]  # Use RGB components

        # Place the frustum with the computed color.
        fov = 2 * onp.arctan2(cam_height / 2, K[0, 0])
        aspect = cam_width / cam_height
        server.scene.add_camera_frustum(
            f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=onp.zeros((1, 1, 3), dtype=onp.uint8),
            wxyz=tf.SO3.from_matrix(T_world_camera[:3, :3]).wxyz,
            position=T_world_camera[:3, 3],
            color=color_rgb,  # Set the color for the frustum
            # thickness=cam_thickness,
        )

        # Add some axes.
        server.scene.add_frame(
            f"/cams/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 40,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    # Add grid
    grid = server.scene.add_grid(
        "grid",
        width=20.0,
        height=20.0,
        cell_size=0.1,
        section_size=1.0,
        position=onp.array([0.0, 0.0, 0.0]),
    )
    

    # Initialize frame visibility.
    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_joints.value:
            frame_node.visible = True
        else:
            frame_node.visible = i == gui_jnts.value


    prev_jnt = gui_jnts.value
    prev_bone = gui_bones.value
    prev_skel = gui_skels.value
    while True:
        time.sleep(1.0 / 30)

    


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process input arguments.")

    # Define arguments
    parser.add_argument(
        "--data",
        type=Path,
        nargs="?",
        default=Path("./demo_tmp/NULL"),
        help="Path to the data"
    )
    parser.add_argument(
        "--conf_thre",
        type=float,
        default=0.1,
        help="Confidence threshold, default is 1.0"
    )
    parser.add_argument(
        "--fg_conf_thre",
        type=float,
        default=0.0,
        help="Foreground confidence threshold, default is 0.1"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.001,
        help="Point size, default is 0.001"
    )
    parser.add_argument(
        "--camera_size",
        type=float,
        default=0.05,
        help="Camera frustum scale, default is 0.02"
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Don't use mask to filter out points",
    )
    parser.add_argument(
        "--wxyz",
        action="store_true",
        help="Use wxyz for SO3 representation",
    )
    parser.add_argument(
        "--axes_scale",
        type=float,
        default=0.1,
        help="Scale of axes",
    )
    parser.add_argument(
        "--bg_downsample",
        type=int,
        default=1,
        help="Background downsample factor",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor",
    )
    parser.add_argument(
        "--init_conf",
        action="store_true",
        help="Share the scene",
    )
    parser.add_argument(
        "--cam_thickness",
        type=float,
        default=2.0,
        help="Camera frustum thickness",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    tyro.cli(main(
        data_path=args.data,
        conf_threshold=args.conf_thre,
        foreground_conf_threshold=args.fg_conf_thre,
        point_size=args.point_size,
        camera_frustum_scale=args.camera_size,
        no_mask=args.no_mask,
        xyzw=not args.wxyz,
        axes_scale=args.axes_scale,
        bg_downsample_factor=args.bg_downsample,
        downsample_factor=args.downsample,
        init_conf=args.init_conf,
        cam_thickness=args.cam_thickness,
        port=args.port,
    ))



