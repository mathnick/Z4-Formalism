import bpy
import numpy as np
import os
from math import pi, radians

# =======================================================
# 1. CONFIGURAÇÕES E PARÂMETROS GLOBAIS
# =======================================================

DATA_FILE_NAME = "scalar_field_Z4c_data_3.npy" 
OBJECT_NAME = "ScalarField_Z4c"
THETA_SEGMENTS = 64  

# Posicionamento do Eixo ajustado para a frente da câmera
AXIS_LOC_X = -10.0  
AXIS_LOC_Y = 3.0   

# =======================================================
# FUNÇÕES DE UTILIDADE E PREPARAÇÃO
# =======================================================

def get_data():
    """Carrega o arquivo .npy."""
    data_dir = bpy.path.abspath('//')
    if data_dir == '//':
        raise FileNotFoundError("ERRO: Salve o arquivo .blend antes de rodar o script.")
    data_path = os.path.join(data_dir, DATA_FILE_NAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"ERRO: Arquivo {DATA_FILE_NAME} não encontrado em {data_path}.")
    try:
        loaded_data = np.load(data_path, allow_pickle=True).item()
        return loaded_data['r_coords'], loaded_data['phi_values']
    except Exception as e:
        raise RuntimeError(f"Falha ao processar arquivo: {e}")

def clean_scene_v2():
    """Remove objetos e materiais da simulação anterior."""
    ensure_object_mode()
    prefixes_to_delete = ["ScalarField_Z4c", "GradientMaterial_", "AxisMaterial_", "Z_Axis_Scale", "Z_Tick_", "Z_Label_"]
    
    for prefix in prefixes_to_delete:
        for obj in bpy.data.objects:
            if obj.name.startswith(prefix):
                bpy.data.objects.remove(obj)
        for mesh in bpy.data.meshes:
            if mesh.name.startswith(prefix):
                bpy.data.meshes.remove(mesh)
        for mat in bpy.data.materials:
            if mat.name.startswith(prefix):
                bpy.data.materials.remove(mat)

def ensure_object_mode():
    """Garante que o Blender está no Object Mode."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT') 

# =======================================================
# CRIAÇÃO DE MALHA E ANIMAÇÃO
# =======================================================

def create_animated_mesh(r_coords, phi_values):
    """Cria o objeto 3D e configura a animação via Shape Keys."""
    TOTAL_FRAMES = phi_values.shape[0]
    NUM_R = len(r_coords)
    verts = []
    faces = []

    for j in range(NUM_R):
        r_val = r_coords[j]
        phi_val = phi_values[0, j]
        for i in range(THETA_SEGMENTS):
            theta = 2 * pi * i / THETA_SEGMENTS
            x = r_val * np.cos(theta)
            y = r_val * np.sin(theta)
            verts.append((x, y, phi_val))

    for j in range(NUM_R - 1): 
        for i in range(THETA_SEGMENTS):
            idx1 = j * THETA_SEGMENTS + i
            idx2 = j * THETA_SEGMENTS + (i + 1) % THETA_SEGMENTS
            idx3 = (j + 1) * THETA_SEGMENTS + (i + 1) % THETA_SEGMENTS
            idx4 = (j + 1) * THETA_SEGMENTS + i
            faces.append((idx1, idx2, idx3, idx4))

    mesh_data = bpy.data.meshes.new(OBJECT_NAME + "_Mesh")
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    ensure_object_mode() 
    
    obj = bpy.data.objects.new(OBJECT_NAME, mesh_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Suaviza a malha
    bpy.ops.object.shade_smooth()

    obj.shape_key_add(name="Base")
    
    print("Criando Shape Keys...")
    for frame in range(1, TOTAL_FRAMES):
        key_name = f"Frame_{frame:04d}"
        shape_key = obj.shape_key_add(name=key_name)
        vert_index = 0
        for j in range(NUM_R):
            phi_val = phi_values[frame, j]
            for i in range(THETA_SEGMENTS):
                vert_co = obj.data.vertices[vert_index].co
                new_co = (vert_co.x, vert_co.y, phi_val)
                shape_key.data[vert_index].co = new_co
                vert_index += 1
        
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame - 1)
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path="value", frame=frame)
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame + 1)
        
    return obj, TOTAL_FRAMES

# =======================================================
# CRIAÇÃO DO MATERIAL (FORÇA MÁXIMA DE COR)
# =======================================================

def setup_gradient_material(obj, z_min_val, z_max_val):
    """Aplica material com emissão forte (auto-iluminado) para as cores pularem na tela."""
    mat_name = f"GradientMaterial_{obj.name}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
    
    if obj.data.materials:
        obj.data.materials.clear() 
    obj.data.materials.append(mat)
    
    mat.use_nodes = True
    node_tree = mat.node_tree
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
        
    material_output = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = 600, 0

    principled_bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = 300, 0
    
    # BRILHO MÁXIMO PARA RESOLVER A COR FRACA
    principled_bsdf.inputs["Roughness"].default_value = 0.5 
    principled_bsdf.inputs["Emission Strength"].default_value = 5.0 # Força Extrema
    
    node_tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    geo_node = node_tree.nodes.new(type='ShaderNodeNewGeometry')
    geo_node.location = -300, 300
    
    separate_xyz = node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = -100, 300
    node_tree.links.new(geo_node.outputs['Position'], separate_xyz.inputs['Vector'])

    range_mapper = node_tree.nodes.new(type='ShaderNodeMapRange')
    range_mapper.location = 100, 300
    range_mapper.inputs['From Min'].default_value = z_min_val  
    range_mapper.inputs['From Max'].default_value = z_max_val  
    range_mapper.inputs['To Min'].default_value = 0.0        
    range_mapper.inputs['To Max'].default_value = 1.0        
    node_tree.links.new(separate_xyz.outputs['Z'], range_mapper.inputs['Value'])

    color_ramp = node_tree.nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = 200, 300
    node_tree.links.new(range_mapper.outputs['Result'], color_ramp.inputs['Fac'])
    
    # Paleta de Cores Científica (Azul escuro -> Ciano -> Amarelo -> Vermelho)
    elements = color_ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.8, 1.0) # Azul
    
    elements[1].position = 1.0
    elements[1].color = (0.8, 0.0, 0.0, 1.0) # Vermelho
    
    element_mid = elements.new(0.5) 
    element_mid.color = (0.0, 0.8, 0.8, 1.0) # Ciano
    
    element_high = elements.new(0.8) 
    element_high.color = (1.0, 0.8, 0.0, 1.0) # Amarelo

    # Conectar ao Base Color e à Emissão
    node_tree.links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])
    if "Emission Color" in principled_bsdf.inputs:
        node_tree.links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Emission Color'])

# =======================================================
# CRIAÇÃO DO EIXO DE ESCALA Z (NOVA POSIÇÃO, ROTAÇÃO E ESPESSURA)
# =======================================================

def add_z_axis_scale(z_min, z_max, location_x, location_y):
    """Cria um eixo vertical com marcas de escala."""
    
    # --- CONTROLES DE TAMANHO E ESPESSURA (AJUSTE AQUI) ---
    ESPESSURA_EIXO = 0.1      # Original era 0.05 (Aumente para engrossar a barra principal)
    COMPRIMENTO_TICK = 0.25   # Original era 0.1 (Aumente para deixar o tracinho mais longo)
    ESPESSURA_TICK = 0.1      # Original era 0.05 (Engrossa o tracinho)
    TAMANHO_TEXTO = 0.5       # Aumente para deixar os números maiores
    AFASTAMENTO_TEXTO = 0.6   # Distância entre o texto e o eixo (aumente se o texto encostar no eixo)
    # ------------------------------------------------------

    axis_mat = bpy.data.materials.new(name="AxisMaterial_LightGray")
    axis_mat.use_nodes = True
    bsdf = axis_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)

    label_mat = bpy.data.materials.new(name="AxisLabelMaterial_WhiteEmissive")
    label_mat.use_nodes = True
    bsdf_label = label_mat.node_tree.nodes["Principled BSDF"]
    bsdf_label.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1)
    bsdf_label.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1)
    bsdf_label.inputs["Emission Strength"].default_value = 2.0 
    
    ensure_object_mode()
    
    axis_height = z_max - z_min
    if axis_height == 0: axis_height = 0.1 
    
    # 1. EIXO PRINCIPAL
    bpy.ops.mesh.primitive_cube_add(
        size=1, 
        location=(location_x, location_y, z_min + axis_height / 2)
    )
    main_axis = bpy.context.object
    main_axis.name = "Z_Axis_Scale"
    # Aplica a espessura no eixo X e Y (mantendo a altura Z)
    main_axis.scale = (ESPESSURA_EIXO, ESPESSURA_EIXO, axis_height / 2)
    main_axis.data.materials.append(axis_mat)

    step = axis_height / 4.0
    tick_positions = [z_min + step * i for i in range(5)]

    for z_pos in tick_positions:
        # 2. MARCAS (TICKS)
        bpy.ops.mesh.primitive_cube_add(
            size=1, 
            location=(location_x - (COMPRIMENTO_TICK/2 + ESPESSURA_EIXO/2), location_y, z_pos) 
        )
        tick = bpy.context.object
        tick.name = f"Z_Tick_{z_pos:.4f}"
        # Aplica os controles no tick
        tick.scale = (COMPRIMENTO_TICK, ESPESSURA_TICK, 0.02)
        tick.data.materials.append(axis_mat)

        # 3. TEXTO DOS NÚMEROS
        bpy.ops.object.text_add(
            location=(location_x - AFASTAMENTO_TEXTO, location_y, z_pos - (TAMANHO_TEXTO/3)) 
        )
        text_obj = bpy.context.object
        text_obj.name = f"Z_Label_{z_pos:.4f}"
        text_obj.data.body = f"{z_pos:.4f}"
        
        # Aplica o tamanho da fonte
        text_obj.data.size = TAMANHO_TEXTO 
        
        text_obj.rotation_euler = (radians(90), 0, radians(-45)) 
        text_obj.data.materials.append(label_mat)

# =======================================================
# EXECUÇÃO PRINCIPAL
# =======================================================

def main():
    try:
        clean_scene_v2()
        
        r_coords, phi_values = get_data()
        
        # Leitura Dinâmica Real
        dynamic_min = float(np.min(phi_values))
        dynamic_max = float(np.max(phi_values))
        
        obj_scalar_field, TOTAL_FRAMES = create_animated_mesh(r_coords, phi_values)
        setup_gradient_material(obj_scalar_field, dynamic_min, dynamic_max)
        add_z_axis_scale(dynamic_min, dynamic_max, AXIS_LOC_X, AXIS_LOC_Y)

        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = TOTAL_FRAMES - 1
        scene.render.fps = 30
        
        # Força o ambiente 3D a usar Material Preview automaticamente
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'

        print("\n*** SETUP COMPLETO COM SUCESSO! ***")

    except Exception as e:
        print(f"\n[ERRO FATAL NA EXECUÇÃO DO SCRIPT]: {e}")
        raise e

main()
