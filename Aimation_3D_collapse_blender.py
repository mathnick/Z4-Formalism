import bpy
import numpy as np
import os
from math import pi

# =======================================================
# 1. CONFIGURAÇÕES E PARÂMETROS GLOBAIS
# =======================================================

DATA_FILE_NAME = "scalar_field_Z4c_data_3.npy" 
OBJECT_NAME = "ScalarField_Z4c"
THETA_SEGMENTS = 64  

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
    """Remove objetos e materiais da simulação anterior (incluindo eixos antigos)."""
    ensure_object_mode()
    # Mantive as tags do eixo na lista de deleção apenas para limpar o que ficou na sua cena
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
# CRIAÇÃO DO MATERIAL (CORES VIBRANTES + ALTO CONTRASTE)
# =======================================================

def setup_gradient_material(obj, z_min_val, z_max_val):
    """Aplica material com emissão forte e cores vibrantes."""
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
    material_output.location = 800, 0

    principled_bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = 500, 0
    principled_bsdf.inputs["Roughness"].default_value = 0.5 
    principled_bsdf.inputs["Emission Strength"].default_value = 3.0 # Brilho ajustado para o Bloom
    
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
    node_tree.links.new(separate_xyz.outputs['Z'], range_mapper.inputs['Value'])

    color_ramp = node_tree.nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = 300, 300
    node_tree.links.new(range_mapper.outputs['Result'], color_ramp.inputs['Fac'])
    
    # PALETA DE CORES VIBRANTE
    elements = color_ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.4, 1.0) # Azul Escuro (Mínimo)
    
    elements[1].position = 1.0
    elements[1].color = (2.0, 1.0, 0.0, 1.0) # Amarelo Intenso (Pico)
    
    element_mid = elements.new(0.5) 
    element_mid.color = (1.0, 0.0, 0.0, 1.0) # Vermelho Puro (Meio)

    # NÓ DE BRILHO E CONTRASTE
    bright_con = node_tree.nodes.new(type='ShaderNodeBrightContrast')
    bright_con.location = 500, 300
    bright_con.inputs['Contrast'].default_value = 1.5 
    
    node_tree.links.new(color_ramp.outputs['Color'], bright_con.inputs['Color'])
    node_tree.links.new(bright_con.outputs['Color'], principled_bsdf.inputs['Base Color'])
    
    if "Emission Color" in principled_bsdf.inputs:
        node_tree.links.new(bright_con.outputs['Color'], principled_bsdf.inputs['Emission Color'])

# =======================================================
# CONFIGURAÇÃO DE ESTÚDIO (BLOOM, FUNDO ESCURO E WIREFRAME)
# =======================================================

def setup_visual_studio(obj_scalar_field):
    """Aplica as configurações globais de renderização e modificadores para contraste."""
    # 1. Ativar o Efeito Bloom
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.eevee.bloom_intensity = 0.05 
    bpy.context.scene.eevee.bloom_radius = 4.0
    
    # 2. Escurecer o Fundo do Mundo
    world = bpy.context.scene.world
    if world is not None and world.use_nodes:
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0) 
            bg_node.inputs[1].default_value = 1.0 

    # 3. Adicionar Modificador Wireframe
    if "Visual_Wireframe" in obj_scalar_field.modifiers:
        obj_scalar_field.modifiers.remove(obj_scalar_field.modifiers["Visual_Wireframe"])
        
    wire_mod = obj_scalar_field.modifiers.new(name="Visual_Wireframe", type='WIREFRAME')
    wire_mod.thickness = 0.015       
    wire_mod.use_replace = False     
    
    print("Estúdio configurado: Fundo Escuro, Bloom e Wireframe.")

# =======================================================
# EXECUÇÃO PRINCIPAL
# =======================================================

def main():
    try:
        clean_scene_v2()
        
        r_coords, phi_values = get_data()
        
        dynamic_min = float(np.min(phi_values))
        dynamic_max = float(np.max(phi_values))
        
        obj_scalar_field, TOTAL_FRAMES = create_animated_mesh(r_coords, phi_values)
        
        # Aplicar material dinâmico
        setup_gradient_material(obj_scalar_field, dynamic_min, dynamic_max)
        
        # Aplicar visual de estúdio (Bloom, Fundo, Wireframe)
        setup_visual_studio(obj_scalar_field)

        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = TOTAL_FRAMES - 1
        scene.render.fps = 30
        
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
