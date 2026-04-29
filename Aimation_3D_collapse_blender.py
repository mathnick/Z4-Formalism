import bpy
import numpy as np
import os
from math import pi

# =======================================================
# 1. CONFIGURAÇÕES E PARÂMETROS GLOBAIS
# =======================================================

# --- Configurações de Arquivo ---
DATA_FILE_NAME = "scalar_field_Z4c_data_3.npy" 
OBJECT_NAME = "ScalarField_Z4c"
THETA_SEGMENTS = 64  

# --- Parâmetros de Cor (Baseados nos seus dados calculados) ---
MIN_PHI_VALUE = -1.446235 
MAX_PHI_VALUE = 1.211280

# --- Parâmetros de Animação e Posição ---
TARGET_FPS = 30 
# Afastado do domínio da malha (r=10) para facilitar visualização
AXIS_LOC_X = 12.0  
AXIS_LOC_Y = -12.0 


# =======================================================
# FUNÇÕES DE UTILIDADE E PREPARAÇÃO V2
# =======================================================

def get_data():
    """Carrega o arquivo .npy usando a detecção de caminho absoluto do Blender."""
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
        raise RuntimeError(f"Falha ao processar arquivo {DATA_FILE_NAME}: {e}")

def clean_scene_v2():
    """Remove objetos e materiais anteriores relacionados a simulação para evitar conflitos."""
    ensure_object_mode()
    
    # Lista de prefixos de nomes a serem deletados
    prefixes_to_delete = ["ScalarField_Z4c", "GradientMaterial_", "AxisMaterial_", "Z_Axis_Scale", "Z_Tick_", "Z_Label_"]
    
    for prefix in prefixes_to_delete:
        # Deletar Objetos
        for obj in bpy.data.objects:
            if obj.name.startswith(prefix):
                bpy.data.objects.remove(obj)
        
        # Deletar Malhas (Meshes) associadas
        for mesh in bpy.data.meshes:
            if mesh.name.startswith(prefix):
                bpy.data.meshes.remove(mesh)
                
        # Deletar Materiais associados (Crucial para limpar shader quebrado)
        for mat in bpy.data.materials:
            if mat.name.startswith(prefix):
                bpy.data.materials.remove(mat)
                
    print("Cena limpa e pronta para novo setup.")

def ensure_object_mode():
    """Garante que o Blender está no Object Mode e limpa a seleção."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT') 


# =======================================================
# CRIAÇÃO DE MALHA E ANIMAÇÃO (SHAPE KEYS)
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
        
        # Inserir keyframes para a transição
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame - 1)
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path="value", frame=frame)
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame + 1)
        
    print(f"Criação de {TOTAL_FRAMES} Shape Keys concluída.")
    return obj, TOTAL_FRAMES


# =======================================================
# CRIAÇÃO DO MATERIAL (GRADIENTE DE COR) - CORRIGIDO V3
# =======================================================

def setup_gradient_material(obj, z_min_val, z_max_val):
    """Cria e aplica o material com gradiente de cor baseado na altura Z em tempo real."""
    
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
        
    # --- Criação dos Nós ---

    material_output = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = 600, 0

    principled_bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = 300, 0
    principled_bsdf.inputs["Roughness"].default_value = 0.1 
    node_tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Nó GEOMETRY (Position)
    geo_node = node_tree.nodes.new(type='ShaderNodeNewGeometry')
    geo_node.location = -300, 300
    
    separate_xyz = node_tree.nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = -100, 300
    node_tree.links.new(geo_node.outputs['Position'], separate_xyz.inputs['Vector'])

    # *** CORREÇÃO AQUI: Usando o nó dedicado ShaderNodeMapRange ***
    range_mapper = node_tree.nodes.new(type='ShaderNodeMapRange')
    range_mapper.location = 100, 300
    # Acessando os inputs pelo nome para garantir que não haja erros de índice na sua versão do Blender
    range_mapper.inputs['From Min'].default_value = z_min_val  
    range_mapper.inputs['From Max'].default_value = z_max_val  
    range_mapper.inputs['To Min'].default_value = 0.0        
    range_mapper.inputs['To Max'].default_value = 1.0        
    node_tree.links.new(separate_xyz.outputs['Z'], range_mapper.inputs['Value'])

    # Color Ramp Node (Gradiente de Cor)
    color_ramp = node_tree.nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = 200, 300
    node_tree.links.new(range_mapper.outputs['Result'], color_ramp.inputs['Fac'])
    
    # --- Configuração Robusta das Cores ---
    elements = color_ramp.color_ramp.elements
    
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.4, 1.0) # Azul escuro (Mínimo)
    
    elements[1].position = 1.0
    elements[1].color = (1.0, 0.8, 0.0, 1.0) # Amarelo Dourado (Máximo)
    
    element_mid = elements.new(0.5) 
    element_mid.color = (0.7, 0.7, 0.7, 1.0) # Cinza Claro (Centro)

    node_tree.links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])

    print("Material dinâmico com gradiente de cor configurado.")

# =======================================================
# CRIAÇÃO DO EIXO DE ESCALA Z (QUANTIFICAÇÃO) - CORRIGIDO (BLENDER 4.0+)
# =======================================================

def add_z_axis_scale(z_min, z_max, location_x, location_y):
    """Cria um eixo vertical com marcas de escala e etiquetas de texto."""
    
    axis_mat = bpy.data.materials.new(name="AxisMaterial_LightGray")
    axis_mat.use_nodes = True
    bsdf = axis_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)

    label_mat = bpy.data.materials.new(name="AxisLabelMaterial_WhiteEmissive")
    label_mat.use_nodes = True
    bsdf_label = label_mat.node_tree.nodes["Principled BSDF"]
    bsdf_label.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1)
    
    # *** CORREÇÃO AQUI: Mudança de "Emission" para "Emission Color" ***
    if "Emission Color" in bsdf_label.inputs:
        bsdf_label.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1)
    elif "Emission" in bsdf_label.inputs: # Fallback para versões mais antigas do Blender
        bsdf_label.inputs["Emission"].default_value = (1.0, 1.0, 1.0, 1)
        
    bsdf_label.inputs["Emission Strength"].default_value = 1.0 
    
    ensure_object_mode()
    
    # 1. Criar o Eixo Principal
    axis_height = z_max - z_min
    bpy.ops.mesh.primitive_cube_add(
        size=1, 
        location=(location_x, location_y, z_min + axis_height / 2)
    )
    main_axis = bpy.context.object
    main_axis.name = "Z_Axis_Scale"
    main_axis.scale = (0.05, 0.05, axis_height / 2)
    main_axis.data.materials.append(axis_mat)

    # 2. Definir Ticks e Etiquetas de Texto
    tick_positions = []
    # Iniciar ponto arredondado para passo de 0.5
    start_point = round(int(z_min * 2) / 2.0, 1)
    while start_point <= z_max + 0.001: 
        if start_point >= z_min:
             tick_positions.append(round(start_point, 3))
        start_point += 0.5
    tick_positions.append(round(z_min, 3))
    tick_positions.append(round(z_max, 3))
    tick_positions = sorted(list(set(tick_positions))) # Remover duplicatas e ordenar

    for z_pos in tick_positions:
        # Marca de Escala (Tick Mark)
        bpy.ops.mesh.primitive_cube_add(
            size=1, 
            location=(location_x + 0.15, location_y, z_pos)
        )
        tick = bpy.context.object
        tick.name = f"Z_Tick_{z_pos:.2f}"
        tick.scale = (0.1, 0.05, 0.01)
        tick.data.materials.append(axis_mat)

        # Etiqueta de Texto (Label)
        bpy.ops.object.text_add(
            location=(location_x + 0.3, location_y + 0.1, z_pos)
        )
        text_obj = bpy.context.object
        text_obj.name = f"Z_Label_{z_pos:.2f}"
        text_obj.data.body = f"{z_pos:.2f}"
        text_obj.data.size = 0.3
        text_obj.rotation_euler = (pi / 2, 0, 0) 
        text_obj.data.materials.append(label_mat)
    
    print("Eixo Z de escala de valores criado com sucesso.")


# =======================================================
# EXECUÇÃO PRINCIPAL
# =======================================================

def main():
    try:
        # Limpar tentativas anteriores conflituosas
        clean_scene_v2()
        
        # Carregar dados
        r_coords, phi_values = get_data()

        # 1. Criar Malha Animada
        obj_scalar_field, TOTAL_FRAMES = create_animated_mesh(r_coords, phi_values)

        # 2. Aplicar Material com Gradiente Dinâmico (Lógica Corrigida)
        setup_gradient_material(obj_scalar_field, MIN_PHI_VALUE, MAX_PHI_VALUE)

        # 3. Criar Eixo de Escala Z (Fora do domínio r=10)
        add_z_axis_scale(MIN_PHI_VALUE, MAX_PHI_VALUE, AXIS_LOC_X, AXIS_LOC_Y)

        # 4. Configurar Timeline
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = TOTAL_FRAMES - 1
        scene.render.fps = TARGET_FPS

        print("\n*** SETUP COMPLETO COM SUCESSO! ***")
        print(f"Animação configurada de frame 0 a {scene.frame_end} a {TARGET_FPS} FPS.")

    except Exception as e:
        # Imprime erro fatal no console interativo e também no console do sistema (Linux)
        print(f"\n[ERRO FATAL NA EXECUÇÃO DO SCRIPT BLENDER]: {e}")
        raise e # Re-lança para que o console exiba o traceback completo

# Chamar a função principal
main()
