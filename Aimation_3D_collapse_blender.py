import bpy
import numpy as np
import os
from math import pi, radians

# =======================================================
# 1. CONFIGURAÇÕES E PARÂMETROS GLOBAIS
# =======================================================

# --- Configurações de Arquivo ---
# DATA_FILE_NAME = "scalar_field_Z4c_data_3.npy" 
DATA_FILE_NAME = "scalar_field_data_MS_A008.npy" 
OBJECT_NAME = "ScalarField_Z4c"
THETA_SEGMENTS = 64  

# --- Parâmetros de Cor (Baseados nos seus dados calculados) ---
# Se as cores estiverem fracas, você pode tentar diminuir o range de MIN/MAX
# para aumentar o contraste numérico do degradê.
MIN_PHI_VALUE = -1.446235 
MAX_PHI_VALUE = 1.211280

# --- Parâmetros de Animação e Posição (AJUSTADOS) ---
TARGET_FPS = 30 
# Aproximando o eixo para a borda do domínio físico (r ~= 10)
AXIS_LOC_X = 10.5  # Logo após a borda (r=10)
AXIS_LOC_Y = 0.0   # Centralizado em Y para facilitar o enquadramento


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
    """Remove objetos e materiais anteriores relacionados a simulação."""
    ensure_object_mode()
    prefixes_to_delete = ["ScalarField_Z4c", "GradientMaterial_", "AxisMaterial_", "Z_Axis_Scale", "Z_Tick_", "Z_Label_", "Animation_Camera"]
    for prefix in prefixes_to_delete:
        # Deletar Objetos
        for obj in bpy.data.objects:
            if obj.name.startswith(prefix):
                bpy.data.objects.remove(obj)
        # Deletar Malhas (Meshes)
        for mesh in bpy.data.meshes:
            if mesh.name.startswith(prefix):
                bpy.data.meshes.remove(mesh)
        # Deletar Materiais
        for mat in bpy.data.materials:
            if mat.name.startswith(prefix):
                bpy.data.materials.remove(mat)
        # Deletar Câmeras
        for cam in bpy.data.cameras:
            if cam.name.startswith(prefix):
                bpy.data.cameras.remove(cam)
    print("Cena limpa e pronta para novo setup.")

def ensure_object_mode():
    """Garante que o Blender está no Object Mode."""
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
# CRIAÇÃO DO MATERIAL (GRADIENTE INTENSO) V3 DEFFINITIVO BLENDER 4.5
# =======================================================

def setup_gradient_material(obj, z_min_val, z_max_val):
    """Cria e aplica material com gradiente de cor Intenso (com Emissão) baseado na altura Z."""
    
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
    principled_bsdf.inputs["Roughness"].default_value = 0.1 
    
    # --- AJUSTE DE INTENSIDADE DA COR (CONSERTAR COR FRACA) ---
    # Aumentamos a força da emissão para fazer as cores brilharem (auto-iluminação)
    principled_bsdf.inputs["Emission Strength"].default_value = 2.0 # Ajuste aqui para mais/menos brilho
    # --------------------------------------------------------
    
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
    
    # Configuração das Cores (Aperte o range se quiser mais contraste)
    elements = color_ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (0.0, 0.0, 0.4, 1.0) # Azul (Fundo)
    elements[1].position = 1.0
    elements[1].color = (1.0, 0.8, 0.0, 1.0) # Amarelo (Topo)
    element_mid = elements.new(0.5) 
    element_mid.color = (0.7, 0.7, 0.7, 1.0) # Cinza (Centro)

    # --- CONECTANDO COR INTENSA (CONSERTAR COR FRACA) ---
    # Conecta o degradê na cor base (difusa)
    node_tree.links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])
    
    # Conecta o degradê na cor da emissão (luz própria) - ISSO FAZ A COR APARECER FORTE
    # No Blender 4.5+ o nome é "Emission Color"
    node_tree.links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Emission Color'])
    # ----------------------------------------------------

    print("Material dinâmico intenso configurado.")


# =======================================================
# CRIAÇÃO DO EIXO DE ESCALA Z (QUANTIFICAÇÃO) - CORRIGIDO (BLENDER 4.5+)
# =======================================================

def add_z_axis_scale(z_min, z_max, location_x, location_y):
    """Cria um eixo vertical com marcas de escala e etiquetas de texto (Aproximado)."""
    
    axis_mat = bpy.data.materials.new(name="AxisMaterial_LightGray")
    axis_mat.use_nodes = True
    bsdf = axis_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)

    label_mat = bpy.data.materials.new(name="AxisLabelMaterial_WhiteEmissive")
    label_mat.use_nodes = True
    bsdf_label = label_mat.node_tree.nodes["Principled BSDF"]
    bsdf_label.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1)
    # Correção para Blender 4.5+ (Emission Color)
    bsdf_label.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1)
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
    start_point = round(int(z_min * 2) / 2.0, 1)
    while start_point <= z_max + 0.001: 
        if start_point >= z_min:
             tick_positions.append(round(start_point, 3))
        start_point += 0.5
    tick_positions.append(round(z_min, 3))
    tick_positions.append(round(z_max, 3))
    tick_positions = sorted(list(set(tick_positions)))

    for z_pos in tick_positions:
        # Tick Mark (aproximado da borda)
        bpy.ops.mesh.primitive_cube_add(
            size=1, 
            location=(location_x + 0.15, location_y, z_pos) # Tick apontando pra malha
        )
        tick = bpy.context.object
        tick.name = f"Z_Tick_{z_pos:.2f}"
        tick.scale = (0.1, 0.05, 0.01)
        tick.data.materials.append(axis_mat)

        # Label (afastado do tick)
        bpy.ops.object.text_add(
            location=(location_x + 0.4, location_y, z_pos) # Texto ao lado
        )
        text_obj = bpy.context.object
        text_obj.name = f"Z_Label_{z_pos:.2f}"
        text_obj.data.body = f"{z_pos:.2f}"
        text_obj.data.size = 0.3
        text_obj.rotation_euler = (radians(90), 0, radians(0)) # Em pé, olhando pro centro
        text_obj.data.materials.append(label_mat)
    
    print("Eixo Z de escala aproximado criado.")

# =======================================================
# SETUP DA CÂMERA DE ANIMAÇÃO (CONSERTAR ENQUADRAMENTO)
# =======================================================

def setup_animation_camera():
    """Cria e enquadra uma câmera para a animação cobrindo a malha e o eixo."""
    
    ensure_object_mode()
    
    # Criar nova câmera
    cam_data = bpy.data.cameras.new("Animation_Camera_Data")
    cam_obj = bpy.data.objects.new("Animation_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # Posicionar câmera em um ângulo elevado para perspectiva 3D
    # (Elevada em X+, Y-, Z+ olhando pro centro do sistema)
    cam_obj.location = (20.0, -18.0, 15.0)
    
    # Criar um "Track To" constraint para centralizar a visão entre a malha e o eixo
    # Calculamos o ponto médio aproximado entre a malha (0,0,0) e o eixo (12, 0, ~0)
    # Ponto de foco centralizado em (5, 0, 0)
    
    # Criar um objeto vazio como alvo do foco
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(5.0, 0.0, 0.0))
    focus_target = bpy.context.object
    focus_target.name = "Animation_Camera_Target"
    bpy.context.collection.objects.unlink(focus_target) # Esconder na renderização
    bpy.context.collection.objects.link(focus_target)
    
    # Adicionar o Constraint na câmera
    track_to = cam_obj.constraints.new(type='TRACK_TO')
    track_to.target = focus_target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'
    
    # Tornar a câmera ativa na cena
    bpy.context.scene.camera = cam_obj
    
    print("Câmera de animação configurada e centralizada.")


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

        # 2. Aplicar Material Dinâmico Intenso (CONSERTAR COR FRACA)
        setup_gradient_material(obj_scalar_field, MIN_PHI_VALUE, MAX_PHI_VALUE)

        # 3. Criar Eixo de Escala Z (CONSERTAR EIXO LONGE)
        add_z_axis_scale(MIN_PHI_VALUE, MAX_PHI_VALUE, AXIS_LOC_X, AXIS_LOC_Y)

        # 4. Configurar Timeline
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = TOTAL_FRAMES - 1
        scene.render.fps = TARGET_FPS
        
        # 5. Setup Câmera (CONSERTAR ENQUADRAMENTO DA ANIMAÇÃO)
        setup_animation_camera()

        print("\n*** SETUP COMPLETO COM SUCESSO! ***")
        print(f"Animação configurada de frame 0 a {scene.frame_end} a {TARGET_FPS} FPS.")

    except Exception as e:
        print(f"\n[ERRO FATAL NA EXECUÇÃO DO SCRIPT BLENDER 4.5]: {e}")
        raise e

# Chamar a função principal
main()
