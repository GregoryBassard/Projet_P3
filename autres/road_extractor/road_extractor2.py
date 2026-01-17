import csv

file_path = './road_pos_extractor/bin/Debug/net8.0/positions_route.csv'

def get_connectors(name, x, z, rot):
    name = name.lower()
    
    # --- LOGIQUE POUR LES LIGNES DROITES ---
    if "curve" not in name:
        if rot == 0: return (x, z - 16), (x, z + 16) # Entre S, Sort N
        if rot == 1: return (x - 16, z), (x + 16, z) # Entre O, Sort E
        if rot == 2: return (x, z + 16), (x, z - 16) # Entre N, Sort S
        if rot == 3: return (x + 16, z), (x - 16, z) # Entre E, Sort O

    # --- LOGIQUE POUR LES VIRAGES (CURVES) ---
    # Un virage standard à 90° (GTCurve)
    # Exemple pour une courbe qui tourne vers la droite :
    if "curve" in name:
        if rot == 0: return (x, z - 16), (x + 16, z) # Entre S, Sort E
        if rot == 1: return (x - 16, z), (x, z - 16) # Entre O, Sort S
        if rot == 2: return (x, z + 16), (x - 16, z) # Entre N, Sort O
        if rot == 3: return (x + 16, z), (x, z + 16) # Entre E, Sort N
        
    return (x, z), (x, z)

try:
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        all_blocks = list(csv.DictReader(csvfile, delimiter=';'))

    # 1. Trouver le départ
    current_block = next((b for b in all_blocks if "StartLine" in b['Nom']), None)
    
    if not current_block:
        print("Erreur : StartLine non trouvée.")
    else:
        ordered_route = []
        visited_indices = set()
        
        print(f"{'ORDRE':<6} | {'NOM DU BLOC':<30} | {'X':<6} | {'Z':<6}")
        print("-" * 60)

        step = 1
        while current_block:
            ordered_route.append(current_block)
            
            x, z = float(current_block['X']), float(current_block['Z'])
            rot = int(current_block['Rotation'])
            
            print(f"#{step:<5} | {current_block['Nom']:<30} | {x:<6.0f} | {z:<6.0f}")
            
            # Calculer où se trouve la sortie de ce bloc
            _, exit_pt = get_connectors(x, z, rot)
            
            # Chercher le prochain bloc : celui dont l'ENTRÉE correspond à notre SORTIE
            next_block = None
            for i, b in enumerate(all_blocks):
                if b == current_block: continue
                
                bx, bz = float(b['X']), float(b['Z'])
                brot = int(b['Rotation'])
                entry_pt, _ = get_connectors(bx, bz, brot)
                
                # Si la sortie du bloc actuel touche l'entrée d'un autre bloc
                if abs(exit_pt[0] - entry_pt[0]) < 1 and abs(exit_pt[1] - entry_pt[1]) < 1:
                    next_block = b
                    break
            
            if "FinishLine" in current_block['Nom']:
                print("\n Ligne d'arrivée atteinte !")
                break
                
            current_block = next_block
            step += 1
            if step > len(all_blocks): break # Sécurité boucle infinie

except Exception as e:
    print(f"Erreur : {e}")