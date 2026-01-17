import csv
import math

file_path = './road_pos_extractor/bin/Debug/net8.0/positions_route.csv'

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_approx_exit(x, z, rot):
    """Estime la zone de sortie vers le bloc suivant"""
    dist = 32 # On cherche le centre du bloc suivant, donc à 32m
    if rot == 0: return (x, z + dist) # Nord
    elif rot == 1: return (x + dist, z) # Est
    elif rot == 2: return (x, z - dist) # Sud
    elif rot == 3: return (x - dist, z) # Ouest
    return (x, z)

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
        
        print(f"{'ORDRE':<6} | {'NOM DU BLOC':<35} | {'X':<6} | {'Z':<6}")
        print("-" * 75)

        step = 1
        while current_block:
            block_id = f"{current_block['X']}_{current_block['Z']}_{current_block['Nom']}"
            visited_indices.add(block_id)
            ordered_route.append(current_block)
            
            x, z = float(current_block['X']), float(current_block['Z'])
            print(f"#{step:<5} | {current_block['Nom']:<35} | {x:<6.0f} | {z:<6.0f}")
            
            if "FinishLine" in current_block['Nom']:
                print("\n🏁 Ligne d'arrivée atteinte !")
                break

            # --- LOGIQUE : LE PLUS PROCHE SANS LIMITE ---
            next_block = None
            min_dist = 999999 # On ouvre les vannes
            
            for b in all_blocks:
                b_id = f"{b['X']}_{b['Z']}_{b['Nom']}"
                if b_id in visited_indices: continue
                
                bx, bz = float(b['X']), float(b['Z'])
                d = get_distance((x, z), (bx, bz))
                
                if d < min_dist:
                    min_dist = d
                    next_block = b
            
            if next_block is None:
                print("\n⚠️ Plus aucun bloc disponible dans la liste.")
                break
                
            current_block = next_block
            step += 1

            current_block = next_block
            step += 1
            if step > 500: break # Sécurité

except Exception as e:
    print(f"Erreur : {e}")