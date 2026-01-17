import csv

file_path = './road_pos_extractor/bin/Debug/net8.0/positions_route.csv'

try:
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        # On définit le délimiteur point-virgule comme dans le script C#
        reader = csv.DictReader(csvfile, delimiter=';')
        
        print(f"{'NOM DU BLOC':<35} | {'X':<5} | {'Y':<5} | {'Z':<5} | {'ROT'}")
        print("-" * 65)
        
        for row in reader:
            print(f"{row['Nom']:<35} | {row['X']:<5} | {row['Y']:<5} | {row['Z']:<5} | {row['Rotation']}")

except FileNotFoundError:
    print(f"Erreur : Le fichier {file_path} est introuvable. Vérifie le chemin.")
except Exception as e:
    print(f"Une erreur est survenue : {e}")