using GBX.NET;
using GBX.NET.Engines.Game;
using GBX.NET.LZO;
using System;
using System.IO;
using System.Text;

namespace RoadExtractor
{
    class Program
    {
        static void Main(string[] args)
        {
            Gbx.LZO = new MiniLZO();
            string mapPath = @"C:\Users\Greg\Documents\TmForever\Tracks\Challenges\My Challenges\challenge_2.Challenge.Gbx";
            string outputPath = "positions_route.csv";

            try
            {
                var gbx = Gbx.ParseNode<CGameCtnChallenge>(mapPath);
                var csv = new StringBuilder();

                // En-tête du CSV
                csv.AppendLine("Nom;X;Y;Z;Rotation");

                foreach (var block in gbx.Blocks)
                {
                    // On filtre les blocs qui contiennent "Road" ou "Start", "Finish", "Checkpoint"
                    string name = block.Name.ToLower();
                    if (name.Contains("road") || name.Contains("start") || name.Contains("finish") || name.Contains("checkpoint"))
                    {
                        // Calcul du centre de la case
                        float realX = (block.Coord.X * 32f) + 16f;
                        float realY = (block.Coord.Y * 8f);
                        float realZ = (block.Coord.Z * 32f) + 16f;

                        // Ajout de la ligne : Nom; X; Y; Z; Direction (0-3)
                        // (int)block.Direction permet d'exporter 0, 1, 2 ou 3 au lieu de North, East...
                        string line = $"{block.Name};{realX};{realY};{realZ};{(int)block.Direction}";
                        csv.AppendLine(line);
                    }
                }

                File.WriteAllText(outputPath, csv.ToString());
                Console.WriteLine($"Succès ! {outputPath} a été généré avec les positions.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur : {ex.Message}");
            }
        }
    }
}