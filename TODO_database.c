#if 0 == 1
/*
Jeu d'images pour l'apprentissage de caractères:
Il nous faudrait au moins une image pour chaque caractere à reconnaitre,
et associer ces images à un caractère unique présent dans la table ASCII.
On aurait:
Ensemble de départ A = ASCII[32( );126(~)] + les lettres accentués.
Ensemble d'arrivée B = ASCII[32( );126(~)].
(on a donc une surjection de A sur B)
On ignorera les accents lors de la reconstruction du texte.
Par exemple, les images représentant les 'é','ê','è' seront considérés comme
des 'e' par le réseau de neurones (plus simple à gérer)
Les images seront de même dimension et le caractère sera centré.
Il faut qu'on définisse une taille pour les images
On binarise l'image en un tableau de flottants à 1 dimension.
On associe l'image binarisée au caractère correspondant
par l'intermediaire de cette structure:
*/
struct TrainingData
{
    float *trainingInputs; // image binarisée: tableau de 0 et 1
    int    res;	// (code ASCII) - 32, on commence à 0
    int   *desiredOutput = indexOutputToVector(res);
       // tableau rempli de 0 avec un seul 1 correspondant au neurone de sortie
};
/*
Une structure TrainingData correspond à une image.
On pourra sauvegarder le tableau de struct TrainingData dans un fichier binaire
avec la fonction:
void buildDataBase(FILE                *f,
                   struct TrainingData  td[],
                   size_t               size_td,
                   size_t               size_inputs,
                   size_t               size_outputs);
utilisé de la manière suivante:
*/

// Initialisation de parametres
FILE *fileTD = fopen("trainingData.bin", "wb");
size_t size_td = |A|; //nombre d'images, taille de l'ensemble A et de td[]
size_t size_inputs = x*y; // taille de l'image en pixels
size_t size_outputs = |B|; // nb de caractères réellement gérés par le réseau

// Allocation de td[]
struct TrainingData *td = malloc(size_td * sizeof (struct TrainingData));

// Remplissage de td[]
td[0].trainingInputs = {0,}; // tableau de 0
td[0].res = 0; // caractere SPACE, 32-32 = 0
td[0].desiredOutput = indexOutputToVector(td[0].res, size_outputs);
// ...
// Plusieurs td[n] pourront avoir le même res !
// caractères concernés: 97(a), 99(c), 101(e), 105(i), 111(o), 117(u)...
// exemple: f((img)e)) = 'e' et f((img)é) = 'e'
// ...
td[size_td - 1].trainingInputs = ...
td[size_td - 1].res = ...
td[size_td - 1].desiredOutput = indexOutputToVector(td[size_td - 1].res,
                                                    size_outputs);
// size_outputs - 1 sera le "res" maximal

buildDataBase(fileTD, td, size_td, size_inputs, size_outputs);

fclose(fileTD);

/*
En résumé, il faut une fonction capable de construire le jeu de données
(tableau de struct TrainingData) depuis un ensemble d'images deja decoupées
et redimensionnées. Ensuite, il faut sauvegarder le jeu de donnees
dans un fichier binaire avec buildDataBase.
*/
#endif
