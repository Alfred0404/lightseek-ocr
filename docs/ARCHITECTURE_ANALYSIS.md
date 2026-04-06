# Analyse Architecturale : Pourquoi LightSeek-OCR ne généralise pas

> Date : 2026-04-05  
> Branche : `feature/decoder`  
> Statut observé : `train_overfit.py` converge (loss → ~0), `train.py` stagne autour de 4.5

---

## Vue d'ensemble du projet

L'objectif est de créer un pipeline **Text → Image → Features visuelles → Text** :

1. Rendre du texte en image (PIL)
2. Extraire des features locales via **SAM** (64×64 tokens)
3. Compresser via **Conv2DCompressor** → (16×16 tokens)
4. Extraire des features globales via **CLIP** frozen
5. Décoder vers le texte original via **GPT-2** partiellement dégelé

Le fait que l'overfit fonctionne prouve que le gradient flow est correct. Le problème est structurel et conceptuel.

---

## Problème 1 — Entraînement circulaire : le modèle mémorise des fingerprints visuels

**Fichiers** : `src/dataset.py`, `src/train/train.py`  
**Sévérité** : CRITIQUE

Le dataset génère du texte aléatoire, le rend en image, extrait les features, et entraîne le modèle à reconstruire ce même texte. Le rendu est **déterministe** : le même texte produit toujours la même image, donc toujours les mêmes features SAM/CLIP.

Le modèle apprend en réalité :

```
visual_features_de_"hello" → "hello"
visual_features_de_"world" → "world"
```

Ce n'est pas de la généralisation, c'est de la **mémorisation d'un dictionnaire visuel**. Si la même chaîne est rendue avec une police différente, une taille différente ou une position différente, les features changent et le modèle échoue.

**Preuve** : `train_overfit.py` mémorise un seul échantillon et atteint loss ≈ 0. `train.py` avec 1000 échantillons ne converge pas parce que chaque text/image est un pattern différent à mémoriser, sans facteur commun exploitable.

---

## Problème 2 — Mismatch train/inférence dans le décodeur

**Fichier** : `src/DeepDecoder.py`, lignes 63–188  
**Sévérité** : CRITIQUE

**Pendant l'entraînement** (forward, L63–118) :

```python
# Le modèle voit : [512 visual tokens] + [N text tokens]
inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
```

**Pendant la génération** (decode, L120–188) :

```python
# Initial forward avec seulement les visual tokens
outputs = self.model(inputs_embeds=visual_embeds, ...)
# Puis génération token par token en autorégressif
```

Le modèle est entraîné avec une séquence `[visuel | texte_cible]` mais génère en mode `[visuel] → token_1 → token_2 → ...`. Les distributions d'attention sont fondamentalement différentes entre ces deux modes. Le modèle apprend à "compléter" du texte vu pendant l'entraînement, pas à "décoder" à partir du visuel seul.

---

## Problème 3 — Signal textuel dilué dans les features visuelles

**Fichier** : `src/DeepEncoder.py`, lignes 84–122  
**Sévérité** : MAJEUR

Le texte est rendu en position fixe `(10, 10)` dans une image **1024×1024**. SAM extrait une grille **64×64 = 4096 tokens** sur toute l'image.

Pour un texte court (1–2 mots à 150pt), le texte occupe une zone d'environ **300×100 pixels**, soit ~3% de l'image. Le reste est de l'espace blanc.

Après compression 16× (Conv2DCompressor) → **16×16 = 256 tokens**, la majorité des tokens représentent du blanc. Le signal réel du texte est dilué et potentiellement écrasé lors de la compression spatiale stride-2.

**Solution évidente non implémentée** : recadrer l'image autour du texte avant l'extraction de features.

---

## Problème 4 — Modèle de langage gelé incompatible avec des features visuelles

**Fichier** : `src/DeepDecoder.py`, lignes 43–77  
**Sévérité** : MAJEUR

Stratégie de gel actuelle :
- SAM, CLIP : **gelés**
- Compresseur, projection canal, projection visuelle : **entraînables**
- GPT-2 : **majoritairement gelé** — seul le dernier bloc + `ln_f` sont entraînables

GPT-2 frozen a été pré-entraîné sur du texte pur. Les couches gelées ont leurs projections Q/K/V calibrées pour des embeddings de tokens textuels. Les features visuelles, même après projection MLP, viennent d'un domaine totalement différent (SAM = segmentation, CLIP = matching image-texte).

Un seul bloc dégelé ne peut pas compenser ce domain gap pour tous les layers en amont. En pratique, les couches gelées de GPT-2 font des calculs d'attention **out-of-distribution** sur les visual tokens, produisant des sorties arbitraires que le seul bloc dégelé doit corriger.

---

## Problème 5 — Interpolation des embeddings positionnels CLIP détruit la structure spatiale

**Fichier** : `src/CLIPVisionProcessor.py`, lignes 86–97  
**Sévérité** : MAJEUR

CLIP attend des tokens dans un ordre spatial 2D. Le code aplatit les features compressées `(B, 768, 16, 16) → (B, 256, 768)` en 1D, puis interpole linéairement les embeddings positionnels de CLIP pour matcher 256 tokens.

**Le problème** : CLIP a des embeddings positionnels 2D appris. Un token en position `[0, 0]` et un token en position `[15, 15]` ont des embeddings positionnels très différents. En aplatissant en 1D et en interpolant, on brise la correspondance entre position spatiale réelle et embedding positionnel. CLIP ne peut plus distinguer "ce token est en haut à gauche" de "ce token est en bas à droite".

---

## Problème 6 — Aucune variation dans les données d'entraînement

**Fichier** : `src/dataset.py`  
**Sévérité** : MAJEUR

Le dataset utilise :
- Une seule police : `arial.ttf`
- Une taille fixe : 150pt
- Une position fixe : `(10, 10)`
- Un fond blanc uniforme
- Aucune augmentation (rotation, flou, bruit, couleur)

En l'absence de variation, le modèle n'apprend aucune invariance. Il ne peut pas généraliser à d'autres fontes, tailles ou mises en page. C'est l'exact opposé de ce qu'un système OCR réel doit faire.

---

## Problème 7 — La fonction de perte n'optimise pas l'encodeur visuel

**Fichier** : `src/train/train.py`, lignes 144–148  
**Sévérité** : MODÉRÉ

```python
visual_padding = torch.full((batch_size, 512), -100, ...)
labels = torch.cat([visual_padding, text_inputs.input_ids], dim=1)
```

La loss cross-entropy est calculée **uniquement** sur les tokens textuels (label -100 = ignoré). Les 512 visual tokens ne reçoivent aucun gradient direct.

Le gradient remonte bien jusqu'au compresseur via backprop à travers les couches GPT-2, mais ce signal est extrêmement indirect et bruité : il passe par le bloc GPT-2 non gelé + `ln_f`, qui ont eux-mêmes du mal à s'adapter (problème 4). L'encodeur visuel ne reçoit donc qu'un signal de gradient très faible et peu informatif.

---

## Problème 8 — Perte d'information dans la compression convolutionnelle

**Fichier** : `src/Conv2DCompressor.py`  
**Sévérité** : MODÉRÉ

```
Conv2d(256→512, k=3, s=2) → Conv2d(512→1024, k=3, s=2)
(B, 256, 64, 64) → (B, 512, 32, 32) → (B, 1024, 16, 16)
```

Chaque token de sortie (16×16) correspond à une zone de **4×4 tokens SAM** (64×64). Deux convolutions stride-2 sans pooling adaptatif : c'est une compression essentiellement spatiale avec peu de fusion sémantique.

Pour du texte (séquence 1D de glyphes), une compression 2D n'est pas le bon inductive bias. Le modèle doit réapprendre qu'un glyphe s'étale sur plusieurs tokens adjacents, sans aucun prior de la nature séquentielle du texte.

---

## Résumé : Pourquoi l'overfit marche mais pas la généralisation

| Comportement | Explication |
|---|---|
| `train_overfit.py` converge (loss → 0) | Le gradient flow est correct. Le modèle mémorise les features d'un seul échantillon. |
| `train.py` stagne à ~4.5 | Pas de pattern généralisant entre les échantillons. Chaque texte est un fingerprint visuel unique non réutilisable. |

**Problème fondamental** : Le modèle est entraîné à faire de la **retrieval** (retrouver le texte dont on a rendu l'image), pas de l'**OCR** (reconnaître des caractères visuels indépendamment du contexte de rendu). Ce n'est pas de la reconnaissance visuelle, c'est de la mémorisation.

---

## Ce qu'il faudrait faire

### Court terme (sans refonte)
1. **Augmentation données** : plusieurs fontes, tailles, positions, rotations, bruit
2. **Recadrage du texte** avant extraction de features (réduire l'espace blanc)
3. **Dégeler plus de GPT-2** (LoRA ou full fine-tuning) pour réduire le domain gap
4. **Corriger le mismatch train/inférence** dans le forward pass du décodeur

### Moyen terme (refonte partielle)
5. Remplacer SAM/CLIP par un encodeur OCR dédié (ex: TrOCR encoder, PaddleOCR backbone)
6. Utiliser des positional embeddings 2D dans CLIP ou ne pas aplatir spatialement
7. Ajouter une loss auxiliaire directement sur les features visuelles (contrastive, reconstruction)

### Long terme (si l'idée vaut le coup)
8. Repenser si le pipeline "texte → image → features → texte" a un sens : un tokenizer classique fait ce travail en O(n) avec une précision parfaite. L'intérêt de ce pipeline n'est justifié que si les images viennent de sources externes (documents scannés, photos).

---

*Rapport généré à partir de l'analyse statique de la codebase. Aucun entraînement n'a été relancé.*
