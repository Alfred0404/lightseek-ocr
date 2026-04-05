# Plan de réparation LightSeek-OCR

> Statut de départ : `train_overfit.py` ✅ converge, `train.py` ❌ stagne à ~4.5

---

## Phase 1 — Bugs bloquants ✅ FAIT

| # | Tâche | Fichier | Statut |
|---|---|---|---|
| 1.1 | Supprimer le bug `image_path = "image.png"` hardcodé | `src/DeepEncoder.py` L117–121 | ✅ |
| 1.2 | Recadrer le texte avant extraction (crop sur pixels non-blancs) | `src/dataset.py` `__getitem__` | ✅ |
| 1.3 | ~~Corriger le mismatch train/inférence dans `decode()`~~ — N/A : le forward original était correct (teacher forcing standard, logits[-1] prédit bien le 1er token texte) | — | ❌ N/A |

**Checkpoint** : Relancer `train_overfit.py` → doit toujours converger. Puis `train.py` 5 epochs → loss doit descendre sous 3.0.

---

## Phase 2 — Augmentation des données

| # | Tâche | Fichier | Statut |
|---|---|---|---|
| 2.1 | Polices variées (arial, times, cour, verdana, calibri) | `src/dataset.py` | ⬜ |
| 2.2 | Taille de police aléatoire (40–180pt) | `src/dataset.py` | ⬜ |
| 2.3 | Position aléatoire (garantir que le texte reste dans l'image) | `src/dataset.py` | ⬜ |
| 2.4 | Couleurs off-white/off-black + flou gaussien léger | `src/dataset.py` | ⬜ |

**Checkpoint** : Vérifier visuellement avec `debug_data.py` que les images générées sont variées.

---

## Phase 3 — Adaptation du modèle

| # | Tâche | Fichier | Statut |
|---|---|---|---|
| 3.1 | Dégeler les 4 derniers blocs GPT-2 (au lieu d'un seul) | `src/train/train.py` L44–77 | ⬜ |
| 3.2 | LR différentiel : 1e-4 pour nouveaux modules, 5e-5 pour GPT-2 | `src/train/train.py` L91–93 | ⬜ |
| 3.3 | Gradient clipping `max_norm=1.0` | `src/train/train.py` avant `optimizer.step()` | ⬜ |
| 3.4 | Corriger le bug batch (`images[0]` traite un seul sample sur 8) | `src/train/train.py` L120–156 | ⬜ |

**Checkpoint** : Vérifier que le nombre de paramètres entraînables est ~20% du total.

---

## Phase 4 — Qualité des features

| # | Tâche | Fichier | Statut |
|---|---|---|---|
| 4.1 | Interpolation 2D bicubique des positional embeddings CLIP (au lieu de 1D linéaire) | `src/CLIPVisionProcessor.py` L86–97 | ⬜ |
| 4.2 | Loss auxiliaire contrastive : cosine distance entre features visuelles poolées et features texte CLIP | `src/train/train.py` boucle intérieure | ⬜ |

**Checkpoint** : La loss auxiliaire doit descendre indépendamment de la loss principale. Vérifier les deux courbes séparément.

---

## Résumé des impacts attendus

| Phase | Impact principal | Risque |
|---|---|---|
| 1 | Active le vrai entraînement — sans ces fixes, rien d'autre ne peut fonctionner | Nul |
| 2 | Brise la mémorisation, force l'invariance visuelle | Faible |
| 3 | Réduit le domain gap visuel/texte dans GPT-2 | VRAM (réduire batch si nécessaire) |
| 4 | Améliore la spatialité CLIP et le gradient vers l'encodeur | Modéré (loss auxiliaire à calibrer) |
