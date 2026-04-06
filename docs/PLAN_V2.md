# PLAN_V2 — LightSeek-OCR : Reproduction fidèle DeepSeek-OCR

> Objectif : Passer d'un pipeline GPT-2 / dataset synthétique / tâche de reconstruction  
> à une architecture fidèle au papier DeepSeek-OCR, déployable sur RTX 3070 8GB VRAM.

---

## Architecture cible

```
Document image → SAM ViT 80M → Conv Tokenizer (10×) → CLIP ViT 300M → 256 vision tokens → LLM → Transcription
                  (local feat.)    (compression)          (global feat.)                    (prompt prefix)
```

Le LLM reçoit les vision tokens comme **préfixe de contexte** (visual prompting) et génère la transcription du document. Ce n'est pas de la reconstruction d'embeddings — c'est de la génération conditionnée sur du visuel.

**Gain de contexte** : une page de document = ~2000 tokens texte → 256 vision tokens = 8× moins de contexte consommé.

---

## Ce qu'on conserve / ce qu'on change

### Conservé
- `src/SAMFeatureExtractor.py` — SAM ViT-Base (91M), sort `(B, 256, 64, 64)` ✅
- `src/Conv2DCompressor.py` — compression `64×64 → 16×16`, 256 tokens ✅
- `src/CLIPVisionProcessor.py` — bypass patch embedding, interpolation à corriger en Phase 3
- `src/DeepEncoder.py` — pipeline SAM + Compressor + CLIP, interface `extract_features()` ✅

### Remplacé
- `src/DeepDecoder.py` — GPT-2 → LLM Instruct + LoRA
- `src/dataset.py` — rendu PIL synthétique → vrai dataset OCR
- `src/train/train.py` — objectif reconstruction → objectif transcription

---

## Budget VRAM estimé

| Composant | Modèle | VRAM |
|---|---|---|
| SAM ViT-Base | facebook/sam-vit-base | ~350 MB (gelé) |
| CLIP ViT-Base-P32 | openai/clip-vit-base-patch32 | ~600 MB (gelé) |
| CLIP ViT-Large-P14 | openai/clip-vit-large-patch14 | ~1 700 MB (gelé) |
| SmolLM2-1.7B | HuggingFaceTB/SmolLM2-1.7B-Instruct | ~3 400 MB (fp16 gelé) |
| LoRA + optimiseur | ~5M params entraînables | ~200 MB |
| Activations (batch=1) | seq ~300 tokens | ~500 MB |

**Phase 1+2 (CLIP-Base + SmolLM2) : ~5.1 GB** — confortable sur 8GB.  
**Phase 3 (CLIP-Large + SmolLM2) : ~6.5–7 GB** — faisable en fp16 strict, batch=1.

---

## Phase 1 — Remplacer GPT-2 par un vrai LLM

### 1.1 Choix du LLM

**Modèle retenu : `HuggingFaceTB/SmolLM2-1.7B-Instruct`**

- Entraîné en mode instruction-following → on peut formuler "Transcris ce document."
- 1.7B tient sur 8GB avec LoRA fp16 aux côtés de SAM + CLIP-Base
- Architecture Llama-like (RoPE, GQA), compatible `transformers` sans dépendances custom
- Alternative si VRAM insuffisante : `Qwen/Qwen2-0.5B-Instruct` (~1 GB)

GPT-2 n'a pas été entraîné sur des visual tokens et n'a pas de capacité instruction-following — c'est la racine du problème actuel.

### 1.2 Fine-tuning via LoRA

Pourquoi LoRA plutôt qu'un dégel partiel :
- Dégeler 4 blocs d'un LLM 1.7B = ~200M params dans l'optimiseur → trop coûteux en VRAM
- LoRA (rang r=16) sur Q, K, V, O de chaque bloc = ~5M params entraînables, overhead VRAM minimal
- Les poids de base restent gelés : l'espace textuel préentraîné est préservé

**Config : `peft.LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type=TaskType.CAUSAL_LM)`**

### 1.3 Visual projection

SmolLM2-1.7B a `hidden_size = 2048`. La `visual_projection` actuelle sort 768.  
La remplacer par un MLP `768 → 2048` (ou `1024 → 2048` après Phase 3).

### 1.4 Câblage des vision tokens (Visual Prompting)

```
Séquence LLM = [vision_tokens × 256] + [prompt_tokens] + [transcription_tokens]
Labels        = [   -100 × 256      ] + [ -100 × N_p  ] + [   token_ids cibles  ]
```

Prompt instruction : `"Transcribe the text visible in this document."` (tokenisé par le tokenizer de SmolLM2, labels = -100).

Même logique que l'actuel `train.py` L161-166 (`visual_padding = -100`), mais avec un prompt intermédiaire et le bon LLM.

### 1.5 Fichiers à modifier

| Fichier | Changements |
|---|---|
| `src/DeepDecoder.py` | Remplacer `GPT2LMHeadModel` par `AutoModelForCausalLM` (SmolLM2). Ajouter LoRA via `peft`. Adapter `visual_projection` à `hidden_size=2048`. Adapter `forward()` et `decode()` pour le prompt instruction. |
| `src/LightSeekOCR.py` | Mettre à jour `decoder_name` par défaut. Adapter `predict_from_image()`. |
| `src/train/train.py` | Adapter la construction des labels avec prompt instruction. Adapter le gel des params pour LoRA. |
| `src/train/lora_config.py` | **Créer** — centraliser la `LoraConfig`. |

---

## Phase 2 — Nouveau dataset et objectif d'entraînement

### 2.1 Choix du dataset

**Dataset primaire : SROIE** (`darentang/sroie` sur HuggingFace)
- Images de tickets de caisse scannés, texte imprimé, bon contraste
- Annotations : transcription complète ligne par ligne
- ~1000 images — taille raisonnable pour un premier run
- Disponible sans compte, chargeable via `datasets`

**Dataset secondaire : IAM Handwriting** (`Teklia/IAM-line`)
- Écriture manuscrite anglaise — plus difficile, teste la robustesse du pipeline visuel
- À utiliser après validation sur SROIE

**Conserver `SyntheticOCRDataset`** pour les tests overfit unitaires.

### 2.2 Interface dataset

Même interface que l'actuel : `__getitem__` retourne `(pil_image, transcription_text)`.  
Préprocessing : resize à 1024×1024 avec padding blanc (pas de distorsion), tronquer la transcription à 128 tokens max.

### 2.3 Objectif d'entraînement

La tâche n'est plus "reconstruire le texte rendu" mais "transcrire une vraie image de document". Le modèle apprend à interpréter les vision tokens comme un contexte documentaire et génère le texte en autorégressif.

### 2.4 Fichiers à modifier

| Fichier | Changements |
|---|---|
| `src/dataset.py` | Créer `RealOCRDataset` (SROIE via `datasets`). Conserver `SyntheticOCRDataset`. Ajouter `build_dataset(name, split)`. |
| `src/train/train.py` | Utiliser `RealOCRDataset`. Ajouter `autocast(fp16)`. Construire prompt instruction. Ajouter `num_workers=2` au DataLoader. |
| `src/train/prompt_template.py` | **Créer** — centraliser les templates de prompt (transcription, Q&A pour Phase 4). |

---

## Phase 3 — Upgrade des encodeurs (optionnel, VRAM-dépendant)

### 3.1 Upgrade CLIP

**Modèle cible : `openai/clip-vit-large-patch14`** (~307M, conforme au papier)

- `hidden_size = 1024` au lieu de 768
- Adapter `channel_projection` dans `DeepEncoder.py` : sortie `1024` (ou supprimer si le compresseur sort déjà 1024)
- Adapter `visual_projection` dans `DeepDecoder.py` : entrée `1024 → 2048`
- VRAM : +1.1 GB par rapport à CLIP-Base
- **Vérifier que la VRAM totale reste < 7.5 GB avant d'activer**

### 3.2 Corriger l'interpolation CLIP (positional embeddings)

Problème actuel dans `CLIPVisionProcessor.py` : interpolation linéaire 1D qui brise la structure spatiale 2D de CLIP.

Correction : reshape les patch embeddings en grille 2D `(H_orig, W_orig, D)`, interpoler en **bicubique 2D** vers `(16, 16, D)`, puis aplatir. CLIP-Base : `7×7 → 16×16`. CLIP-Large : `14×14 → 16×16`.

### 3.3 SAM

Conserver `facebook/sam-vit-base` (91M). `sam-vit-huge` (+1.2 GB) dépasserait la VRAM avec CLIP-Large.

### 3.4 Fichiers à modifier

| Fichier | Changements |
|---|---|
| `src/CLIPVisionProcessor.py` | Corriger `_interpolate_pos_embedding` avec bicubique 2D. Rendre `clip_model_name` paramétrable. |
| `src/DeepEncoder.py` | Adapter `channel_projection`. Rendre `clip_model_name` paramétrable. |
| `src/DeepDecoder.py` | Rendre `vision_hidden_size` paramétrable (768 ou 1024). |
| `src/LightSeekOCR.py` | Exposer `clip_model_name` dans le constructeur. |

---

## Phase 4 — Évaluation et métriques OCR

### 4.1 Métriques

**CER (Character Error Rate)** — métrique principale OCR  
`CER = (insertions + suppressions + substitutions) / nb_caractères_référence`  
Librairie : `jiwer.cer(reference, hypothesis)`

**WER (Word Error Rate)** — métrique secondaire  
`WER = édition_mots / nb_mots_référence`

### 4.2 Script d'évaluation

Créer `src/eval/evaluate.py` :
1. Charger un checkpoint
2. Inférence sur le split `test` de SROIE
3. Génération greedy ou beam search (`num_beams=4`)
4. Calculer CER moyen, WER moyen, distribution des erreurs
5. Sauvegarder dans `src/eval/results.json`

### 4.3 Monitoring pendant l'entraînement

Ajouter un calcul CER toutes les 5 époques sur 50 images de validation.  
La loss seule ne suffit pas : un modèle peut avoir une loss basse mais halluciner des tokens fréquents.

### 4.4 Références de performance sur SROIE

| Modèle | CER |
|---|---|
| PaddleOCR | ~1–3% |
| TrOCR-base | ~3–5% |
| **LightSeek-OCR cible (preuve de concept)** | **< 20%** |

### 4.5 Fichiers à créer

| Fichier | Rôle |
|---|---|
| `src/eval/evaluate.py` | Script d'évaluation complet |
| `src/eval/metrics.py` | Wrapper `jiwer` avec normalisation texte (lowercase, ponctuation) |

---

## Séquence d'exécution

```
Phase 1 → train_overfit.py (SmolLM2 doit mémoriser 1 transcription)
         → train.py 10 epochs sur SyntheticOCRDataset (loss doit descendre)
Phase 2 → Intégrer SROIE, relancer 30 epochs, observer CER
Phase 3 → Activer CLIP-Large seulement si Phase 2 atteint CER < 30%
         → Vérifier VRAM < 7.5 GB avant de lancer
Phase 4 → Évaluation formelle sur SROIE test set
```

---

## Dépendances à ajouter dans `requirements.txt`

```
peft          # LoRA pour SmolLM2
datasets      # Chargement SROIE / IAM via HuggingFace
jiwer         # CER / WER
accelerate    # Requis par transformers + peft
```

---

## Résumé des fichiers par priorité

| Priorité | Fichier | Phase | Type |
|---|---|---|---|
| CRITIQUE | `src/DeepDecoder.py` | 1 | Réécriture complète |
| CRITIQUE | `src/dataset.py` | 2 | Extension + nouveau dataset |
| CRITIQUE | `src/train/train.py` | 1+2 | Adaptation majeure |
| HAUTE | `src/LightSeekOCR.py` | 1 | Adaptation câblage |
| HAUTE | `src/train/lora_config.py` | 1 | Création |
| HAUTE | `src/train/prompt_template.py` | 2 | Création |
| MOYENNE | `src/CLIPVisionProcessor.py` | 3 | Correction interpolation |
| MOYENNE | `src/DeepEncoder.py` | 3 | Adaptation dimensions |
| NORMALE | `src/eval/evaluate.py` | 4 | Création |
| NORMALE | `src/eval/metrics.py` | 4 | Création |
