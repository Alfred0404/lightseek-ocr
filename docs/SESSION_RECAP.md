# Session Recap — LightSeek-OCR

> Rédigé pour permettre à une future session de reprendre immédiatement sans relire toute la codebase.

---

## Objectif du projet

Reproduire l'architecture **DeepSeek-OCR** :

```
Document image → SAM ViT 80M → Conv Tokenizer (10×) → CLIP ViT 300M → 256 vision tokens → LLM → Transcription
```

La valeur : compresser une page de document (~2000 tokens texte) en ~256 vision tokens, multipliant la capacité de contexte d'un LLM par ~8×. Les vision tokens servent de **préfixe de contexte** au LLM (visual prompting), pas de substitut d'embeddings texte.

---

## Ce qui a été fait cette session

### 1. Analyse architecturale complète → `ARCHITECTURE_ANALYSIS.md`

8 problèmes identifiés qui empêchaient toute généralisation :
- **Bug critique** : `image_path = "image.png"` hardcodé dans `DeepEncoder.py` L117–120 écrasait silencieusement chaque image générée avec le même fichier statique → **corrigé**
- Entraînement circulaire (texte → PIL → features → même texte → mémorisation, pas généralisation)
- Signal textuel dilué (texte = ~3% de l'image 1024×1024)
- GPT-2 trop gelé, domain gap visuel/texte
- Interpolation 1D des positional embeddings CLIP (doit être 2D bicubique)
- Aucune variation dans les données (une seule police, taille fixe, position fixe)
- Bug batch : `images[0]` utilisait seulement le premier sample de chaque batch

### 2. Corrections appliquées → `PLAN.md` (phases 1–3 du plan intermédiaire)

**`src/DeepEncoder.py`** :
- Suppression des lignes 117–120 (bug `image_path = "image.png"`)

**`src/dataset.py`** :
- Ajout de 8 polices candidates (arial, courier, verdana, calibri, comic, georgia, trebuchet) avec fallback gracieux si absente
- Taille de police aléatoire 40–180pt
- Position aléatoire calculée pour que le texte reste dans l'image
- Fond off-white + texte off-black aléatoires
- Flou gaussien léger 30% du temps
- Crop sur la bounding box du texte + marge 30px + resize → signal textuel passe de ~3% à ~85% de l'image

**`src/train/train.py`** :
- Bug batch corrigé : boucle sur tous les samples du batch (plus `images[0]`)
- 4 blocs GPT-2 dégelés au lieu de 1 (`h[-1], h[-2], h[-3], h[-4]`)
- LR différentiel : 1e-4 pour encoder/visual_projection, 5e-5 pour blocs GPT-2
- Gradient clipping `max_norm=1.0`
- Courbe de loss live : `src/train/training_metrics/loss_curve.png` mise à jour après chaque epoch (backend `Agg`, sauvegarde fichier)
- `BATCH_SIZE=4` (physique), `ACCUMULATION_STEPS=8` (effective batch = 32)

### 3. Résolution CUDA

- Problème : `torch.cuda.is_available()` retournait `False` — torch installé en version CPU-only
- Fix : `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall`
- Résultat : RTX 3070 8GB détectée, CUDA 12.8

### 4. Résultat des runs

- `train_overfit.py` : **loss finale 0.065** ✅ — gradient flow correct
- `train.py` 30 epochs : **loss finale 5.3** ❌ — stagne encore, mais les phases 2+3 du plan intermédiaire venaient d'être appliquées sans re-run

### 5. Pivot architectural → `PLAN_V2.md`

Après discussion, identification que le vrai problème n'est pas le tuning de GPT-2 mais le **mauvais décodeur** :
- GPT-2 est un LM texte-only de 2019, pas un VLM
- Il n'a jamais été entraîné à recevoir des visual tokens
- Le dégeler partiellement compense un défaut architectural fondamental

**Décision : remplacer GPT-2 par SmolLM2-1.7B-Instruct + LoRA**, passer sur de vraies données OCR (SROIE), et aligner l'objectif d'entraînement sur de la transcription réelle.

---

## État actuel des fichiers clés

| Fichier | État |
|---|---|
| `src/DeepEncoder.py` | ✅ Bug image_path supprimé |
| `src/dataset.py` | ✅ Augmentation + crop implémentés |
| `src/train/train.py` | ✅ 4 blocs dégelés, LR différentiel, grad clipping, batch fix, courbe live |
| `src/DeepDecoder.py` | ❌ Toujours GPT-2 — à remplacer (Phase 1 PLAN_V2) |
| `src/CLIPVisionProcessor.py` | ❌ Interpolation 1D à corriger en bicubique 2D (Phase 3 PLAN_V2) |
| `src/LightSeekOCR.py` | ❌ Câblage à adapter au nouveau décodeur (Phase 1 PLAN_V2) |

---

## Ce qu'il reste à faire

Le plan complet est dans **`PLAN_V2.md`**. Résumé :

### Phase 1 — Remplacer GPT-2 par SmolLM2-1.7B-Instruct + LoRA (CRITIQUE)

**`src/DeepDecoder.py`** — réécriture complète :
- `GPT2LMHeadModel` → `AutoModelForCausalLM("HuggingFaceTB/SmolLM2-1.7B-Instruct")`
- `GPT2Tokenizer` → `AutoTokenizer`
- Ajouter LoRA : `peft.get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]))`
- `visual_projection` : MLP `768 → 2048` (hidden_size de SmolLM2 = 2048)
- `forward()` : ajouter le prompt instruction entre vision tokens et transcription cible
- `decode()` : utiliser `model.generate()` de HuggingFace

**`src/train/lora_config.py`** — créer : centraliser la `LoraConfig`

**`src/LightSeekOCR.py`** — adapter le câblage au nouveau décodeur

**`src/train/train.py`** — adapter labels avec prompt instruction, adapter le gel pour LoRA

VRAM estimée Phase 1 : **~5.1 GB** (SAM 350MB + CLIP-Base 600MB + SmolLM2 fp16 3400MB + LoRA 200MB + activations 500MB)

### Phase 2 — Dataset SROIE + objectif transcription (CRITIQUE)

**`src/dataset.py`** :
- Créer `RealOCRDataset` : charge SROIE via `datasets` HuggingFace (`darentang/sroie`)
- Interface identique : `__getitem__` retourne `(pil_image, transcription_text)`
- Resize à 1024×1024 avec padding blanc (pas de distorsion)
- Tronquer transcription à 128 tokens max
- Conserver `SyntheticOCRDataset` pour les tests overfit

**`src/train/prompt_template.py`** — créer : template `"Transcribe the text visible in this document."`

**`src/train/train.py`** :
- Utiliser `RealOCRDataset`
- Ajouter `autocast(fp16)` pour réduire la VRAM
- Construire la séquence : `[vision×256 | prompt_tokens | transcription_tokens]`
- Labels : `[-100×256 | -100×N_prompt | token_ids]`

### Phase 3 — Upgrade encodeurs (MOYENNE, après validation Phase 2)

**`src/CLIPVisionProcessor.py`** :
- Corriger `_interpolate_pos_embedding` : interpolation bicubique 2D
  - CLIP-Base : grille `7×7 → 16×16` ; CLIP-Large : `14×14 → 16×16`
  - Reshape `(num_patches, D)` → `(1, D, H_orig, W_orig)` → `F.interpolate(..., mode="bicubic")` → flatten
- Option upgrade vers `openai/clip-vit-large-patch14` (1024 dims, ~307M — conforme au papier)

**`src/DeepEncoder.py`** :
- Adapter `channel_projection` si CLIP-Large : `1024 → 1024` (ou supprimer)

**`src/DeepDecoder.py`** :
- Rendre `vision_hidden_size` paramétrable (768 ou 1024)

VRAM estimée Phase 3 avec CLIP-Large : **~6.5–7 GB** — vérifier avant de lancer

### Phase 4 — Évaluation CER/WER (NORMALE)

- Créer `src/eval/evaluate.py` et `src/eval/metrics.py`
- Librairie : `jiwer` (`jiwer.cer()`, `jiwer.wer()`)
- Cible réaliste : **CER < 20%** sur SROIE test set (preuve de concept)
- Référence : TrOCR-base = ~3–5% CER

---

## Dépendances à installer avant de commencer

```bash
pip install peft accelerate datasets jiwer
```

---

## Ordre de travail recommandé pour la prochaine session

1. `pip install peft accelerate datasets jiwer`
2. Réécrire `src/DeepDecoder.py` (SmolLM2 + LoRA)
3. Créer `src/train/lora_config.py`
4. Adapter `src/LightSeekOCR.py`
5. Adapter `src/train/train.py` (labels + prompt instruction)
6. Vérifier overfit sur 1 image synthétique (`train_overfit.py`)
7. Créer `RealOCRDataset` dans `src/dataset.py` + `prompt_template.py`
8. Lancer `train.py` 30 epochs sur SROIE, observer CER
9. Si CER < 30% → Phase 3 (CLIP-Large + interpolation bicubique)
10. Phase 4 : évaluation formelle

---

## Infos environnement

- **OS** : Windows 11 Pro
- **GPU** : RTX 3070 8GB VRAM
- **CUDA** : 12.8 (driver 591.59)
- **torch** : 2.11.0+cu128
- **torchvision** : 0.26.0
- **Python** : 3.13
- **Working dir** : `C:\Users\derfl\Documents\Code\Python\lightseek-ocr`
- **Lancer les scripts depuis la racine** : `python src/train/train.py`
