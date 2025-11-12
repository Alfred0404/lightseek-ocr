# Cahier des Charges / Roadmap : Implémentation du DeepEncoder (SAM + CLIP)

**Objectif :** Créer un module PyTorch `DeepEncoder` qui reproduit l'encodeur de DeepSeek-OCR, en fusionnant un extracteur local (SAM) et un extracteur global (CLIP).

**Composants Clés :**
1.  **Backbone SAM (ViTDet) :** Extracteur de caractéristiques locales haute résolution.
2.  **Compresseur (CNN) :** Sous-échantillonneur spatial (pont).
3.  **Corps de CLIP (ViT) :** Extracteur de caractéristiques globales basse résolution.

---

## Phase 1 : Environnement et Prérequis

- [ ] **Mettre en place l'environnement :**
    - `pip install torch transformers`
    - (Optionnel) `pip install segment-anything` si vous utilisez le dépôt officiel, mais `transformers` est plus simple pour cette tâche.
- [ ] **Valider les modèles :** Choisir les checkpoints à utiliser.
    - **SAM (Exemple) :** `facebook/sam-vit-base-patch16` (Dimensions : `D_sam = 768`)
    - **CLIP (Exemple) :** `openai/clip-vit-large-patch14` (Dimensions : `D_clip = 1024`)

---

## Phase 2 : Étape 1 - Extracteur Local (SAM)

**Objectif :** Charger SAM et le configurer pour qu'il renvoie sa carte de caractéristiques interne (la sortie de l'encodeur).

- [ ] **Charger le ViT de SAM :**
    - Utiliser `SamVisionModel` de `transformers` (plus simple que le dépôt SAM officiel pour la "chirurgie").
    - `sam_model = SamVisionModel.from_pretrained("facebook/sam-vit-base-patch16")`
- [ ] **Tester l'extraction :**
    - Préparer une image "dummy" (ex: `torch.randn(1, 3, 1024, 1024)`).
    - Effectuer un *forward pass* : `outputs = sam_model(dummy_image)`
    - Récupérer la sortie de l'encodeur : `feature_map = outputs.last_hidden_state`
- [ ] **Valider les dimensions :**
    - Vérifier que `feature_map.shape` est `[Batch, 64, 64, 768]`. (Pour une image 1024x1024 avec patchs 16x16, `1024/16 = 64`).
    - *Note :* `transformers` renvoie `(B, H, W, D)`. C'est le format "Channels-Last".

---

## Phase 3 : Étape 2 - Compresseur (CNN)

**Objectif :** Réduire la résolution spatiale de la carte de SAM d'un facteur 16.

- [ ] **Définir la couche `Conv2d` :**
    - `in_channels = D_sam` (768)
    - `out_channels = D_clip` (1024, pour aligner directement sur la dimension de CLIP)
    - `kernel_size = 16`, `stride = 16`
    - `self.compressor = nn.Conv2d(768, 1024, kernel_size=16, stride=16)`
- [ ] **Gérer le format des données (Crucial) :**
    - `nn.Conv2d` attend un format "Channels-First" : `(B, C, H, W)`.
    - La sortie de SAM est "Channels-Last" : `(B, H, W, D)`.
    - **Action :** Il faut "permuter" (permute) les dimensions avant d'appliquer le `Conv2d`.
    - `feature_map = feature_map.permute(0, 3, 1, 2)` (devient `[B, 768, 64, 64]`)
- [ ] **Valider la sortie du compresseur :**
    - `compressed_map = self.compressor(feature_map)`
    - La forme de sortie doit être `[B, 1024, 4, 4]`. (Car `64 / 16 = 4`).

---

## Phase 4 : Étape 3 - Injection Globale (CLIP)

**Objectif :** Injecter la carte compressée dans le "corps" de CLIP, en sautant sa couche d'embedding.

- [ ] **Charger le ViT de CLIP :**
    - `clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")`
    - `D_clip = clip_model.config.hidden_size` (validera 1024)
- [ ] **Isoler les modules de CLIP :**
    - La couche à **sauter** : `clip_model.embeddings`
    - Le corps à **utiliser** : `clip_model.encoder`
    - L'embedding positionnel (à gérer) : `clip_model.embeddings.position_embedding`
- [ ] **Préparer les tokens pour l'injection :**
    - La sortie du CNN est `[B, 1024, 4, 4]`.
    - Le `clip_model.encoder` attend une séquence de tokens : `(B, N, D)`.
    - **Action :** Aplatir (flatten) la dimension spatiale :
    - `tokens = compressed_map.flatten(2)` (devient `[B, 1024, 16]`)
    - `tokens = tokens.permute(0, 2, 1)` (devient `[B, 16, 1024]`, c'est-à-dire `(B, N, D)`)
- [ ] **Gérer les Embeddings Positionnels (Point le plus complexe) :**
    - Le ViT de CLIP a des embeddings positionnels (PE) pré-calculés pour sa grille (ex: `[1, 257, 1024]`, soit 1 token [CLS] + grille 16x16=256).
    - Votre nouvelle grille est de `4x4 = 16` tokens.
    - **Solution :** Vous devez **interpoler en 2D** les PE de CLIP pour qu'ils matchent votre nouvelle grille de `4x4`.
        1.  Extraire les PE d'origine : `original_pe = clip_model.embeddings.position_embedding.weight`
        2.  Retirer le token `[CLS]` (le 1er token) : `original_pe_grid = original_pe[:, 1:, :]`
        3.  Reformer la grille 2D d'origine (16x16 pour CLIP-L/14) : `grid_2d = original_pe_grid.reshape(1, 16, 16, D_clip)`
        4.  Permuter en `(B, C, H, W)` : `grid_2d = grid_2d.permute(0, 3, 1, 2)`
        5.  **Interpoler** à la nouvelle taille (4x4) : `new_grid_2d = F.interpolate(grid_2d, size=(4, 4), mode='bicubic')`
        6.  Aplatir en `(B, N, D)` : `new_pe = new_grid_2d.permute(0, 2, 3, 1).flatten(1, 2)` (devient `[1, 16, 1024]`)
        7.  Ajouter ces `new_pe` à vos `tokens`.

---

## Phase 5 : Étape 4 - Assemblage du `DeepEncoder`

**Objectif :** Combiner le tout dans un seul module `nn.Module`.

- [ ] **Créer la classe `DeepEncoder(nn.Module)` :**
- [ ] **`__init__()` :**
    - `self.sam_backbone = SamVisionModel.from_pretrained(...)`
    - `self.compressor = nn.Conv2d(768, 1024, kernel_size=16, stride=16)`
    - `clip_model = CLIPVisionModel.from_pretrained(...)`
    - `self.clip_body = clip_model.encoder`
    - `self.register_buffer('clip_pos_embed', clip_model.embeddings.position_embedding.weight)` (pour stocker les poids)
- [ ] **Fonction d'interpolation des PE :**
    - `def _interpolate_pos_embed(self, target_size=(4, 4))`
    - Implémenter la logique de la Phase 4 (Gérer les PE).
- [ ] **`forward(image)` :**
    1.  `sam_features = self.sam_backbone(image).last_hidden_state` (Sortie: `[B, 64, 64, 768]`)
    2.  `x = sam_features.permute(0, 3, 1, 2)` (Sortie: `[B, 768, 64, 64]`)
    3.  `x = self.compressor(x)` (Sortie: `[B, 1024, 4, 4]`)
    4.  `target_size = x.shape[2:]`
    5.  `x = x.flatten(2).permute(0, 2, 1)` (Sortie: `[B, 16, 1024]`)
    6.  `pos_embed = self._interpolate_pos_embed(target_size=target_size)`
    7.  `x = x + pos_embed`
    8.  `clip_output = self.clip_body(inputs_embeds=x)`
    9.  `return clip_output.last_hidden_state, sam_features` (le papier fusionne les deux, donc il faut renvoyer les deux)

---

## Phase 6 : Étape 5 - Validation

**Objectif :** Tester l'intégration complète.

- [ ] **Test d'intégration :**
    - Instancier `encoder = DeepEncoder()`
    - Créer une image : `dummy_image = torch.randn(1, 3, 1024, 1024)`
    - Exécuter : `global_features, local_features = encoder(dummy_image)`
- [ ] **Vérifier les sorties :**
    - S'assurer que `global_features.shape` est `[B, 16, 1024]`.
    - S'assurer que `local_features.shape` est `[B, 64, 64, 768]`.
    - S'assurer qu'aucun gradient n'est cassé (si l'entraînement est prévu).