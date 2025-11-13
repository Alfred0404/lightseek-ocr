# TODO — LightSeek-OCR

Suivi du développement de l'implémentation de DeepSeek-OCR (encodeur + décodeur).

---

## Encodeur (DeepEncoder)

### Architecture & Structure

- [x] Créer l'architecture modulaire (SAMFeatureExtractor, CLIPVisionProcessor, Conv2DCompressor)
- [x] Implémenter la classe `DeepEncoder` orchestrant le pipeline complet
- [x] Structurer le projet en fichiers séparés pour chaque composant
- [x] Documenter l'architecture dans le README (section Architecture)

### Extracteur Local (SAM)

- [x] Charger le modèle SAM via `transformers`
- [x] Extraire les features locales (256 canaux, 64×64)
- [x] Valider les dimensions de sortie

### Compresseur (CNN)

- [x] Implémenter la convolution stride-16 pour compression spatiale
- [x] Gérer la conversion de format (Channels-Last → Channels-First)
- [x] Valider la sortie compressée (768 canaux, 4×4)

### Extracteur Global (CLIP)

- [x] Charger le modèle CLIP et isoler l'encodeur vision
- [x] Contourner la couche d'embedding de CLIP
- [x] Implémenter l'interpolation des embeddings positionnels
- [x] Retourner la séquence complète (pas de pooling CLS)
- [x] Valider la sortie (16 tokens, 768-dim)

### Pipeline Complet

- [x] Intégrer texte → image → SAM → compression → CLIP
- [x] Tester le pipeline end-to-end
- [x] Valider les "visual plugs" (local + global features)

---

## Décodeur (à implémenter)

### Architecture Générale

- [ ] Étudier l'architecture du décodeur DeepSeek-OCR (papier + code GitHub)
- [ ] Définir la structure du décodeur (Transformer-based)
- [ ] Choisir le modèle de base (LLaMA, GPT-2, ou autre)

### Cross-Attention Multi-Échelle

- [ ] Implémenter la cross-attention avec features locales (SAM 64×64)
- [ ] Implémenter la cross-attention avec features globales (CLIP 16 tokens)
- [ ] Fusionner les deux niveaux d'attention (stratégie de fusion à définir)

### Tokenizer & Vocabulaire

- [ ] Choisir/adapter un tokenizer pour l'OCR (tokens texte + tokens spéciaux)
- [ ] Gérer les tokens de début/fin de séquence
- [ ] Implémenter le masquage causal pour la génération autorégressive

### Module de Décodage

- [ ] Créer la classe `DeepDecoder(nn.Module)`
- [ ] Implémenter `forward()` avec auto-régression
- [ ] Gérer les embeddings de position pour la séquence de sortie
- [ ] Implémenter la tête de prédiction (Linear → logits → softmax)

### Intégration Encodeur-Décodeur

- [ ] Créer la classe `LightSeekOCR` combinant `DeepEncoder` + `DeepDecoder`
- [ ] Définir l'interface d'inférence (texte → features → texte décodé)
- [ ] Implémenter la génération avec beam search / sampling

### Tests & Validation

- [ ] Test unitaire du décodeur avec features dummy
- [ ] Test d'intégration encodeur + décodeur
- [ ] Validation des dimensions à chaque étape
- [ ] Test de génération de texte (même sans entraînement, vérifier que ça tourne)

---

## Entraînement (futur)

### Données

- [ ] Identifier/créer un dataset texte-image pour OCR
- [ ] Implémenter un DataLoader compatible
- [ ] Prétraiter les données (augmentation, normalisation)

### Boucle d'Entraînement

- [ ] Définir la fonction de loss (CrossEntropy pour séquences)
- [ ] Implémenter la boucle d'entraînement avec teacher forcing
- [ ] Configurer l'optimiseur (AdamW) et le scheduler
- [ ] Ajouter le logging (wandb/tensorboard)

### Fine-Tuning

- [ ] Geler/dégeler sélectivement les composants (SAM, CLIP, décodeur)
- [ ] Expérimenter avec différentes stratégies de fine-tuning
- [ ] Évaluer les performances (CER, WER, accuracy)

---

## Documentation & Qualité

### Code

- [ ] Ajouter des docstrings complètes à tous les modules
- [ ] Ajouter des type hints partout
- [ ] Créer des tests unitaires (pytest)
- [ ] Configurer un linter (black, flake8)

### Documentation

- [x] Section Architecture dans README
- [ ] Guide d'utilisation complet
- [ ] Exemples d'inférence (notebooks Jupyter)
- [ ] Documenter les hyperparamètres et configurations

### Ressources

- [ ] Créer un dossier `examples/` avec scripts de démo
- [ ] Ajouter des visualisations (attention maps, features)
- [ ] Créer un notebook de tutoriel complet

---

## Historique

- **2025-11-13** — Création de l'encodeur modulaire (DeepEncoder, SAMFeatureExtractor, CLIPVisionProcessor, Conv2DCompressor)
- **2025-11-13** — Ajout section Architecture dans README
- **2025-11-10** — Création initiale de TODO.md
