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
- [x] Valider les dimensions de sortie: `torch.Size([1, 256, 64, 64])`

### Compresseur (CNN)

- [x] Implémenter la convolution stride-16 pour compression spatiale
- [x] Gérer la conversion de format (Channels-Last → Channels-First)
- [x] Valider la sortie compressée (768 canaux, 16×16): `torch.Size([1, 768, 16, 16])`

### Extracteur Global (CLIP)

- [x] Charger le modèle CLIP et isoler l'encodeur vision
- [x] Contourner la couche d'embedding de CLIP
- [x] Implémenter l'interpolation des embeddings positionnels
- [x] Retourner la séquence complète (pas de pooling CLS)
- [x] Valider la sortie (256 tokens, 768-dim): `torch.Size([1, 256, 768])`

### Pipeline Complet

- [x] Intégrer texte → image → SAM → compression → CLIP
- [x] Tester le pipeline end-to-end
- [x] Valider les "visual plugs" (local + global features)

---

## Décodeur (à implémenter)

### Architecture Générale

- [x] Étudier l'architecture du décodeur DeepSeek-OCR (papier + code GitHub)
- [x] Définir la structure du décodeur: **Transformer Decoder-only** (4-6 layers, 8 heads)
- [x] Choisir le modèle de base: **Scratch** (recommandé) ou **Qwen-0.5B** (avec LoRA)

### Cross-Attention Multi-Échelle

- [x] Implémenter la cross-attention avec features locales (SAM 64×64 = 256 tokens)
- [x] Implémenter la cross-attention avec features globales (CLIP = 256 tokens)
- [x] Fusionner les deux niveaux d'attention: **Concaténation** (Total 512 visual tokens)

### Tokenizer & Vocabulaire

- [ ] Choisir/adapter un tokenizer pour l'OCR (tokens texte + tokens spéciaux)
- [ ] Gérer les tokens de début/fin de séquence
- [ ] Implémenter le masquage causal pour la génération autorégressive

### Module de Décodage

- [x] Créer la classe `DeepDecoder(nn.Module)`
- [x] Implémenter `forward()` avec auto-régression
- [x] Gérer les embeddings de position pour la séquence de sortie
- [x] Implémenter la tête de prédiction (Linear → logits → softmax)

### Intégration Encodeur-Décodeur

- [x] Créer la classe `LightSeekOCR` combinant `DeepEncoder` + `DeepDecoder`
- [x] Définir l'interface d'inférence (texte → features → texte décodé)
- [x] Implémenter la génération avec beam search / sampling

### Tests & Validation

- [x] Test unitaire du décodeur avec features dummy
- [x] Test d'intégration encodeur + décodeur
- [x] Validation des dimensions à chaque étape
- [x] Test de génération de texte (même sans entraînement, vérifier que ça tourne)

---

## Entraînement (futur)

### Données

- [ ] Identifier/créer un dataset texte-image pour OCR
- [ ] Implémenter un DataLoader compatible
- [ ] **Optimisation**: Pré-calculer les features SAM/CLIP sur disque pour éviter de les charger en RAM pendant l'entraînement

### Boucle d'Entraînement

- [ ] Définir la fonction de loss (CrossEntropy pour séquences)

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
