# Tank Learning Game

## Entraînement Multi-Agents:

C'est un jeu de tanks en 2D où deux tanks s'affrontent sur une grille. La particularité : les tanks peuvent **apprendre à jouer tout seuls** grâce à un algorithme d'apprentissage par renforcement appelé **DQN (Deep Q-Network)**.

Chaque tank peut :
- Tourner à gauche / droite
- Avancer
- Tirer des balles
- Poser des mines
- Rester immobile

Plus les agents jouent, plus ils deviennent intelligents. Tu peux aussi sauvegarder les agents entraînés et les recharger plus tard.

### Modes de jeu disponibles :
| Mode | Description |
|------|-------------|
| `NEW vs NEW` | Deux nouveaux agents apprennent ensemble |
| `NEW vs AGENT` | Un nouvel agent affronte un agent déjà entraîné |
| `AGENT vs AGENT` | Deux agents déjà entraînés s'affrontent |

---

## Version Python requise

```
Python 3.11 ou plus récent (recommandé sur Python 3.11).
```

---

## Packages à installer

```bash
pip install pygame torch numpy
```

| Package | Utilité |
|---------|---------|
| `pygame` | Affichage du jeu et interface graphique |
| `torch` | Réseau de neurones et entraînement DQN |
| `numpy` | Calculs mathématiques et tableaux |

> Si tu as une carte GPU NVIDIA, PyTorch l'utilisera automatiquement pour accélérer l'entraînement.

---

## Lancer le jeu

```bash
python main.py
```

---

## Sauvegarder un agent

À la fin d'une session, un bouton **"Save & Close"** te permet de sauvegarder les agents entraînés dans le dossier `saved_agents/` au format `.pt`. Tu pourras les recharger lors d'une prochaine partie.
