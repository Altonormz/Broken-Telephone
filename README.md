# Broken Telephone AI

Play the classic "Telephone" game with AI! This project creates a chain of AI-powered speech-to-text and text-to-speech conversions, demonstrating how messages can hilariously transform - just like in the childhood game.


![ALT text](./Broken_Telephone.png)


## Overview

Broken Telephone AI takes your voice input in English and passes it through multiple AI models, each converting between speech and text. Watch as your original message evolves through each transformation, often with unexpected and entertaining results.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Altonormz/Broken-Telephone.git

# Navigate to project directory
cd Broken-Telephone

# Install dependencies
pip install -r requirements.txt
```

## Setup Requirements

1. Download the LibriTTS model:
   - Follow instructions at [StyleTTS2](https://github.com/yl4579/StyleTTS2)
   - Place model file at: `Broken-Telephone/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`

2. Add required assets:
   - Reference voice files (WAV format) → `Broken-Telephone/StyleTTS2/reference_voices/`
   - Character images (SVG format) → `Broken-Telephone/characters/`

## Features

- Voice recording input
- Multiple AI-powered conversions
- Speech-to-text and text-to-speech transformations
- Visual character representations
- Entertaining message evolution


Disclaimer: 
This is just a clickbait to see my LinkedIn.
I made this project after giving the original task of making the "Broken-Telephone AI" to interns, as an opprtuinity to work with STT/TTS models, and API.
My favorite part was working with them on the data analysis and getting real insights that helped us once we started fine-tuning both models.   
