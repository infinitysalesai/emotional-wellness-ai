# Human OS — Emotional Voice Engine

This repository contains the **voice interaction engine** used by the Human OS project.

The engine listens to a user’s voice, analyzes emotional cues, generates an empathetic response, and produces a voice reply.

This module is designed to run as a **Hugging Face Space backend service** that can be connected to external applications such as Lovable frontends or mobile apps.

---

## What This Repository Contains

This repository includes only the **voice engine component**, which performs:

• Speech-to-text transcription
• Emotional tone detection from audio
• Conversational AI response generation
• Emotion-aware text-to-speech synthesis

Other Human OS components (UI, orchestration, data storage, etc.) are **not included in this repository**.

---

## Role in the Human OS System

The voice engine acts as the **AI processing layer** in the Human OS architecture.

User Voice
→ Hugging Face Voice Engine (this repo)
→ AI Response
→ Frontend Application

See **architecture.png** for system context.

---

## Main File

`app.py`

This script launches a Gradio application that:

1. Accepts microphone input
2. Converts speech to text
3. Detects emotional characteristics in the audio
4. Generates an empathetic response
5. Produces a voice reply

The application can be deployed as a **Hugging Face Space**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/human-os-voice-engine.git
cd human-os-voice-engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
```

---

## Hugging Face Deployment

This repository is intended to run inside a Hugging Face Space.

To deploy:

1. Create a new Space
2. Upload this repository
3. Ensure `requirements.txt` is present
4. Hugging Face will automatically launch the Gradio app

---

## Safety Notice

This project is an AI conversational system and **not a medical or mental health service**.

If a user expresses crisis-related language, they should be directed to professional help resources.

---

## Future Improvements

• Streaming voice conversations
• Advanced emotion recognition models
• Real-time response generation
• Improved emotional voice synthesis

---

## License

MIT License
