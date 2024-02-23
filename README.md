# HealthConnect MedKick: Automated Patient Care and Management System

HealthConnect MedKick is designed to streamline patient care and management through automation, leveraging cutting-edge technologies across several components.

## Components Overview

### 1. Nurse-Patient Audio Call Transcription

#### What I Have Done
- **Model Loading**: Loaded the pre-trained Wav2Vec2 processor and model from "facebook/wav2vec2-base-960h" for audio preprocessing and prediction.
- **Audio Processing**: Utilized the `librosa` library for audio file loading and `noisereduce` library for noise reduction to enhance input audio quality.
- **Transcription**: Processed audio through Wav2Vec2 to encode into model-suitable input values, obtaining logits converted to predicted ids using argmax, and decoded back to text.

#### Future Plans
- **Fine-tuning Wav2Vec2 Model**: Fine-tune on a custom dataset using Hugging Face Transformers.
- **Conformer Model Architecture**: Implement Conformer model architecture, integrating depthwise convolution and self-attention mechanisms for enhanced transcription accuracy.
- **Training and Inference**: Adapt training processes for the Conformer model to improve transcription accuracy and efficiency.

### 2. Secure Communication via BB84 Quantum Protocol

#### Implementation Overview
- **Secure Communication**: Establish a secure channel between the nurse and patient using the BB84 protocol, ensuring conversation confidentiality.
- **Quantum Encryption**: Protect sensitive medical information during nurse-patient calls with quantum encryption, allowing access only to authorized parties.

### 3. Redaction of Personal Information

#### What I Have Done
- **Loading Spacy NLP Model**: Loaded "en_core_web_md" for named entity recognition (NER).
- **Redacting Personal Information**: Applied SpaCy's NER to identify and redact personal information from transcriptions.

#### Future Plans
- **Redaction Integration**: Integrate redaction process into workflow post-transcription, ensuring sensitive information protection.

### 4. Transcript Summary for Doctors

#### Development Steps
- Utilized the PegasusForConditionalGeneration model for summarizing transcriptions, providing concise reports for doctors.

#### Future Development
- Plan to fine-tune the model on powerful systems with custom datasets for tailored summary generation.

### 5. Patient Note Report Generation

#### Implementation Details
- **Model Initialization**: Loaded and prepared the AlpaCare LLM, fine-tuned on local datasets, for generating patient-specific notes.

### 6. Calendar Scheduling via Federated Learning

#### Proposed Approach
- Leverage inputs from various hospital devices to dynamically optimize scheduling, maintaining privacy and security through federated learning.

### 7. Dashboard Portal Application Development

#### Development Plan
- Use Flask and Django for creating a secure dashboard for managing patient information and appointments.
- Integrate with Federated Learning for dynamic scheduling.
- Provide access to patient summaries, medical records, and relevant medical resources, ensuring data protection and privacy.
