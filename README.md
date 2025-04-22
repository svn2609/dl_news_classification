# 📰 AG News Classification with LoRA + RoBERTa

This repository contains the code and notebook for a deep learning project that fine-tunes the `roberta-base` model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) on the [AG News dataset](https://huggingface.co/datasets/ag_news). The project explores parameter-efficient fine-tuning, contextual data augmentation, and mixed-precision (FP16) training to achieve state-of-the-art performance with minimal resource usage.

---

## 🚀 Project Highlights

- 📚 **Model**: RoBERTa-base with LoRA adapters (trainable params < 1M)
- 🧠 **Technique**: Parameter-Efficient Fine-Tuning using LoRA
- 🔁 **Augmentation**: Contextual word replacement using MLM (RoBERTa)
- ⚙️ **Training**: Mixed precision FP16 with Hugging Face Trainer
- 📈 **Accuracy**: ~95.3% on AG News test set with macro F1 parity
- 💾 **Efficient**: Trainable parameters ≈ 0.6% of full model

---

## 📂 Repository Structure

```bash
├── Roberta_news_classification.ipynb     # Jupyter notebook with full implementation
├── accuracy-vs-epoch.png         
├── loss-vs-epoch.png             
├── main.tex                     
└── README.md
├── inference_output.csv                   
└── test_unlabelled.pkl
