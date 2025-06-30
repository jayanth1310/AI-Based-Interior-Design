# 🛋️ AI Interior Designer – Room Redesign with Stable Diffusion

## 📌 Overview

AI Interior Designer is a web-based application that transforms your room photos into redesigned spaces using generative AI. Powered by **Stable Diffusion with ControlNet**, the system allows users to upload a photo of their room and provide a textual prompt describing design preferences (e.g., "modern white sofa with wooden flooring"). The app then generates photorealistic redesigns while preserving room structure using depth estimation.

---

## ✨ Key Features

- 📸 Upload a room photo
- 💬 Input interior design prompts (e.g., furniture, color scheme, decor)
- 🧠 AI-powered transformation using Stable Diffusion + ControlNet
- 🎯 Uses depth estimation to maintain room layout realism
- 🖼️ Visual comparison of original and redesigned room
- 🌐 Shareable public URL using **ngrok**
- ☁️ **Run seamlessly in Google Colab** if local GPU isn't available

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **AI Models:**
  - Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
  - ControlNet Depth Guidance (`lllyasviel/sd-controlnet-depth`)
  - Depth Estimation (`Intel/dpt-large`)
- **Libraries:** 
  - PyTorch, Hugging Face Diffusers & Transformers
  - OpenCV, Pillow, NumPy
  - Streamlit, pyngrok

---

## 📦 Installation (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-interior-designer.git
cd ai-interior-designer
```

##☁️ Run on Google Colab (No GPU? No Problem!)
Running Stable Diffusion models locally requires a good GPU. You can run this app on Google Colab for free with GPU support and It is recommended for you to use google colab.
💡 Tip: Enable GPU in Colab by going to Runtime > Change runtime type > GPU
