```bash
git clone https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
git clone https://github.com/kijai/ComfyUI-KJNodes      
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite

pip install -r Stand-In_Preprocessor_ComfyUI/requirements.txt
pip install -r ComfyUI-WanVideoWrapper/requirements.txt
pip install -r ComfyUI-KJNodes/requirements.txt
pip install -r ComfyUI-VideoHelperSuite/requirements.txt

wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0_fp32.safetensors
cp Stand-In_wan2.1_T2V_14B_ver1.0_fp32.safetensors ComfyUI/models/loras/
featurize dataset download c926e468-854b-4b17-aacf-f543d25c748f
unzip wan_loras.zip
cp wan_loras/* ComfyUI/models/loras/

wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors
cp Wan2_1-T2V-14B_fp8_e4m3fn.safetensors ComfyUI/models/diffusion_models

wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors
cp umt5-xxl-enc-bf16.safetensors ComfyUI/models/text_encoders/

wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
cp wan_2.1_vae.safetensors ComfyUI/models/vae/
```

# Stand-In Official Preprocessor ComfyUI Nodes
  <h1>
    <img src="Stand-In.png" width="85" alt="Logo" valign="middle">
    Stand-In
  </h1>

[![Project Page](https://img.shields.io/badge/Project_Page-www.stand--in.tech-green)](https://www.stand-in.tech)
[![Main Repo](https://img.shields.io/badge/Main_Repository-Stand--In-blue)](https://github.com/WeChatCV/Stand-In)

> 🧩 Only images processed with this official preprocessor can fully unleash the power of **Stand-In**.

---

This repository provides a **temporary** ComfyUI node implementation of the Stand-In preprocessor, aiming to correct the misunderstanding of Stand-In’s preprocessing logic in [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

We **strongly recommend** using the workflows provided in this repo to ensure proper compatibility and the best identity-preserving performance.

> **Note**: This is a **temporary node**. Features such as KV cache support and advanced attention mechanisms are **not yet implemented**. We are actively developing the **official full-featured ComfyUI nodes** for Stand-In.


## ⚠️ Important Usage Tips

> **Prompt Writing Tip:**  
> If you do **not** wish to alter the subject's facial features, simply use **"a man"** or **"a woman"** without adding extra descriptions of their appearance.  
> Prompts support both **Chinese** and **English** input.  
> The prompt is intended for generating **frontal, medium-to-close-up videos**.

> **Input Image Recommendation:**  
> For best results, use a **high-resolution frontal face image**.  
> There are **no restrictions** on resolution or file extension, as our **built-in preprocessing pipeline** will handle them automatically.

---
We are also actively developing a more flexible version of Stand-In.

---

## 🌐 Project Links

- 📘 **Main Repository**: [WeChatCV/Stand-In](https://github.com/WeChatCV/Stand-In)
- 🏠 **Project Page**: [www.stand-in.tech](https://www.stand-in.tech)

---

## 🚀 Installation
Clone this repository into the `custom_nodes` directory of your ComfyUI installation:

```bash
git clone https://github.com/WeChatCV/Stand-In_Preprocessor_ComfyUI.git
cd Stand-In_Preprocessor_ComfyUI
pip install -r requirements.txt
