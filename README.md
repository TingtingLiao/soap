<h1 align="center">
  <img src="./assets/logo.png" style="width: 40px; height: 40px; vertical-align: middle; margin-right: 1px;">
  SOAP: Style-Omniscient Animatable Portraits 
</h1> 

<p align="center">
    <a href="https://tingtingliao.github.io/"><strong>Tingting Liao</strong></a>
    ·
    <a href=""><strong>Yujian Zheng</strong></a>
    ·
    <a href="http://xiuyuliang.cn/"><strong>Yuliang Xiu</strong></a>
    · 
    <a href=""><strong>Adilbek Karmanov</strong></a>
    ·
    <a href=""><strong>Liwen Hu</strong></a>
    ·
    <a href="https://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html"><strong>Hao Li</strong></a>
</p>  
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href=''><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=red'></a>
  <a href='https://tingtingliao.github.io/soap'><img src='https://img.shields.io/badge/project-homepage-orange?logo=Homepage&logoColor=orange'></a>
  <a href="https://github.com/TingtingLiao/soap"><img src="https://img.shields.io/github/stars/TingtingLiao/soap?logo=github&logoColor=white"></a>
  <a href=''><img src='https://img.shields.io/badge/license-MIT-blue?logo=C&logoColor=blue'></a>
</div>
<br>  

https://github.com/user-attachments/assets/408b3250-0c41-45e2-a43a-25b837800a2e

## 🔥 News 
- **`2025/04/01`** 🌟 Luanch the project.

<!-- We released the **code** and [**webpage**](https://tingtingliao.github.io/soap) of SOAP -->

# 🍀 Install  
#### 1. Install environment    
```bash
git clone --single-branch --branch main  git@github.com:TingtingLiao/soap.git
cd soap 
conda create -n soap python=3.10  
conda activate soap   
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
pip install -r requirements.txt  
``` 
#### 2. Download weights
**Extra Data**
``` 
cd data 
``` 
**Diffusion Model**
Download [Deep3dface](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP) pretrained model under ./data/deep3dface

# 🍉 Usage 

**Processing**  
If you have problem accessing huggingface, set **`export HF_ENDPOINT=https://hf-mirror.com`** or download weights here. 
```bash  
python process.py image=data/examples/75997f09a870ae86120d26b5f934a7cae3698f09_high.jpg
```

**Reconstruction**
```bash  
python main.py image=data/examples/75997f09a870ae86120d26b5f934a7cae3698f09_high.jpg
```  
The generated results will be saved under **`./output/75997f09a870ae86120d26b5f934a7cae3698f09_high/6-views/w-eyes/`**.

# 🍋 GUI 
We provide `gui.py` for visualization and interation with the editing the face shape.
```bash 
python gui.py -i results/75997f09a870ae86120d26b5f934a7cae3698f09_high 
```
 
# Acknowledgments
We thank the following projects for their contributions to the development of SOAP:
- [Unique3D](https://github.com/AiuniAI/Unique3D) for multi-view diffusion initialization. 
- [FaceParsing](https://huggingface.co/jonathandinu/face-parsing) for hair and eyes segmentation. 
- [face-alignment](https://github.com/1adrianb/face-alignment) and [MediaPipe](https://github.com/google-ai-edge/mediapipe) for landmark detection and cropping. 
- [EMOCA](https://github.com/radekd91/emoca) and [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for parametric model estimation. 
- [Continuous Remeshing](https://github.com/Profactor/continuous-remeshing) for mesh processing. 
- [FLAME](https://flame.is.tue.mpg.de/) for parametric head model initialization. 