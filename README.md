<h1 align="center">
  SOAP: Style-Omniscient Animatable Portraits 
</h1> 

<p align="center">
    <a href="https://tingtingliao.github.io/"><strong>Tingting Liao</strong></a>
    ·
    <a href="https://paulyzheng.github.io/about/"><strong>Yujian Zheng</strong></a>
    · 
    <a href="https://www.linkedin.com/in/adilbek-karmanov?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"><strong>Adilbek Karmanov</strong></a>
    · 
    <a href="https://scholar.google.com/citations?user=Mvq6pGcAAAAJ&hl=en"><strong>Liwen Hu</strong></a>
    ·
    <a href=""><strong>Leyang Jin</strong></a>
    ·
    <a href="http://xiuyuliang.cn/"><strong>Yuliang Xiu</strong></a>
    ·
    <a href="https://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html"><strong>Hao Li</strong></a>
</p>  
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/abs/2505.05022'><img src='https://img.shields.io/badge/arXiv-Paper-darkred?logo=arxiv&logoColor=darkred'></a>  &ensp;
  <a href='https://tingtingliao.github.io/soap'><img src='https://img.shields.io/badge/project-homepage-orange?logo=Homepage&logoColor=orange'></a>  &ensp;
  <a href="https://huggingface.co/Luffuly/head-mvimage-diffuser"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Released&message=Models&color=green"></a> &ensp;
  <a href="https://github.com/TingtingLiao/soap"><img src="https://img.shields.io/github/stars/TingtingLiao/soap?logo=github&logoColor=black"></a>  &ensp;
  <a href=''><img src='https://img.shields.io/badge/license-MIT-blue?logo=C&logoColor=blue'></a>
</div>  

<h3 align="center">Siggraph 2025 (Conference Track)</h3>

<!-- https://github.com/user-attachments/assets/408b3250-0c41-45e2-a43a-25b837800a2e -->

<video src="https://github.com/user-attachments/assets/408b3250-0c41-45e2-a43a-25b837800a2e" autoplay loop muted playsinline width="640" height="360"> 
</video>

## 🔥 News 
- **`2025/04/30`** 🌟 **Code** and [**Webpage**](https://tingtingliao.github.io/soap) are released. 
- **`2025/04/29`** 🚀 SOAP is accpted by Siggraph2025.  

# 🍀 Install  
#### 1. Install environment    
```bash
sudo apt-get install libegl1-mesa-dev
sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
sudo apt-get install mesa-utils

git clone --single-branch --branch main  https://github.com/TingtingLiao/soap.git
cd soap 
conda create -n soap python=3.10 -y
conda activate soap   
# For cuda-12.1 
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
# For cuda-11.8 
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 

pip install -r requirements.txt  
``` 
#### 2. Download Extra Data 
Download [extra data](https://drive.google.com/file/d/1fiFsFV-fu94i3szSDjQp3mSdY18b1zKZ/view?usp=sharing) and unzip it as `./data`.


# 🍉 Usage 

**1. Processing**   
The pre-processing step will generate the multive images and normals, initial FLAME model, detect facial landmarks and face parsing. Please login huggingface **`huggingface-cli login`** before running: 

```bash   
python process.py image=assets/examples/00.png 
```


**2. Reconstruction**
```bash  
python main.py image=assets/examples/00.png
```  
The generated results will be saved under **`./output/examples/00/6-views/`**.

These results are rendered by `nvidiffrast`. We recommend you to use `blender3.6` for realistic rendering. 

**3. Blender Rendering**

The code is coming soon. 


# 🍋 GUI 
We provide `gui.py` for visualization and interation with the editing the face shape.
```bash 
python gui.py -i output/examples/00
```
https://github.com/user-attachments/assets/20ed6d19-c6d5-4ee1-a355-527b00a5bc82

# 🤗 Acknowledgments
We thank the following projects for their contributions to the development of SOAP:
- [Unique3D](https://github.com/AiuniAI/Unique3D) for multi-view diffusion initialization. 
- [FaceParsing](https://huggingface.co/jonathandinu/face-parsing) for hair and eyes segmentation. 
- [face-alignment](https://github.com/1adrianb/face-alignment) and [MediaPipe](https://github.com/google-ai-edge/mediapipe) for landmark detection and cropping. 
- [EMOCA](https://github.com/radekd91/emoca) and [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch) for parametric model estimation. 
- [Continuous Remeshing](https://github.com/Profactor/continuous-remeshing) for mesh processing. 
- [FLAME](https://flame.is.tue.mpg.de/) for parametric head model initialization. 


# Contributors

<a href="https://github.com/TingtingLiao/soap/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TingtingLiao/soap" />
</a>

# 📖 Citation
```bibtex
@inproceedings{liao2025soap,
  title={{SOAP: Style-Omniscient Animatable Portraits }},
  author={Liao, Tingting and Zheng, Yujian and Karmanov, Adilbek and Hu, Liwen and Jin, Leyang and Xiu, Yuliang and Hao Li},
  booktitle={International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH)},
  year={2025}
}
```


