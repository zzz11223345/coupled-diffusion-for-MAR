# coupled-diffusion-for-MAR

[paper url](https://ieeexplore.ieee.org/document/11072804) Coupled Diffusion Models for Metal Artifact Reduction of Clinical Dental CBCT Images
> This method is applicable to unpaired image artifact reduction. The paper focuses on dental CBCT metal artifact reduction, but theoretically can be applied to other domains.

## ✨ Prerequisites
* 🚀 Datasets: Metal artifact CBCT image dataset and clean CBCT image dataset
* 💡 Diffusion Model Training: Train diffusion models using [guided-diffusion](https://github.com/openai/guided-diffusion), separately training 2 diffusion models with metal artifact CBCT image dataset and clean CBCT image dataset.
* 🔧 Paired Data Synthesis: Use [DuDoNet](https://github.com/MIRACLE-Center/DuDoNet) method to synthesize corresponding metal artifact CBCT from clean CBCT, thus obtaining paired metal artifact and clean images.

## 🛠️ Environment Setup
```bash
# Clone the repository
git clone https://github.com/zzz11223345/coupled-diffusion-for-MAR.git
# Navigate to project directory
cd coupled-diffusion-for-MAR
# Install dependencies
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 ##Install torch according to your environment
pip install numpy opencv-python h5py tqdm timm
```

## 🚀 Training and Inference Pipeline
```bash
# Training
python3 train.py --config/train_config.yaml ##Modify training configuration file as needed
# Inference
python3 inference.py --config/inference_config.yaml ##Modify inference configuration file as needed
# MA-adaptive merge
python3 ma_adaptive_merge.py
```

## 📚 Usage for Other Domains
* 📁 Applicable Scope: Suitable for image restoration scenarios where: accurate ground truth is lacking, although pairing can be achieved through synthesis methods, there exists a large domain gap between synthetic and real data.
* 🚀 Data Preparation: Real-world corrupted image dataset, unpaired uncorrupted image dataset, and paired dataset obtained through synthesis methods.
* 💡 Training and Inference: The subsequent processes for diffusion model training, noise transformation module training, and inference follow the same pipeline as described above. Custom datasets for data loading can be defined in ./utils/madataset.py.


```bibtex
@ARTICLE{11072804,
  author={Zhang, Zhouzhuo and Yan, Juncheng and Shi, Yuxuan and Cui, Zhiming and Xu, Jun and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Coupled Diffusion Models for Metal Artifact Reduction of Clinical Dental CBCT Images}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Diffusion models;Dentistry;Noise;Metals;Noise reduction;Mars;Image restoration;Image segmentation;Training;Noise measurement;Metal artifact reduction;coupled diffusion models;noise transformation;MA-adaptive inference},
  doi={10.1109/TMI.2025.3587131}
}
```

---
If this project helps you, please give it a ⭐️ for support!