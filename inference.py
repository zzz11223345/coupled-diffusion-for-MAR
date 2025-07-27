import torch
from torch import nn
import argparse
import os
import yaml
from model.diffusion_model_new import Noise_level, perform_denoising_step
from model.network_swinir import SwinIR
from utils.madataset import MAClinicalDataset
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

INTERPOLATION = TF.InterpolationMode.BILINEAR


class InferenceConfig:
    """推理配置类"""
    def __init__(self, config_path=None):
        self.test_folder = ""
        self.img_size = 256
        self.batch_size = 4
        self.num_workers = 8
        
        self.sample_type = 'ddim'
        self.num_custom_steps = 40
        self.encoding_steps = 5
        self.eta = 0.01
        self.source_model_path = ""
        self.target_model_path = ""
        self.config_path = ""
        
        self.upscale = 1
        self.window_size = 8
        self.heights = 256
        self.widths = 256
        self.depths = [6, 6, 6, 6]
        self.embed_dim = 60
        self.num_heads = [6, 6, 6, 6]
        self.mlp_ratio = 2
        self.upsampler = 'pixelshuffledirect'
        self.img_range = 1.0
        
        self.noise_transform_model_path = "" 
        self.output_dir = ""  
        self.save_images = True  
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)


class InferenceEngine:
    """推理引擎类"""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.setup_data()
        self.setup_models()
        
    def setup_data(self):
        """设置数据加载器（与训练相同）"""
        dataset = MAClinicalDataset(
            data_root=self.config.test_folder, 
            imgsize=self.config.img_size
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers
        )
        
        print(f"Loaded test dataset with {len(dataset)} samples")
        
    def setup_models(self):
        """设置扩散模型和噪声转换模块"""
        print("Loading diffusion model...")
        self.diffusion_model = Noise_level(
            sampling_mode=self.config.sample_type,
            num_custom_steps=self.config.num_custom_steps,
            encoding_steps=self.config.encoding_steps,
            source_checkpoint_path=self.config.source_model_path,
            target_checkpoint_path=self.config.target_model_path,
            yaml_config_path=self.config.config_path,
            ddim_eta_param=self.config.eta
        )
        
        print("Loading noise transform model...")
        self.noise_transform = SwinIR(
            upscale=self.config.upscale,
            img_size=(self.config.heights, self.config.widths),
            window_size=self.config.window_size,
            img_range=self.config.img_range,
            depths=self.config.depths,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            upsampler=self.config.upsampler
        )
        
        if os.path.exists(self.config.noise_transform_model_path):
            self.noise_transform.load_state_dict(torch.load(self.config.noise_transform_model_path, map_location=self.device))
            print(f"Loaded noise transform model from {self.config.noise_transform_model_path}")
        else:
            raise FileNotFoundError(f"Noise transform model not found at {self.config.noise_transform_model_path}")
        
        self.generator = self.diffusion_model.source_generator.to(self.device)
        self.noise_transform = self.noise_transform.to(self.device)
        
        self.generator.eval()
        self.noise_transform.eval()
        
    def compute_time_sequence(self):
        if (self.diffusion_model.t_0 + 1) % self.diffusion_model.num_custom_steps == 0:
            seq_inv = range(0, self.diffusion_model.t_0 + 1, 
                          (self.diffusion_model.t_0 + 1) // self.diffusion_model.num_custom_steps)
            assert len(seq_inv) == self.diffusion_model.num_custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.diffusion_model.num_custom_steps) * self.diffusion_model.t_0
        
        seq_inv = [int(s) for s in list(seq_inv)][:self.diffusion_model.encoding_steps]
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.diffusion_model.encoding_steps]
        
        return seq_inv, seq_inv_next
        
    def inference_batch(self, data_ma):
        """对一个batch执行推理"""
        with torch.no_grad():
            batch_size = data_ma.shape[0]
            
            # 计算噪声水平
            all_eps = self.diffusion_model.compute_eps(data_ma)
            eps_list = all_eps.view(batch_size, self.config.encoding_steps, 3, 256, 256)
            
            # 噪声转换
            x_T = eps_list[:, 0]
            y_T = self.noise_transform(x_T)
            x = y_T

            # 计算时间步序列
            seq_inv, seq_inv_next = self.compute_time_sequence()
            
            # 逐步去噪
            for it, (k, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                t = (torch.ones(batch_size) * k).to(self.device)
                t_next = (torch.ones(batch_size) * j).to(self.device)
                
                if it < self.config.encoding_steps - 1:
                    eps = eps_list[:, it]
                else:
                    eps = torch.randn_like(x)
                
                x = perform_denoising_step(
                    x, noise_eps=eps, current_t=t, next_t=t_next, neural_models=self.generator,
                    log_variances=self.diffusion_model.log_variance,
                    denoise_method=self.diffusion_model.sampling_mode,
                    beta_schedule=self.diffusion_model.beta_schedule,
                    ddim_eta=self.config.eta,
                    sigma_learning=self.diffusion_model.sigma_learning
                )
            
            x = self.diffusion_model.output_processor(x)
            
        return x
        
    def save_batch_images(self, images, name):
        """保存batch中的图像"""
        if not self.config.save_images:
            return
            
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().clamp(0, 1)
            img = TF.to_pil_image(img_tensor)
            filename = name[i]
            save_path = os.path.join(self.config.output_dir, filename)
            img.save(save_path)
        
    def run_inference(self):
        """运行推理"""
        print(f"Starting inference on test dataset...")
        print(f"Total batches: {len(self.dataloader)}")
        
        if self.config.save_images:
            os.makedirs(self.config.output_dir, exist_ok=True)
            print(f"Results will be saved to: {self.config.output_dir}")
        
        total_samples = 0
        
        for batch_idx, data in enumerate(self.dataloader):
            data_ma = data["data_ma"].to(device=self.device, dtype=torch.float32)
            data_name = data["name"]
            print(f"Processing batch {batch_idx + 1}/{len(self.dataloader)}, batch size: {data_ma.shape[0]}")
            
            output = self.inference_batch(data_ma)
            
            if self.config.save_images:
                self.save_batch_images(output, data_name)
            
            
            total_samples += data_ma.shape[0]
        
        print(f"Inference completed! Processed {total_samples} samples")
        
        if self.config.save_images:
            print(f"All results saved to: {self.config.output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Noise Transform Module Inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to inference configuration YAML file')
    parser.add_argument('--test_folder', type=str, default=None,
                       help='Test data folder path')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained noise transform model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for saving results')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save output images (only calculate metrics)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = InferenceConfig(args.config)

    if args.test_folder:
        config.test_folder = args.test_folder
    if args.model_path:
        config.noise_transform_model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_save:
        config.save_images = False
    
    if not config.test_folder:
        raise ValueError("Test folder path must be specified")
    if not config.noise_transform_model_path:
        raise ValueError("Noise transform model path must be specified")
    if config.save_images and not config.output_dir:
        raise ValueError("Output directory must be specified when saving images")
    
    inference_engine = InferenceEngine(config)
    inference_engine.run_inference()


if __name__ == "__main__":
    main()