import torch
from torch import nn
import argparse
import os
import yaml
from model.diffusion_model_new import Noise_level, perform_denoising_step
from model.network_swinir import SwinIR
from utils.madataset import MAPairedDataset
import numpy as np
import torchvision.transforms.functional as TF

INTERPOLATION = TF.InterpolationMode.BILINEAR


class TrainingConfig:
    """训练配置类"""
    def __init__(self, config_path=None):
        # 数据相关参数
        self.train_folder = ""
        self.img_size = 256
        self.batch_size = 4
        self.num_workers = 8
        
        # 扩散模型参数
        self.sample_type = 'ddim'
        self.num_custom_steps = 40
        self.encoding_steps = 5
        self.eta = 0.01
        self.source_model_path = ""
        self.target_model_path = ""
        self.config_path = ""
        
        # SwinIR模型参数
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
        self.swinir_pretrain_path = ""
        
        # 训练参数
        self.num_epochs = 100
        self.noise_transform_lr = 1e-6
        self.gen_lr = 0
        self.save_interval = 10
        self.output_dir = ""
        
        # 设备
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
    
    def save_to_yaml(self, save_path):
        """保存配置到YAML文件"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class Trainer:
    """训练器类"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.setup_data()
        self.setup_models()
        self.setup_optimizer()
        
    def setup_data(self):
        """设置数据加载器"""
        dataset = MAPairedDataset(
            data_root=self.config.train_folder, 
            imgsize=self.config.img_size
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.config.num_workers
        )
        
    def setup_models(self):
        """设置扩散模型和噪声转换模块"""

        self.diffusion_model = Noise_level(
            sampling_mode=self.config.sample_type,
            num_custom_steps=self.config.num_custom_steps,
            encoding_steps=self.config.encoding_steps,
            source_checkpoint_path=self.config.source_model_path,
            target_checkpoint_path=self.config.target_model_path,
            yaml_config_path=self.config.config_path,
            ddim_eta_param=self.config.eta
        )
        
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
        
        if os.path.exists(self.config.swinir_pretrain_path):
            self.noise_transform.load_state_dict(torch.load(self.config.swinir_pretrain_path))
            print(f"Loaded noise_transform pretrained weights from {self.config.swinir_pretrain_path}")
        
        self.generator = self.diffusion_model.source_generator.to(self.device)
        self.noise_transform = self.noise_transform.to(self.device)
        
        self.criterion = nn.L1Loss().to(self.device)
        
    def setup_optimizer(self):
        """设置优化器"""
        params_groups = [
            {'params': self.noise_transform.parameters(), 'lr': self.config.noise_transform_lr},
            {'params': self.generator.parameters(), 'lr': self.config.gen_lr}
        ]
        self.optimizer = torch.optim.Adam(params_groups)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.noise_transform.train()
        self.generator.train()
        
        epoch_loss = 0.0
        num_batches = len(self.dataloader)
        
        for i, data in enumerate(self.dataloader):
            data_ma = data["data_ma"].to(device=self.device, dtype=torch.float32)
            data_gt = data["data_gt"].to(device=self.device, dtype=torch.float32)
            data_ma = data_ma.requires_grad_(True)
            data_gt = data_gt.requires_grad_(True)
            
            #计算噪声水平
            with torch.no_grad():
                all_eps = self.diffusion_model.compute_eps(data_ma)
                eps_list = all_eps.view(self.config.batch_size, self.config.encoding_steps, 3, 256, 256)
            
            x_T = eps_list[:, 0].requires_grad_(True)
            y_T = self.noise_transform(x_T)
            
            x = y_T
            eps_list = eps_list.detach()
            
            seq_inv, seq_inv_next = self.compute_time_sequence()
            
            for it, (k, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                t = (torch.ones(self.config.batch_size) * k).to(self.device)
                t_next = (torch.ones(self.config.batch_size) * j).to(self.device)
                
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
            loss = self.criterion(data_gt, x)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:  # 每10个batch打印一次
                print(f'Epoch {epoch}, Batch {i}/{num_batches}, Loss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}')
        return avg_loss
        
    def compute_time_sequence(self):
        """计算时间步序列"""
        if (self.diffusion_model.t_0 + 1) % self.diffusion_model.num_custom_steps == 0:
            seq_inv = range(0, self.diffusion_model.t_0 + 1, 
                          (self.diffusion_model.t_0 + 1) // self.diffusion_model.num_custom_steps)
            assert len(seq_inv) == self.diffusion_model.num_custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.diffusion_model.num_custom_steps) * self.diffusion_model.t_0
        
        seq_inv = [int(s) for s in list(seq_inv)][:self.diffusion_model.encoding_steps]
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.diffusion_model.encoding_steps]
        
        return seq_inv, seq_inv_next
        
    def save_model(self, epoch):
        """保存模型"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, f'model_epoch{epoch}.pt')
        torch.save(self.noise_transform.state_dict(), save_path)
        print(f'Model saved to {save_path}')
        
    def train(self):
        """主训练循环"""
        print(f'Starting training for {self.config.num_epochs} epochs...')
        
        for epoch in range(1, self.config.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            
            # 定期保存模型
            if epoch % self.config.save_interval == 0:
                self.save_model(epoch)
        
        print('Training completed!')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description=' Noise Transform Module Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--train_folder', type=str, default=None,
                       help='Training data folder path')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate for SwinIR')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for saving models')
    parser.add_argument('--save_interval', type=int, default=None,
                       help='Interval for saving models')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化配置
    config = TrainingConfig(args.config)
    
    # 从命令行参数覆盖配置
    if args.train_folder:
        config.train_folder = args.train_folder
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.lr:
        config.noise_transform_lr = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.save_interval:
        config.save_interval = args.save_interval
    
    # 保存当前配置
    config_save_path = os.path.join(config.output_dir, 'training_config.yaml')
    os.makedirs(config.output_dir, exist_ok=True)
    config.save_to_yaml(config_save_path)
    print(f'Configuration saved to {config_save_path}')
    
    # 初始化训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()