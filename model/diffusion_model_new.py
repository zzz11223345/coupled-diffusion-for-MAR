import os
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from .improved_ddpm.script_util import create_model, Defalut_DICT


def setup_ddpm_ddim_config(yaml_config_path):
    def dict_to_namespace(config_dict):
        namespace_obj = argparse.Namespace()
        for param_key, param_value in config_dict.items():
            if isinstance(param_value, dict):
                converted_value = dict_to_namespace(param_value)
            else:
                converted_value = param_value
            setattr(namespace_obj, param_key, converted_value)
        return namespace_obj
    
    with open(yaml_config_path, 'r') as config_file:
        loaded_config = yaml.safe_load(config_file)
    processed_config = dict_to_namespace(loaded_config)

    return processed_config


def gather_coefficients(coeff_array, timestep_indices, tensor_shape):
    """Extract coefficients from coeff_array based on timestep_indices and reshape to make it
    broadcastable with tensor_shape."""
    batch_size, = timestep_indices.shape
    assert tensor_shape[0] == batch_size
    gathered_values = torch.gather(torch.tensor(coeff_array, dtype=torch.float, device=timestep_indices.device), 0, timestep_indices.long())
    assert gathered_values.shape == (batch_size,)
    reshaped_output = gathered_values.reshape((batch_size,) + (1,) * (len(tensor_shape) - 1))
    return reshaped_output

def create_beta_schedule(start_beta, end_beta, total_timesteps):
    beta_values = np.linspace(start_beta, end_beta,
                        total_timesteps, dtype=np.float64)
    beta_tensor = torch.from_numpy(beta_values).float()
    return beta_tensor


def perform_denoising_step(current_x, noise_eps, current_t, next_t, *,
                            neural_models,
                            log_variances,
                            beta_schedule,
                            denoise_method='ddpm',
                            ddim_eta=0.0,
                            sigma_learning=False,
                            return_x0_pred=False,
                            ):
    neural_net = neural_models
    predicted_noise = neural_net(current_x, current_t)
    predicted_noise = predicted_noise.detach()
    if predicted_noise.shape != current_x.shape:
        predicted_noise, model_variance_vals = torch.split(predicted_noise, predicted_noise.shape[1] // 2, dim=1)
    if sigma_learning:
        predicted_noise, model_variance_vals = torch.split(predicted_noise, predicted_noise.shape[1] // 2, dim=1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        current_beta = gather_coefficients(beta_schedule, current_t, current_x.shape)
        current_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), current_t, current_x.shape)  # current_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)
        next_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), next_t, current_x.shape)  # next_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)
        posterior_var = current_beta * (1.0 - next_alpha_bar) / (1.0 - current_alpha_bar)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        min_log_val = torch.log(posterior_var.clamp(min=1e-6))
        max_log_val = torch.log(current_beta)
        interpolation_factor = (model_variance_vals + 1) / 2
        computed_logvar = interpolation_factor * max_log_val + (1 - interpolation_factor) * min_log_val
    else:
        computed_logvar = gather_coefficients(log_variances, current_t, current_x.shape)

    # Compute the next x
    current_beta = gather_coefficients(beta_schedule, current_t, current_x.shape)  # current_beta is the \beta_t
    current_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), current_t, current_x.shape)  # current_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)

    if next_t.sum() == -next_t.shape[0]:  # if next_t is -1
        next_alpha_bar = torch.ones_like(current_alpha_bar)
    else:
        next_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), next_t, current_x.shape)  # next_alpha_bar is the \hat{\alpha}_{t_next}

    next_x = torch.zeros_like(current_x)
    if denoise_method == 'ddpm':
        noise_weight = current_beta / torch.sqrt(1 - current_alpha_bar)

        predicted_mean = 1 / torch.sqrt(1.0 - current_beta) * (current_x - noise_weight * predicted_noise)
        random_noise = noise_eps
        timestep_mask = 1 - (current_t == 0).float()
        timestep_mask = timestep_mask.reshape((current_x.shape[0],) + (1,) * (len(current_x.shape) - 1))
        next_x = predicted_mean + timestep_mask * torch.exp(0.5 * computed_logvar) * random_noise
        next_x = next_x.float()

    elif denoise_method == 'ddim':
        predicted_x0 = (current_x - predicted_noise * (1 - current_alpha_bar).sqrt()) / current_alpha_bar.sqrt()  # predicted predicted_x0
        if ddim_eta == 0:
            next_x = next_alpha_bar.sqrt() * predicted_x0 + (1 - next_alpha_bar).sqrt() * predicted_noise
        else:
            eta_coeff1 = ddim_eta * ((1 - current_alpha_bar / (next_alpha_bar)) * (1 - next_alpha_bar) / (1 - current_alpha_bar)).sqrt()  # sigma_t
            eta_coeff2 = ((1 - next_alpha_bar) - eta_coeff1 ** 2).sqrt()  # direction pointing to x_t
            next_x = next_alpha_bar.sqrt() * predicted_x0 + eta_coeff2 * predicted_noise + eta_coeff1 * noise_eps

    if return_x0_pred == True:
        return next_x, predicted_x0
    else:
        return next_x



def calculate_noise_eps(current_x, next_x, current_t, next_t, neural_models, denoise_method, beta_schedule, log_variances, ddim_eta, sigma_learning):

    assert ddim_eta is None or ddim_eta > 0
    # Compute noise and variance
    if type(neural_models) != list:
        neural_net = neural_models
        predicted_noise = neural_net(current_x, current_t)
        if predicted_noise.shape != current_x.shape:
            predicted_noise, model_variance_vals = torch.split(predicted_noise, predicted_noise.shape[1] // 2, dim=1)
        if sigma_learning:
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            current_beta = gather_coefficients(beta_schedule, current_t, current_x.shape)
            current_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), current_t, current_x.shape)  # current_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)
            next_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), next_t, current_x.shape)  # next_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)
            posterior_var = current_beta * (1.0 - next_alpha_bar) / (1.0 - current_alpha_bar)
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            min_log_val = torch.log(posterior_var.clamp(min=1e-6))
            max_log_val = torch.log(current_beta)
            interpolation_factor = (model_variance_vals + 1) / 2
            computed_logvar = interpolation_factor * max_log_val + (1 - interpolation_factor) * min_log_val
        else:
            computed_logvar = gather_coefficients(log_variances, current_t, current_x.shape)
    else:
        raise NotImplementedError()

    # Compute the next x
    current_beta = gather_coefficients(beta_schedule, current_t, current_x.shape)  # current_beta is the \beta_t
    current_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), current_t, current_x.shape)  # current_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not next_t.sum() == -next_t.shape[0]  # next_t should never be -1
    assert not current_t.sum() == 0  # current_t should never be 0
    next_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), next_t, current_x.shape)  # next_alpha_bar is the \hat{\alpha}_{t_next}

    if denoise_method == 'ddpm':
        noise_weight = current_beta / torch.sqrt(1 - current_alpha_bar)

        predicted_mean = 1 / torch.sqrt(1.0 - current_beta) * (current_x - noise_weight * predicted_noise)
        print('torch.exp(0.5 * computed_logvar).sum()', torch.exp(0.5 * computed_logvar).sum())
        calculated_eps = (next_x - predicted_mean) / torch.exp(0.5 * computed_logvar)

    elif denoise_method == 'ddim':
        predicted_x0 = (current_x - predicted_noise * (1 - current_alpha_bar).sqrt()) / current_alpha_bar.sqrt()  # predicted predicted_x0

        eta_coeff1 = ddim_eta * ((1 - current_alpha_bar / (next_alpha_bar)) * (1 - next_alpha_bar) / (1 - current_alpha_bar)).sqrt()  # sigma_t
        eta_coeff2 = ((1 - next_alpha_bar) - eta_coeff1 ** 2).sqrt()  # direction pointing to x_t
        calculated_eps = (next_x - next_alpha_bar.sqrt() * predicted_x0 - eta_coeff2 * predicted_noise) / eta_coeff1
    else:
        raise ValueError()

    return calculated_eps


def generate_next_sample(clean_x0, current_x, current_t, next_t, denoise_method, beta_schedule, ddim_eta):
    current_beta = gather_coefficients(beta_schedule, current_t, current_x.shape)  # current_beta is the \beta_t
    current_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), current_t, current_x.shape)  # current_alpha_bar is the \hat{\alpha}_t (DDIM does not use \hat notation)

    assert not next_t.sum() == -next_t.shape[0]  # next_t should never be -1
    assert not current_t.sum() == 0  # current_t should never be 0
    next_alpha_bar = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), next_t, current_x.shape)  # next_alpha_bar is the \hat{\alpha}_{t_next}

    if denoise_method == 'ddpm':
        weight_x0 = next_alpha_bar.sqrt() * current_beta / (1 - current_alpha_bar)
        weight_xt = (1 - current_beta).sqrt() * (1 - next_alpha_bar) / (1 - current_alpha_bar)
        sampling_mean = weight_x0 * clean_x0 + weight_xt * current_x

        sampling_variance = current_beta * (1 - next_alpha_bar) / (1 - current_alpha_bar)

        next_sample = sampling_mean + sampling_variance.sqrt() * torch.randn_like(clean_x0)
    elif denoise_method == 'ddim':
        posterior_noise = (current_x - current_alpha_bar.sqrt() * clean_x0) / (1 - current_alpha_bar).sqrt()  # posterior posterior_noise given clean_x0 and current_x
        eta_coeff1 = ddim_eta * ((1 - current_alpha_bar / next_alpha_bar) * (1 - next_alpha_bar) / (1 - current_alpha_bar)).sqrt()  # sigma_t
        eta_coeff2 = ((1 - next_alpha_bar) - eta_coeff1 ** 2).sqrt()  # direction pointing to x_t
        next_sample = next_alpha_bar.sqrt() * clean_x0 + eta_coeff2 * posterior_noise + eta_coeff1 * torch.randn_like(clean_x0)
    else:
        raise ValueError()

    return next_sample


def add_noise_to_sample(clean_x0, timestep_t, beta_schedule):
    alpha_bar_t = gather_coefficients((1.0 - beta_schedule).cumprod(dim=0), timestep_t, clean_x0.shape)
    # print('alpha_bar_t', alpha_bar_t)
    noisy_sample = alpha_bar_t.sqrt() * clean_x0 + (1 - alpha_bar_t).sqrt() * torch.randn_like(clean_x0)
    return noisy_sample


class Noise_level(torch.nn.Module):

    def __init__(self, sampling_mode, num_custom_steps, encoding_steps, source_checkpoint_path=None, target_checkpoint_path=None,
                yaml_config_path=None, ddim_eta_param=None, max_timestep=None, t0=0):
        super(Noise_level, self).__init__()

        self.num_custom_steps = num_custom_steps
        self.sampling_mode = sampling_mode
        self.ddim_eta_param = ddim_eta_param
        self.max_timestep = max_timestep if max_timestep is not None else 999
        self.t_0 = t0
        self.encoding_steps = encoding_steps
        self.cuda_device = torch.device("cuda")
        if self.sampling_mode == 'ddim':
            assert self.ddim_eta_param > 0
        elif self.sampling_mode == 'ddpm':
            assert self.ddim_eta_param is None
        else:
            raise ValueError()

        diffusion_config = setup_ddpm_ddim_config(yaml_config_path)
        
        self.beta_schedule = create_beta_schedule(
            start_beta=diffusion_config.diffusion.beta_start,
            end_beta=diffusion_config.diffusion.beta_end,
            total_timesteps=diffusion_config.diffusion.num_diffusion_timesteps
        )
        self.total_timesteps = self.beta_schedule.shape[0]

        alpha_values = 1.0 - self.beta_schedule
        alpha_cumprod_vals = np.cumprod(alpha_values, axis=0)
        alpha_cumprod_prev_vals = np.append(1.0, alpha_cumprod_vals[:-1])
        posterior_var_vals = self.beta_schedule * (1.0 - alpha_cumprod_prev_vals) / (1.0 - alpha_cumprod_vals)
    
        self.source_generator = create_model(**Defalut_DICT)
        self.target_model = create_model(**Defalut_DICT)
        self.sigma_learning = False
        self.log_variance = np.log(np.maximum(posterior_var_vals, 1e-20))

        source_checkpoint = torch.load(source_checkpoint_path)
        target_checkpoint = torch.load(target_checkpoint_path)
        self.target_model.load_state_dict(source_checkpoint)
        self.source_generator.load_state_dict(target_checkpoint)
        self.target_model = self.target_model.cuda()
        self.source_generator = self.source_generator.cuda()

        self.image_resolution = diffusion_config.data.image_size
        self.image_channels = diffusion_config.data.channels
        self.encoding_dimension = self.image_resolution ** 2 * self.image_channels * self.encoding_steps

        self.output_processor = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])])

    def compute_eps(self, input_image):

        if (self.max_timestep + 1) % self.num_custom_steps == 0:
            reverse_sequence = range(0, self.max_timestep + 1, (self.max_timestep + 1) // self.num_custom_steps)
            assert len(reverse_sequence) == self.num_custom_steps
        else:
            reverse_sequence = np.linspace(0, 1, self.num_custom_steps) * self.max_timestep
        reverse_sequence = [int(step) for step in list(reverse_sequence)][:self.encoding_steps]
        next_sequence = ([-1] + list(reverse_sequence[:-1]))[:self.encoding_steps]

        input_image = (input_image - 0.5) * 2.0
        clean_image = input_image
        batch_sz = clean_image.shape[0]

        final_timestep = (torch.ones(batch_sz) * (self.encoding_steps - 1)).to(self.cuda_device)
        final_noisy_sample = add_noise_to_sample(clean_x0=clean_image, timestep_t=final_timestep, beta_schedule=self.beta_schedule)
        all_noise_estimates = [final_noisy_sample, ]

        current_sample = final_noisy_sample
        for iteration_idx, (current_step, next_step) in enumerate(zip(reversed(reverse_sequence), reversed(next_sequence))):
            current_timestep = (torch.ones(batch_sz) * current_step).to(self.cuda_device)
            next_timestep = (torch.ones(batch_sz) * next_step).to(self.cuda_device)

            if iteration_idx < self.encoding_steps - 1:
                next_sample = generate_next_sample(
                    clean_x0=clean_image,
                    current_x=current_sample,
                    current_t=current_timestep,
                    next_t=next_timestep,
                    denoise_method=self.sampling_mode,
                    beta_schedule=self.beta_schedule,
                    ddim_eta=self.ddim_eta_param,
                )
                noise_estimate = calculate_noise_eps(
                    current_x=current_sample,
                    next_x=next_sample,
                    current_t=current_timestep,
                    next_t=next_timestep,
                    neural_models=self.target_model,
                    denoise_method=self.sampling_mode,
                    beta_schedule=self.beta_schedule,
                    log_variances=self.log_variance,
                    ddim_eta=self.ddim_eta_param,
                    sigma_learning=self.sigma_learning,
                )
                current_sample = next_sample
                all_noise_estimates.append(noise_estimate)
            else:
                break

        stacked_noise_estimates = torch.stack(all_noise_estimates, dim=1).view(batch_sz, -1)
        return stacked_noise_estimates