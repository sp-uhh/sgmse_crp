import torch
from torchaudio import load
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
 


from .other import si_sdr, pad_spec

# Settings
sr = 16000


def evaluate_model(model, num_eval_files, inference_N, inference_start=0.5):

    
    N = inference_N
    reverse_start_time = inference_start
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)



    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        #requires only for BWE as the dataset has different length of clean and noisy files
        if x.shape[1] != y.shape[1]:
            len_min = min(x.shape[1], y.shape[1])
            x = x[:, :len_min]
            y = y[:, :len_min]

        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor
        x = x / norm_factor
        noise = y-x

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)
        X = pad_spec(X)
        
        Noise = torch.unsqueeze(model._forward_transform(model._stft(noise.cuda())), 0)
        Noise = pad_spec(Noise)      
        
        
        y = y * norm_factor
        x = x * norm_factor
        
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        total_loss = 0
        timesteps = torch.linspace(reverse_start_time, 0.03, N, device=Y.device)
        #prior sampling starting from reverse_start_time 
        std = model.sde._std(reverse_start_time*torch.ones((Y.shape[0],), device=Y.device))
        z = torch.randn_like(Y)
        X_t = Y + z * std[:, None, None, None]
        
        #reverse steps by Euler Maruyama
        for i in range(len(timesteps)):
            t = timesteps[i]
            if i != len(timesteps) - 1:
                dt = t - timesteps[i+1]
            else:
                dt = timesteps[-1]
            with torch.no_grad():
                #take Euler step here
                f, g = model.sde.sde(X_t, t, Y)
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
                score = model.forward(X_t, vec_t, Y, vec_t[:,None,None,None])
                mean_x_tm1 = X_t - (f - g**2*score)*dt #mean of x t minus 1 = mu(x_{t-1})
                if i == len(timesteps) - 1: #output
                    X_t = mean_x_tm1 
                    break
                z = torch.randn_like(X) 

                X_t = mean_x_tm1 + z*g*torch.sqrt(dt)

        sample = X_t
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)

    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, total_loss/num_eval_files


def convert_to_audio(X, deemp, T_orig, model, norm_factor):
        sample = X

        sample = sample.squeeze()
        if len(sample.shape)==4:
            sample = sample*deemp[None, None, :, None].to(device=sample.device)
        elif len(sample.shape)==3:
            sample = sample*deemp[None, :, None].to(device=sample.device)
        else:
            sample = sample*deemp[:, None].to(device=sample.device)
   
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        return x_hat
