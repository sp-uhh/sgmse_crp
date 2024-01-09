import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import torch.nn.functional as F
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import OUVESDE
from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec
from pesq import pesq
from wvmos import get_wvmos
from pystoi import stoi

from utils import energy_ratios, ensure_dir, print_mean_std

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




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, help="Name of destination folder")
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")

    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    wvmos_model = get_wvmos(cuda=True)
    checkpoint_file = args.ckpt

    target_dir = "/export/home/lay/PycharmProjects/ncsnpp/enhanced/{}/".format(
        args.type)

    ensure_dir(target_dir + "files/")

    # Settings
    sr = 16000
    N = args.N
    reverse_starting_point = args.reverse_starting_point


    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    sr = 16000
    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": [], "WVMOS": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)
        if x.shape[1] != y.shape[1]:
            len_min = min(x.shape[1], y.shape[1])
            x = x[:, :len_min]
            y = y[:, :len_min]

        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor
        x = x / norm_factor
        
        noise = y - x

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

        timesteps = torch.linspace(reverse_starting_point, 0.03, N, device=Y.device)
        std = model.sde._std(reverse_starting_point*torch.ones((Y.shape[0],), device=Y.device))
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
                mean_x_tm1 = X_t - (f - g**2*model.forward(X_t, vec_t, Y, vec_t[:,None,None,None]))*dt #mean of x t minus 1 = mu(x_{t-1})
                mean_gt, _ = model.sde.marginal_prob(X, torch.ones(Y.shape[0], device=Y.device) * (t-dt), Y)  
                if i == len(timesteps) - 1: #output
                    mean_gt, _ = model.sde.marginal_prob(X, torch.ones(Y.shape[0], device=Y.device) * (t-dt), Y)
                    X_t = mean_x_tm1 
                    break
                z = torch.randn_like(X) 
                #Euler Maruyama
                X_t = mean_x_tm1 + z*g*torch.sqrt(dt)


        sample = X_t
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().detach().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        wvmos = wvmos_model.calculate_one(target_dir + "files/" + filename)
        data["WVMOS"].append(wvmos)


    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))
        file.write("WVMOS: {} \n".format(print_mean_std(data["WVMOS"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("N: {}\n".format(N))
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))



