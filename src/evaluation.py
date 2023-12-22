import numpy as np
from CMGAN.src.models import generator
from natsort import natsorted
import os
from CMGAN.src.tools.compute_metrics import compute_metrics
import torchaudio
import soundfile as sf
import argparse
import torch
from .utils import power_compress, power_uncompress, power_compress_with_low_f_delete, power_uncompress_with_low_f_delete


@torch.no_grad()
def get_low_frequency(audio, n_fft=400, hop=100):

    audio = audio.cuda()

    c = torch.sqrt(audio.size(-1) / torch.sum((audio**2.0), dim=-1))
    audio = torch.transpose(audio, 0, 1)
    audio = torch.transpose(audio * c, 0, 1)

    audio_spec = torch.stft(
        audio, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )
    _, temp_mag, temp_phase = power_compress_with_low_f_delete(audio_spec)

    return temp_mag, temp_phase

@torch.no_grad()
def enhance_one_track_with_temp_lowf(
    model, noisy, n_fft=400, hop=100
):
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )

    noisy_spec, temp_mag, temp_phase = power_compress_with_low_f_delete(noisy_spec, delete_f_bin=8)

    noisy_spec = noisy_spec.permute(0, 1, 3, 2)

    est_real, est_imag, est_real2, est_imag2 = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_real2, est_imag2 = est_real2.permute(0, 1, 3, 2), est_imag2.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress_with_low_f_delete(est_real, est_imag, temp_mag, temp_phase).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio).cpu().numpy()

    est_spec_uncompress2 = power_uncompress_with_low_f_delete(est_real2, est_imag2, temp_mag, temp_phase).squeeze(1)
    est_audio2 = torch.istft(
        est_spec_uncompress2,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio2 = est_audio2 / c
    est_audio2 = torch.flatten(est_audio2).cpu().numpy()
    return est_audio, est_audio2

@torch.no_grad()
def enhance_one_track(
    model, noisy, n_fft=400, hop=100
):
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    # length = noisy.size(-1)
    # frame_num = int(np.ceil(length / 100))
    # padded_len = frame_num * 100
    # padding_len = padded_len - length
    # noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    # if padded_len > cut_len:
    #     batch_size = int(np.ceil(padded_len / cut_len))
    #     while 100 % batch_size != 0:
    #         batch_size += 1
    #     noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )
    noisy_spec = power_compress(noisy_spec)
    noisy_spec = noisy_spec.permute(0, 1, 3, 2)

    est_real, est_imag, est_real2, est_imag2 = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    est_real2, est_imag2 = est_real2.permute(0, 1, 3, 2), est_imag2.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio).cpu().numpy()

    est_spec_uncompress2 = power_uncompress(est_real2, est_imag2).squeeze(1)
    est_audio2 = torch.istft(
        est_spec_uncompress2,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio2 = est_audio2 / c
    est_audio2 = torch.flatten(est_audio2).cpu().numpy()
    return est_audio, est_audio2


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in audio_list:
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)
        est_audio, length = enhance_one_track(
            model, noisy_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks
        )
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
    )

def inference(model_path, noisy_path):
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    est_audio = enhance_one_track(
        model, noisy_path, 16000 * 16, n_fft, n_fft // 4
    )
    return est_audio


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir)
