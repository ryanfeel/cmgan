from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper, cal_si_snr

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*15, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='/nas/ryan/saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
parser.add_argument("--exp_name", type=str, default='default',
                    help="dir of saved model")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, gpu_id: int):
        self.n_fft = 800
        self.hop = 400
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.epoch = 0
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(gpu_id)
        # self.discriminator = discriminator.Discriminator(ndf=16).to(gpu_id)
        print(gpu_id)
        if gpu_id == 0:
            summary(
                self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
            )
            # summary(
            #     self.discriminator,
            #     [
            #         (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
            #         (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
            #     ],
            # )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        # self.optimizer_disc = torch.optim.AdamW(
        #     self.discriminator.parameters(), lr=2 * args.init_lr
        # )

        self.model = DDP(self.model, device_ids=[gpu_id])
        # self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def forward_generator_step(self, clean, clean2, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean, clean2 = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1), torch.transpose(clean2, 0, 1)
        noisy, clean, clean2 = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1), torch.transpose(clean2 * c, 0, 1)

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_spec2 = torch.stft(
            clean2,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        # print(noisy_spec.size())
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_spec2 = power_compress(clean_spec2)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        clean_real2 = clean_spec2[:, 0, :, :].unsqueeze(1)
        clean_imag2 = clean_spec2[:, 1, :, :].unsqueeze(1)

        est_real, est_imag, est_real2, est_imag2 = self.model(noisy_spec)
        est_real, est_imag, est_real2, est_imag2 = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2), est_real2.permute(0, 1, 3, 2), est_imag2.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        est_mag2 = torch.sqrt(est_real2**2 + est_imag2**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        clean_mag2 = torch.sqrt(clean_real2**2 + clean_imag2**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_uncompress2 = power_uncompress(est_real2, est_imag2).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        est_audio2 = torch.istft(
            est_spec_uncompress2,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        
        clean_uncompress = power_uncompress(clean_real, clean_imag).squeeze(1)
        clean_uncompress2 = power_uncompress(clean_real2, clean_imag2).squeeze(1)
        clean_audio = torch.istft(
            clean_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_audio2 = torch.istft(
            clean_uncompress2,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
            "est_real2": est_real2,
            "est_imag2": est_imag2,
            "est_mag2": est_mag2,
            "clean_real2": clean_real2,
            "clean_imag2": clean_imag2,
            "clean_mag2": clean_mag2,
            "est_audio2": est_audio2,
            "clean_audio": clean_audio,
            "clean_audio2": clean_audio2
        }

    def calculate_sisnr_loss_without_perm(self, generator_outputs, near_weight=0.5):
        clean_audio = generator_outputs["clean"]
        clean_audio2 = generator_outputs["clean2"]
        est_audio = generator_outputs["est_audio"]
        est_audio2 = generator_outputs["est_audio2"]

        clean_audio = clean_audio.unsqueeze(-1).permute(1, 0, 2)
        clean_audio2 = clean_audio2.unsqueeze(-1).permute(1, 0, 2)
        est_audio = est_audio.unsqueeze(-1).permute(1, 0, 2)
        est_audio2 = est_audio2.unsqueeze(-1).permute(1, 0, 2)
        
        loss1 = cal_si_snr(clean_audio, est_audio)
        loss2 = cal_si_snr(clean_audio2, est_audio2)
        
        loss = loss1 * near_weight + loss2 * (1 - near_weight)

        return loss

    def calculate_sisnr_loss_with_perm(self, generator_outputs):
        clean_audio = generator_outputs["clean"]
        clean_audio2 = generator_outputs["clean2"]
        est_audio = generator_outputs["est_audio"]
        est_audio2 = generator_outputs["est_audio2"]
        source = torch.stack([clean_audio, clean_audio2], -1)
        est = torch.stack([est_audio, est_audio2], -1)
        loss = get_si_snr_with_pitwrapper(source, est)

        return loss

    def calculate_scheduling_loss(self, generator_outputs):
        if self.epoch < 10:
            loss = self.calculate_sisnr_loss_with_perm(generator_outputs)
        elif self.epoch < 20:
            mag_loss = self.calculate_mag_loss(generator_outputs)
            loss = self.calculate_sisnr_loss_without_perm(generator_outputs, 0.5)
            loss = loss + mag_loss
        elif self.epoch < 30:
            mag_loss = self.calculate_mag_loss(generator_outputs, 0.8)
            loss = self.calculate_sisnr_loss_without_perm(generator_outputs, 0.8)
            loss = loss + mag_loss * 5.0
        else:
            mag_loss = self.calculate_mag_loss(generator_outputs, 1.0)
            loss = self.calculate_sisnr_loss_without_perm(generator_outputs, 1.0)
            loss = loss + mag_loss * 10.0

        return loss
            

    def calculate_generator_loss(self, generator_outputs):

        # predict_fake_metric = self.discriminator(
        #     generator_outputs["clean_mag"], generator_outputs["est_mag"]
        # )
        # gen_loss_GAN = F.mse_loss(
        #     predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        # )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss_mag2 = F.mse_loss(
            generator_outputs["est_mag2"], generator_outputs["clean_mag2"]
        )
        loss_ri2 = F.mse_loss(
            generator_outputs["est_real2"], generator_outputs["clean_real2"]
        ) + F.mse_loss(generator_outputs["est_imag2"], generator_outputs["clean_imag2"])

        time_loss2 = torch.mean(
            torch.abs(generator_outputs["est_audio2"] - generator_outputs["clean2"])
        )

        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
            # + args.loss_weights[3] * gen_loss_GAN
        )
        loss2 = (
            args.loss_weights[0] * loss_ri2
            + args.loss_weights[1] * loss_mag2
            + args.loss_weights[2] * time_loss2
            # + args.loss_weights[3] * gen_loss_GAN
        )
        loss = loss * 0.8 + loss2 * 0.2

        return loss

    def calculate_mag_loss(self, generator_outputs, near_weight=0.5):
        loss1 = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )

        loss2 = F.mse_loss(
            generator_outputs["est_mag2"], generator_outputs["clean_mag2"]
        )

        loss = loss1 * near_weight + loss2 * (1 - near_weight)

        return loss

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score.to(self.gpu_id))
        else:
            discrim_loss_metric = None

        return discrim_loss_metric
        
    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        clean2 = batch[1].to(self.gpu_id)
        noisy = batch[2].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            clean2,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean
        generator_outputs["clean2"] = clean2

        # loss = self.calculate_generator_loss(generator_outputs)
        loss = self.calculate_scheduling_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        # discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        # if discrim_loss_metric is not None:
        #     self.optimizer_disc.zero_grad()
        #     discrim_loss_metric.backward()
        #     self.optimizer_disc.step()
        # else:
        #     discrim_loss_metric = torch.tensor([0.0])

        return loss.item() #, discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        clean2 = batch[1].to(self.gpu_id)
        noisy = batch[2].to(self.gpu_id)
        one_labels = torch.ones(args.batch_size).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            clean2,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean
        generator_outputs["clean2"] = clean2

        # loss = self.calculate_generator_loss(generator_outputs)
        loss = self.calculate_scheduling_loss(generator_outputs)

        # discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        # if discrim_loss_metric is None:
        #     discrim_loss_metric = torch.tensor([0.0])

        return loss.item() #si_snr1.item(), si_snr2.item() #, discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        # self.discriminator.eval()
        gen_loss_total = 0.0
        # disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            # loss, disc_loss = self.test_step(batch)
            loss = self.test_step(batch)
            gen_loss_total += loss
            # disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        # disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}"
        # logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))
        logging.info(template.format(self.gpu_id, gen_loss_avg))

        return gen_loss_avg

    def train(self, writer):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        # scheduler_D = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        # )
        for epoch in range(args.epochs):
            self.epoch = int(epoch)
            self.model.train()
            # self.discriminator.train()
            step_num = len(self.train_ds)
            
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                # loss, disc_loss = self.train_step(batch)
                loss = self.train_step(batch)
                if self.gpu_id == 0:
                    writer.add_scalar("loss", loss, (step_num * epoch) + step)
                # writer.add_scalar("loss_disc", disc_loss, step)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}"
                # logging.info(
                #         template.format(self.gpu_id, epoch, (step_num * epoch) + step, loss)
                #     )
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss)
                    )
            gen_loss = self.test()
            path = os.path.join(
                args.save_model_dir, args.exp_name,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            model_dir = os.path.join(args.save_model_dir, args.exp_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), path)
            scheduler_G.step()
            # scheduler_D.step()


def main(rank: int, world_size: int, args):
    writer = SummaryWriter(comment=args.exp_name)
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 2, args.cut_len
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train(writer)
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    world_size = 4
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
