import os
import sys
import librosa
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import scipy
import numpy.fft as fft
from torch.utils.data import DataLoader
from random import choice
from sklearn import metrics
import csv
from tqdm import tqdm
from pydantic import BaseModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.base_model import BaseModel
from networks.dcase2023t2_ae.network import AENet
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_anm_score import AnmScoreFigData
from tools.plot_loss_curve import csv_to_figdata

import torch.nn as nn

class PitchShiftPredictor(nn.Module):
    def __init__(self, input_dim):
        super(PitchShiftPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output one value for pitch shift steps
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DCASE2023T2AE(BaseModel):
    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        self.args.epochs = 100
        parameter_list = [{"params": self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_list, lr=self.args.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)
        self.early_stopping_patience = 5  # Number of epochs to wait for improvement
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.pitch_predictor = PitchShiftPredictor(input_dim=640).to(self.device)
        self.pitch_optimizer = optim.Adam(self.pitch_predictor.parameters(), lr=self.args.learning_rate)
        self.mse_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mse.pickle"
        self.mahala_score_distr_file_path = self.model_dir / f"score_distr_{self.args.model}_{self.args.dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}_mahala.pickle"

    def init_model(self):
        self.block_size = self.data.height
        return AENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    def train(self, epoch):
        if epoch <= self.epoch or self.early_stop:
            return
        torch.autograd.set_detect_anomaly(True)
        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []
        train_loader = self.train_loader

        is_calc_cov = False
        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            cov_x_source = np.zeros((self.block_size, self.block_size))
            cov_x_source = torch.from_numpy(cov_x_source).to(self.device).float()
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            self.pitch_predictor.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0]
            data = data.to(self.device).float()
            data = data.cpu().numpy()
            shifted_data = np.zeros_like(data)
            #features = np.array([extract_features(x) for x in data])
            #features = torch.tensor(features).to(self.device).float()
            pitch_shift_steps = self.pitch_predictor(torch.tensor(data).to(self.device)).detach().cpu().numpy()
            #print("Pitch steps: ", pitch_shift_steps)
            
            for i in range(data.shape[0]):
                shifted_data[i] = librosa.effects.pitch_shift(data[i], sr=16000, n_steps=pitch_shift_steps[i])

            data = torch.from_numpy(shifted_data).to(self.device).float()
            if data.shape[0] <= 1:
                continue
            data_name_list = batch[3]
            machine_id = torch.argmax(batch[2], dim=1).long().to(self.device)
            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            if not is_calc_cov:
                self.optimizer.zero_grad()
                self.pitch_optimizer.zero_grad()
            recon_batch, z = self.model(data)
            if is_calc_cov:
                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list
                )
                cov_x_source_batch = cov_v(diff=cov_diff_source, num=1)
                cov_x_source += cov_x_source_batch.clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target_batch = cov_v(diff=cov_diff_target, num=1)
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target
            else:
                score_2d = self.loss_fn(recon_batch, data)
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)
            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target) if n_target > 0 else 0

            self.loss = recon_loss
            if not is_calc_cov:
                self.loss.backward()
                self.optimizer.step()
                self.pitch_optimizer.step()
            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)
            y_pred.append(self.loss.item())
            if batch_idx % self.args.log_interval == 0 and not is_calc_cov:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), self.loss.item()))
        if is_calc_cov:
            cov_x_source /= num_source - 1
            cov_x_target = cov_x_source.clone().detach() if num_target == 0 else cov_x_target / (num_target - 1)
            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target
            inv_cov_source, inv_cov_target = calc_inv_cov(self.model, self.device)
            y_pred_mahala = []
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                y_pred_mahala = self.calc_valid_mahala_score(data=batch[0], y_pred=y_pred_mahala, inv_cov_source=inv_cov_source, inv_cov_target=inv_cov_target)
            for batch_idx, batch in enumerate(self.valid_loader):
                y_pred_mahala = self.calc_valid_mahala_score(data=batch[0], y_pred=y_pred_mahala, inv_cov_source=inv_cov_source, inv_cov_target=inv_cov_target)
            self.fit_anomaly_score_distribution(y_pred=y_pred_mahala, score_distr_file_path=self.mahala_score_distr_file_path)

        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            self.pitch_predictor.eval()
            for batch_idx, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                recon_batch, _ = self.model(data)
                score = self.loss_fn(recon_batch, data)
                loss = score.mean()
                val_loss += float(loss)
        val_loss /= len(self.valid_loader)
        self.scheduler.step(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), self.model_path)
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.loss}, self.checkpoint_path)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping triggered")
                self.early_stop = True

        if not is_calc_cov:
            print('====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader), val_loss))
            with open(self.log_path, 'a') as log:
                np.savetxt(log, ["{0},{1},{2},{3},{4}".format(
                    train_loss / len(train_loader), val_loss, train_recon_loss / len(train_loader),
                    train_recon_loss_source / len(train_loader), train_recon_loss_target / len(train_loader))], fmt="%s")
            csv_to_figdata(file_path=self.log_path, column_heading_list=self.column_heading_list,
                        ylabel="loss", fig_count=len(self.column_heading_list), cut_first_epoch=True)
            self.fit_anomaly_score_distribution(y_pred=y_pred, score_distr_file_path=self.mse_score_distr_file_path)
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.loss}, self.checkpoint_path)

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)
        loss_source, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                cov=inv_cov_source, use_precision=True, reduction=False)
        loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
        loss_target, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size,
                                                cov=inv_cov_target, use_precision=True, reduction=False)
        loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def test(self):
        anm_score_figdata = AnmScoreFigData()
        mode = self.data.mode
        csv_lines = []
        block_size = self.data.height
        if mode:
            performance_over_all = []
            performance = []
        print("============== MODEL LOAD ==============")
        if not os.path.exists(self.model_path):
            print(f"model not found -> {self.model_path} ")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        if self.args.score == "MAHALA":
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mahala_score_distr_file_path)
        else:
            decision_threshold = self.calc_decision_threshold(score_distr_file_path=self.mse_score_distr_file_path)
        dir_name = "test"
        inv_cov_source, inv_cov_target = calc_inv_cov(self.model, self.device)
        for idx, test_loader_tmp in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
            anomaly_score_csv = result_dir / f"anomaly_score_{self.args.dataset}{section_name}{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            anomaly_score_list = []
            decision_result_csv = result_dir / f"decision_result_{self.args.dataset}{section_name}{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv"
            decision_result_list = []
            domain_list = None
            if mode:
                domain_list = []
            print("\n============== BEGIN TEST FOR A SECTION ==============")
            y_pred = []
            y_true = []
            test_loader = test_loader_tmp
            with torch.no_grad():
                y_pred, anomaly_score_list, decision_result_list, domain_list = self.eval(
                    test_loader=test_loader, y_pred=y_pred, anomaly_score_list=anomaly_score_list,
                    decision_result_list=decision_result_list, domain_list=domain_list, y_true=y_true,
                    decision_threshold=decision_threshold, mode=mode, inv_cov_source=inv_cov_source, inv_cov_target=inv_cov_target)
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print("anomaly score result ->  {}".format(anomaly_score_csv))
            save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
            print("decision result ->  {}".format(decision_result_csv))
            if mode:
                y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
                y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
                y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
                y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
                y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
                y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source"]
                y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]
                y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target"]
                auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=self.args.max_fpr)
                tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
                prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
                recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
                f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)
                anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                    scores=[[t, p] for t, p in zip(y_true_s, y_pred_s)], title=f"{section_name}_source_AUC{auc_s}"
                ))
                print("AUC (source) : {}".format(auc_s))
                print("pAUC : {}".format(p_auc))
                print("pAUC (source) : {}".format(p_auc_s))
                print("precision (source) : {}".format(prec_s))
                print("recall (source) : {}".format(recall_s))
                print("F1 score (source) : {}".format(f1_s))
                if len(y_true_t) > 0:
                    auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
                    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=self.args.max_fpr)
                    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]).ravel()
                    prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
                    recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
                    f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["source_target"])
                    csv_lines.append([section_name.split("_", 1)[1], auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    performance_over_all.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
                    anm_score_figdata.append_figdata(anm_score_figdata.anm_score_to_figdata(
                        scores=[[t, p] for t, p in zip(y_true_t, y_pred_t)], title=f"{section_name}_target_AUC{auc_t}"
                    ))
                    print("AUC (target) : {}".format(auc_t))
                    print("pAUC (target) : {}".format(p_auc_t))
                    print("precision (target) : {}".format(prec_t))
                    print("recall (target) : {}".format(recall_t))
                    print("F1 score (target) : {}".format(f1_t))
                else:
                    if len(csv_lines) == 0:
                        csv_lines.append(self.result_column_dict["single_domain"])
                    csv_lines.append([section_name.split("_", 1)[1], auc_s, p_auc, prec_s, recall_s, f1_s])
                    performance.append([auc_s, p_auc, prec_s, recall_s, f1_s])
                    performance_over_all.append([auc_s, p_auc, prec_s, recall_s, f1_s])
            print("\n============ END OF TEST FOR A SECTION ============")
        if mode:
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])
            anm_score_figdata.show_fig(
                title=self.args.model + "_" + self.args.dataset + self.model_name_suffix + self.eval_suffix + "_anm_score",
                export_dir=result_dir
            )
        else:
            return
        result_path = result_dir / f"result_{self.args.dataset}_{dir_name}_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}_roc.csv"
        print("results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)

    def eval(self, test_loader, y_pred, anomaly_score_list, decision_result_list, domain_list, y_true, decision_threshold, mode, inv_cov_source, inv_cov_target, pitch_shift_steps=0):
        for j, batch in enumerate(test_loader):
            data = batch[0].to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]
            recon_data, _ = self.model(data)
            if self.args.score == "MAHALA":
                loss_source, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size, cov=inv_cov_source, use_precision=True, reduction=False)
                loss_source = self.loss_reduction(score=self.loss_reduction_1d(loss_source), n_loss=num)
                loss_target, num = loss_function_mahala(recon_x=recon_data, x=data, block_size=self.block_size, cov=inv_cov_target, use_precision=True, reduction=False)
                loss_target = self.loss_reduction(score=self.loss_reduction_1d(loss_target), n_loss=num)
                y_pred.append(min(loss_target.item(), loss_source.item()))
            else:
                y_pred.append(self.loss_fn(recon_x=recon_data, x=data).mean().item())
            anomaly_score_list.append([basename, y_pred[-1]])
            if y_pred[-1] > decision_threshold:
                decision_result_list.append([basename, 1])
            else:
                decision_result_list.append([basename, 0])
            if mode:
                domain_list.append("target" if "target" in basename else "source")
        return y_pred, anomaly_score_list, decision_result_list, domain_list

def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)
