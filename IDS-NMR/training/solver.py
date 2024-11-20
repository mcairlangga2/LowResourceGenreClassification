import os
import time
import numpy as np
from sklearn import metrics
import datetime
import sys
import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class Solver(object):
    def __init__(self, data_loader, valid_loader, config):
        # data loader
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard
        self.map_num = config.map_num
        self.pad_num = config.pad_num
        self.reprog_front = config.reprog_front
        if self.dataset == "gtzan":
            self.n_class = 9

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.build_model()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable parameters: {str(params)}")

        # Tensorboard
        self.writer = SummaryWriter()

    def get_model(self):      
        if self.model_type == "ast":
            from models.ASTModel import AST
            return AST(
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
            )

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr, weight_decay=1e-4
        )

    def to_var(self, x):
        if isinstance(x, dict):
            for d in x.keys():
                if torch.cuda.is_available():
                    x[d] = Variable(x[d]).cuda().squeeze()
            return x
        else:
            if torch.cuda.is_available():
                x = x.cuda()
            return Variable(x).squeeze()

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = "adam"
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0
        
        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        
        # Iterate over epochs
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for x, y in self.data_loader:
                ctr += 1
                # Convert x and y to tensors
                x = self.to_var(x)

                # Unpack y if it’s a tuple
                if isinstance(y, tuple):
                    y = y[0]
                y = self.to_var(y)

                # Forward pass
                out = self.model(x)

                # If out is a tuple, extract the first element
                if isinstance(out, tuple):
                    out = out[0]

                # Check for size compatibility between out and y
                if out.size() != y.size():
                    y = y.view_as(out)  # Adjust y shape to match out if necessary

                # Compute loss
                loss = reconst_loss(out, y)
                total_loss += loss.item()

                # Calculate predictions and accuracy
                if self.n_class == 1:  # Binary classification
                    predictions = (torch.sigmoid(out) > 0.5).float()
                else:  # Multi-class classification
                    predictions = torch.argmax(out, dim=1)
                    y = torch.argmax(y, dim=1)

                correct = (predictions == y).sum().item()
                total_correct += correct
                total_samples += y.size(0)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)

            # Calculate average training loss and accuracy
            avg_loss = total_loss / len(self.data_loader)
            training_accuracy = total_correct / total_samples

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/train", training_accuracy, epoch)

            print(f"Epoch [{epoch+1}/{self.n_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {training_accuracy:.4f}")

            # Validation
            val_acc, val_loss = self.get_validation_score(epoch)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("Validation loss improved. Saving model.")
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, "best_model.pth"))
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epoch(s).")

            # Stop training if validation loss does not improve for 'patience' epochs
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

            # Schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print(
            "[%s] Training finished. Elapsed time: %s"
            % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.timedelta(seconds=time.time() - start_t),
            )
        )


    def load(self, filename):
        S = torch.load(filename)
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in S.items() if k in model_dict and "delta" not in k
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == "adam" and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, "best_model.pth"))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                0.001,
                momentum=0.9,
                weight_decay=0.0001,
                nesterov=True,
            )
            current_optimizer = "sgd_1"
            drop_counter = 0
            print("sgd 1e-3")
        # first drop
        if current_optimizer == "sgd_1" and drop_counter == 50:
            self.load(os.path.join(self.model_save_path, "best_model.pth"))
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0005
            current_optimizer = "sgd_2"
            drop_counter = 0
            print("sgd 5e-4")

        # second drop
        if current_optimizer == "sgd_2" and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, "best_model.pth"))
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.00001
            current_optimizer = "sgd_3"
            print("sgd 1e-5")

        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({"model": model}, filename)

    def get_auc(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        print("roc_auc: %.4f pr_auc: %.4f" % (roc_aucs, pr_aucs))
        return roc_aucs, pr_aucs

    def get_acc(self, est_array, gt_array):
        acc = metrics.accuracy_score(gt_array, est_array)
        print("Validation Accuracy: %.4f" % acc)
        return acc

    def print_log(self, epoch, ctr, loss_bce, start_t):
        if (ctr) % self.log_step == 0:
            sys.stdout.write("\r")
            sys.stdout.write(
                "[%s] Epoch [%d/%d] Iter [%d/%d] train bce loss: %.4f Time: %s"
                % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    self.n_epochs,
                    ctr,
                    len(self.data_loader),
                    loss_bce.item(),
                    datetime.timedelta(seconds=time.time() - start_t),
                )
            )
            sys.stdout.flush()

    def validation(self, best_metric, epoch):
        acc, loss = self.get_validation_score(epoch)
        score = acc
        if score > best_metric:
            print("best model!")
            best_metric = score
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_save_path, "best_model.pth"),
            )
        return best_metric

    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        with torch.no_grad():
            est_array = []
            gt_array = []
            losses = []
            reconst_loss = self.get_loss_function()
            index = 0
            for x, y in tqdm.tqdm(self.valid_loader):
                x = self.to_var(x)
                
                # Unpack y if it’s a tuple
                if isinstance(y, tuple):
                    y = y[0]
                y = self.to_var(y).repeat(len(x), 1)
                
                out = self.model(x)

                # If out is a tuple, extract the first element
                if isinstance(out, tuple):
                    out = out[0]
                
                loss = reconst_loss(out, y)
                losses.append(float(loss.data))
                out = out.detach().cpu()

                estimated = np.array(out).mean(axis=0)
                est_array.append(estimated)

                gt_array.append(y.detach().cpu().numpy()[0])
                index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print("Validation loss: %.4f" % loss)

        acc = self.get_acc(np.argmax(est_array, axis=1),
                           np.argmax(gt_array, axis=1))
        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        self.writer.add_scalar("Loss/valid", loss, epoch)
        self.writer.add_scalar("Val ACC", acc, epoch)
        return acc, loss   


