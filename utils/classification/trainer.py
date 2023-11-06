import torch
import pandas as pd

import os
from tqdm import tqdm
import time
import gc
import wandb


class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class Trainer:
    def __init__(self, 
                 model, 
                 path_output,
                 experiment,
                 criteriation,
                 train_dataloader=None, 
                 val_dataloader=None,
                 test_dataloader=None,
                 optimizer=None,
                 epochs=None,
                 early_stopping=False,
                 device = 'cuda:0',
                 log = True,
                 schedule=None
                 ) -> None:

        self.model = model.to(device)

        self.path_output = path_output
        self.experiment = experiment

        if self.path_output:
            if not os.path.exists(self.path_output):
                os.mkdir(self.path_output)


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.criteriation  = criteriation 
        self.optimizer = optimizer
        self.epochs = epochs

        self.early_stopping = early_stopping

        self.device = device
        self.log = log
        self.schedule=schedule

    def train(self, checkpoint=False):

        if checkpoint:
            epoch_saved = self.load_checkpoint(os.path.join(self.path_output, f"{self.experiment}.pth"))
        else:
            epoch_saved =  0

        val_acc_current = float('-inf')
        val_acc_early_stoppong= float('-inf')

        for epoch in range(epoch_saved, self.epochs+1):
            train_loss, train_acc, train_time_on_epooch = self.perform_epoch(is_train=True, is_test=False, epoch_number=epoch)
            val_loss, val_acc, _ = self.perform_epoch(is_train=False, is_test=False)
            print(f"TRAIN:      Epoch - {epoch} : loss {train_loss}, accuracy {train_acc}, time: {train_time_on_epooch}")
            print(f"VALIDATION: Epoch - {epoch} : loss {val_loss}, accuracy {val_acc}")


            if self.log:

                wandb.log({"TRAIN loss": train_loss, "VALIDATION loss": val_loss, "epoch": epoch})
                wandb.log({"TRAIN accuracy": train_acc, "VALIDATION accuracy": val_acc, "epoch": epoch})
                
            if (self.path_output is not None) and (val_acc > val_acc_current):
                self.save_checkpoint(os.path.join(self.path_output, f"{self.experiment}.pth"), epoch)
                val_acc_current = val_acc

            if self.early_stopping and (abs(val_acc - val_acc_early_stoppong) < 1e-5):
                print('EARLY STOPPING')
                break

            val_acc_early_stoppong = val_acc

            if self.schedule:
                self.schedule.step()
                _lr = [group['lr'] for group in self.optimizer.param_groups][0]
                print('SCHEDULE: ', _lr)

            print()



    def validate(self, checkpoint=False):

        if checkpoint:
            self.load_checkpoint(os.path.join(self.path_output, f"{self.experiment}.pth"))

        loss, acc, time_on_epooch = self.perform_epoch(is_train=False, is_test=False)
        print(f"Final: loss {loss}, accuracy {acc}, time: {time_on_epooch}\n")


    def test(self, checkpoint=None):

        if checkpoint:
            _= self.load_checkpoint(os.path.join(self.path_output, f"{self.experiment}.pth"))


        val_pred = self.perform_epoch(is_train=False, is_test=True, stage_test=False) # val set
        test_pred = self.perform_epoch(is_train=False, is_test=True, stage_test=True) # test set

        output = pd.DataFrame(columns=['id', 'pred'])
        output['pred'] = val_pred + test_pred
        
        output.to_csv(f'./prediction/{self.experiment}.csv', index=False)

    

    def perform_epoch(self, is_train: bool, is_test: bool, stage_test=False, epoch_number=0) -> tuple:
        if is_train:
            stage='train'

            self.model.train()
            torch.set_grad_enabled(True)
            loader = self.train_dataloader
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

            if stage_test:
                stage='test'
                loader = self.test_dataloader

            else:
                stage='val'
                loader = self.val_dataloader


        total_loss = 0
        total_acc = 0
        total_n = 0

        # -------------------------CUSTOM-----------------------------------------
        predictions = []
        instance = []
        # ------------------------------------------------------------------------
        

        start_time = time.time()
        with EmptyContext() if is_train else torch.no_grad():
            for batch_data, batch_labels in tqdm(loader, total=len(loader), desc=f"epoch: {str(epoch_number).zfill(3)} | {stage:5}"):
                batch_data = batch_data.to(self.device,  dtype=torch.float)

                if not stage_test:
                    batch_labels = batch_labels.to(self.device)
                else:
                    batch_labels = torch.ones((batch_data.shape[0]),  dtype=torch.long).to(self.device)
                    instance.extend(list(batch_labels))

                model_output = self.model(batch_data)
                model_prediction = torch.max(model_output, 1)[1]

                new_loss = self.criteriation(model_output, batch_labels)
                
                if is_train:
                    self.optimizer.zero_grad()
                    new_loss.backward()
                    self.optimizer.step()

                predictions.extend(model_prediction.cpu().detach().numpy().tolist())

                one_batch_loss = float(self.criteriation(model_output, batch_labels))

                one_batch_acc = self.metrics(model_prediction, batch_labels)

                total_loss += one_batch_loss
                total_acc += one_batch_acc
                total_n += 1

        
        output =  (total_loss / total_n, total_acc / total_n, (time.time() - start_time)/60)

        gc.collect()
        torch.cuda.empty_cache() 

        return output
    

    def save_checkpoint(self, filename, EPOCH):
            with open(filename, "wb") as fp:
                torch.save({
                    'epoch': EPOCH,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, filename)
                

    def load_checkpoint(self, filename):
        with open(filename, "rb") as fp:
            checkpoint = torch.load(fp, map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']

    
    
    def metrics(self, model_prediction: torch.tensor, batch_labels: torch.tensor) -> float:
        return torch.mean((model_prediction == batch_labels).float())
