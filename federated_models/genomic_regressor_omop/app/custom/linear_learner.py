import copy
import os
import re
import pickle
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from typing import Union
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

from linear_regressor import LinearKernelRegressor, MSEWithRegularization
from logistic_regression import LogisticRegressor, BCEWithRegularization
from genomic_datamanager import GenomicDataset, generate_datasets_from_OMOP

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType

#SPLIT_PER=0.8
#SEED=0

#GENOMIC_DATA_PATH = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-wide.csv")
#GENOMIC_LABELS_PATH = os.path.join("/workspace", "DEMO", "genomic_regressor", "data", "CNAE-9-labels.csv")

# training_data_path = "/home/data/training_data_client.csv"
# validation_data_path = "/home/data/validation_data_client.csv" 

# Will be set during initialize()
training_data_path = None
validation_data_path = None
input_dim = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Linear_Learner(Learner): 
    def __init__(
        self,
        train_idx: Union[str, None] = None,
        aggregation_epochs: int = 2,
        lr: float = 0.001,
        l2: float = 0.01,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 16,
        num_workers: int = 4
        ):
        f"""Simple trainer for x-ray chest images of COVID patients

        Args:
            train_idx: file with site training indices.
            aggregation_epochs: the number of training epochs for a round. Defaults to 4.
            lr: local learning rate. Float number. Defaults to 1e-3.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.train_idx = train_idx
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.l2 = l2
        self.best_loss = 100000
        self.acc = 0
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.transform = None
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.training_data_path = None
        self.validation_data_path = None
        self.input_dim = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        """
        Note: this code assumes a FL simulation setting
        Datasets will be initialized in train() and validate() when calling self._create_datasets()
        as we need to make sure that the server has already downloaded and split the data.
        """
        # Generate datasets from OMOP database
        self.training_data_path, self.validation_data_path = generate_datasets_from_OMOP()
        
        if self.training_data_path is None or self.validation_data_path is None:
            raise ValueError("Failed to generate datasets from OMOP database. Check database connection and data availability.")
        
        # Load training dataset to determine input dimension
        train_dataset = GenomicDataset(data_file=self.training_data_path)
        self.input_dim = train_dataset.data.shape[1]
             
        # when the run starts, this is where the actual settings get initialized for trainer

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         
        # self.model = LinearKernelRegressor(input_dim=self.input_dim).to(self.device)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.criterion = MSEWithRegularization(self.model, lambda_l2=self.l2)
        self.model = LogisticRegressor(input_dim=self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = BCEWithRegularization(self.model, lambda_l2=self.l2)


    def _create_datasets(self, fl_ctx: FLContext):
        """This is the function that splits data into servers."""
        # Here the data are mixed and then splitted between training and validation datasets
        # SEED = int(re.search(r'\d+', self.client_id).group())
        # torch.manual_seed(SEED)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(SEED)
        #     torch.cuda.manual_seed_all(SEED)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        # training_samples = 1000 
        # validation_samples = 200
        # X_train = torch.rand((training_samples, RES))
        # y_train = torch.randint(0, 2, (training_samples,))  
        # X_valid = torch.rand((validation_samples, RES))
        # y_valid = torch.randint(0, 2, (validation_samples,))
        if self.train_dataset is None or self.train_loader is None:             
            #self.train_dataset = DBDatamanager("genomic_result", fields="VCF2matrix", transform=self.transform, 
            #                                   split='train', split_perc=SPLIT_PER, shuffle=True, onehot=onehot, 
            #                                   labels=None)
            #self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, 
            #                               shuffle=True, num_workers=self.num_workers
            #                               )
            self.train_dataset = GenomicDataset(data_file=self.training_data_path)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=self.num_workers
                                          )  
        if self.valid_dataset is None or self.valid_loader is None:
            #self.valid_dataset = DBDatamanager("genomic_result", fields="VCF2matrix", transform=self.transform, 
            #                                   split='valid', split_perc=SPLIT_PER, shuffle=False, onehot=onehot, 
            #                                   labels=None)                                               
            #self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, 
            #                               shuffle=False, num_workers=self.num_workers
            #                               )
            self.valid_dataset = GenomicDataset(data_file=self.validation_data_path)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=self.num_workers
                                           )
            
    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(self, fl_ctx, train_loader, model_global, abort_signal: Signal, val_freq: int = 0, current_round=0):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            self.log_info(fl_ctx, f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})")
            avg_loss = 0.
            
            # Initialize lists to store predictions and labels for confusion matrix
            all_preds = []
            all_labels = []
            
            # Calculate pos_weight for this batch
            all_batch_labels = []
            for inputs, labels in train_loader:
                all_batch_labels.extend(labels.numpy())
            neg_count = sum(1 for x in all_batch_labels if x == 0)
            pos_count = sum(1 for x in all_batch_labels if x == 1)
            pos_weight = torch.tensor([neg_count / pos_count]).to(self.device)
            
            # Update criterion with pos_weight
            self.criterion = BCEWithRegularization(self.model, lambda_l2=self.l2, pos_weight=pos_weight)
            
            for i, (inputs, labels) in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())   

                # Store predictions and labels for confusion matrix
                preds = (outputs > 0).float().cpu().numpy()  # No need for sigmoid
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                results_file = "/home/train_loss.txt"
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                with open(results_file, "a") as f:
                    f.write(f"Loss: {loss}\n")
                    f.write(f"Round: {current_round}\n")
                    f.write(f"Epoch: {epoch}\n")
                    f.write("\n\n")

                loss.backward()
                self.optimizer.step()
                
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            
            # Calculate confusion matrix at the end of each epoch
            conf_matrix = confusion_matrix(all_labels, all_preds)
            tn, fp, fn, tp = conf_matrix.ravel()
            
            # Write confusion matrix to file
            with open("/home/train_loss.txt", "a") as f:
                f.write(f"Training Confusion Matrix (Round {current_round}, Epoch {epoch}):\n")
                f.write(f"TN: {tn}, FP: {fp}\n")
                f.write(f"FN: {fn}, TP: {tp}\n")
                f.write("-" * 50 + "\n\n")

    def save_model(self, is_best=False):
        # save model
        if is_best:
            print("SAVING BEST MODEL INTO:", self.best_local_model_file)
            # Move model to CPU before saving
            model_to_save = self.model.to('cpu')
            torch.save(model_to_save, self.best_local_model_file)
            self.model = self.model.to(self.device)  # Move back to original device
        else:
            print("SAVING MODEL INTO:", self.local_model_file)
            # Move model to CPU before saving
            model_to_save = self.model.to('cpu')
            torch.save(model_to_save, self.local_model_file)
            self.model = self.model.to(self.device)  # Move back to original device



    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self._create_datasets(fl_ctx)

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference 
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=0,
            current_round=current_round
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        valloss = self.local_valid(self.valid_loader, abort_signal, tb_id="val_loss_local_model", fl_ctx=fl_ctx)
        
        # Write validation results to file
        results_file = "/home/val_regression.txt"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Create directory if it doesn't exist
        with open(results_file, "a") as f:  # 'a' for append mode
            f.write(f"\nRound {current_round + 1}\n")
            f.write(f"VALLOSS: {valloss}\n")
            f.write("-" * 50 + "\n")  # Add a separator line
        
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_loss_local_model: MSE={valloss['mse']:.4f}, "
                             f"Accuracy={valloss['accuracy']:.4f}, "
                             f"Precision={valloss['precision']:.4f}, "
                             f"Recall={valloss['recall']:.4f}, "
                             f"F1={valloss['f1']:.4f}")
        self.log_info(fl_ctx, f"Confusion Matrix:\n"
                             f"TN: {valloss['confusion_matrix']['true_negatives']}, "
                             f"FP: {valloss['confusion_matrix']['false_positives']}\n"
                             f"FN: {valloss['confusion_matrix']['false_negatives']}, "
                             f"TP: {valloss['confusion_matrix']['true_positives']}")

        # save model
        self.save_model(is_best=False)
        if valloss['accuracy'] > self.acc:
            self.acc = valloss['accuracy']
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except BaseException as e:
                raise ValueError("Unable to load best model") from e

            # Create DXO and shareable from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_weights)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(fl_ctx, f"best local model not found at {self.best_local_model_file}.")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.
            all_preds = []
            all_labels = []
            for _i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                val_loss += loss.item()
                
                # Convert regression outputs to binary predictions using sigmoid then threshold at 0.5
                probs = torch.sigmoid(outputs.data)
                preds = (probs > 0.5).float().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
            val_loss /= len(valid_loader)
            
            # Calculate classification metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_preds)
            tn, fp, fn, tp = conf_matrix.ravel()  # For binary classification
            
            print(f"{self.client_id} MSE: {val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Confusion Matrix:")
            print(f"TN: {tn}, FP: {fp}")
            print(f"FN: {fn}, TP: {tp}")
            
            if tb_id:
                self.writer.add_scalar(f"{tb_id}_mse", val_loss, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_accuracy", accuracy, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_precision", precision, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_recall", recall, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_f1", f1, self.epoch_global)
                # Add confusion matrix metrics
                self.writer.add_scalar(f"{tb_id}_true_negatives", tn, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_false_positives", fp, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_false_negatives", fn, self.epoch_global)
                self.writer.add_scalar(f"{tb_id}_true_positives", tp, self.epoch_global)
            
            return {
                "mse": val_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                }
            }

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self._create_datasets(fl_ctx)

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "global_model"

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except BaseException as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            metrics = self.local_valid(self.valid_loader, abort_signal, tb_id="val_error_global_model", fl_ctx=fl_ctx)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_error_global_model ({model_owner}): MSE={metrics['mse']:.4f}, "
                                 f"Accuracy={metrics['accuracy']:.4f}, "
                                 f"Precision={metrics['precision']:.4f}, "
                                 f"Recall={metrics['recall']:.4f}, "
                                 f"F1={metrics['f1']:.4f}")
            self.log_info(fl_ctx, f"Confusion Matrix ({model_owner}):\n"
                                 f"TN: {metrics['confusion_matrix']['true_negatives']}, "
                                 f"FP: {metrics['confusion_matrix']['false_positives']}\n"
                                 f"FN: {metrics['confusion_matrix']['false_negatives']}, "
                                 f"TP: {metrics['confusion_matrix']['true_positives']}")

            return DXO(data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: metrics}, meta={}).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            train_metrics = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training metrics ({model_owner}): MSE={train_metrics['mse']:.4f}, "
                                 f"Accuracy={train_metrics['accuracy']:.4f}, "
                                 f"Precision={train_metrics['precision']:.4f}, "
                                 f"Recall={train_metrics['recall']:.4f}, "
                                 f"F1={train_metrics['f1']:.4f}")
            self.log_info(fl_ctx, f"Training Confusion Matrix ({model_owner}):\n"
                                 f"TN: {train_metrics['confusion_matrix']['true_negatives']}, "
                                 f"FP: {train_metrics['confusion_matrix']['false_positives']}\n"
                                 f"FN: {train_metrics['confusion_matrix']['false_negatives']}, "
                                 f"TP: {train_metrics['confusion_matrix']['true_positives']}")

            val_metrics = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation metrics ({model_owner}): MSE={val_metrics['mse']:.4f}, "
                                 f"Accuracy={val_metrics['accuracy']:.4f}, "
                                 f"Precision={val_metrics['precision']:.4f}, "
                                 f"Recall={val_metrics['recall']:.4f}, "
                                 f"F1={val_metrics['f1']:.4f}")
            self.log_info(fl_ctx, f"Validation Confusion Matrix ({model_owner}):\n"
                                 f"TN: {val_metrics['confusion_matrix']['true_negatives']}, "
                                 f"FP: {val_metrics['confusion_matrix']['false_positives']}\n"
                                 f"FN: {val_metrics['confusion_matrix']['false_negatives']}, "
                                 f"TP: {val_metrics['confusion_matrix']['true_positives']}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {
                "train_error": train_metrics["mse"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_confusion_matrix": train_metrics["confusion_matrix"],
                "val_error": val_metrics["mse"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_confusion_matrix": val_metrics["confusion_matrix"]
            }

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)                 

if __name__=='__main__':
    print("Hey boy, fed and shut!") 
