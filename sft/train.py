import sys
sys.path.append(".")

from arguments import Arguments
from model import MCQAModel
from dataset import MCQADataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning import Trainer
import torch, os, sys
import pandas as pd
from tqdm import tqdm
import time, argparse

EXPERIMENT_DATASET_FOLDER = "/experiment/dataset"
WB_PROJECT = "pubmed_dataset_project"

def train(args,
          exp_dataset_folder,
          experiment_name,
          models_folder,
          version):

    pl.seed_everything(42)
    
    EXPERIMENT_FOLDER = os.path.join(models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    experiment_string = experiment_name+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'

    # wb_logger = WandbLogger(project=WB_PROJECT, name=experiment_name, version=version)
    csv_logger = CSVLogger(models_folder, name=experiment_name, version=version)

    train_dataset = MCQADataset(args.train_csv)
    test_dataset = MCQADataset(args.test_csv)
    val_dataset = MCQADataset(args.dev_csv)

    es_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=2,
                                    verbose=True,
                                    mode='min')

    cp_callback = ModelCheckpoint(monitor='val_loss',
                                  filename=experiment_string,
                                    dirpath=EXPERIMENT_FOLDER,
                                    save_top_k=1,
                                    save_weights_only=True,
                                    mode='min')

    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name,
                      args=args.__dict__)
    
    mcqaModel.prepare_dataset(train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              val_dataset=val_dataset)

    trainer = Trainer(accelerator="auto",
                    devices="auto",
                    logger=[csv_logger],
                    callbacks= [es_callback,cp_callback],
                    max_epochs=args.num_epochs,
                    log_every_n_steps=1)
    
    trainer.fit(mcqaModel)

    ckpt = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.ckpt')]

    inference_model = MCQAModel.load_from_checkpoint(os.path.join(EXPERIMENT_FOLDER, ckpt[0]))
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()

    test_results = trainer.test(ckpt_path=os.path.join(EXPERIMENT_FOLDER,ckpt[0]))
    print('Test results: ', test_results)
    csv_logger.log_metrics(test_results[0])

    #Persist test dataset predictions
    test_df = pd.read_csv(args.test_csv)
    test_df.loc[:, "predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.test_dataloader(), args)]
    test_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"test_results.csv"),index=False)
    print(f"Test predictions written to {os.path.join(EXPERIMENT_FOLDER,'test_results.csv')}")

    val_df = pd.read_csv(args.dev_csv)
    val_df.loc[:,"predictions"] = [pred+1 for pred in run_inference(inference_model,mcqaModel.val_dataloader(),args)]
    val_df.to_csv(os.path.join(EXPERIMENT_FOLDER,"dev_results.csv"),index=False)
    print(f"Val predictions written to {os.path.join(EXPERIMENT_FOLDER,'dev_results.csv')}")

    del mcqaModel
    del inference_model
    del trainer
    torch.cuda.empty_cache()
    
def run_inference(model, dataloader, args):
    predictions = []
    for idx, (inputs,labels) in tqdm(enumerate(dataloader)):
        batch_size = len(labels)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions

if __name__ == "__main__":
    models = ["allenai/scibert_scivocab_uncased", "bert-base-uncased", "FacebookAI/roberta-base", "mental/mental-bert-base-uncased"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="name of the model")
    parser.add_argument("--epoch", default=5, type=int, help="number of epochs")
    parser.add_argument("--dataset_folder_name", default="./final_dataset/big_splits", help="dataset folder containing train.csv, test.csv, val.csv")
    cmd_args = parser.parse_args()

    exp_dataset_folder =  cmd_args.dataset_folder_name
    model = cmd_args.model
    print(f"Training started for model - {model} variant - {exp_dataset_folder}")

    args = Arguments(train_csv=os.path.join(exp_dataset_folder,"train.csv"),
                    test_csv=os.path.join(exp_dataset_folder,"test.csv"),
                    dev_csv=os.path.join(exp_dataset_folder,"val.csv"),
                    pretrained_model_name=model,
                    use_context=False,
                    device="cuda",
                    num_epochs=int(cmd_args.epoch))
    
    exp_name = f"{model}-{os.path.basename(exp_dataset_folder)}".replace("/","_")

    train(args=args,
        exp_dataset_folder=exp_dataset_folder,
        experiment_name=exp_name,
        models_folder="./sft/models",
        version=exp_name)
    
    time.sleep(60)

