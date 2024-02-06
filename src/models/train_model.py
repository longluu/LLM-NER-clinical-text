from src.data.data_loader import *
from src.models.model import *
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import f1_score
import argparse
import numpy as np

class ModelTrainer():
    def __init__(self, dataset_name: str, path_umls_semtype: str, model_name: str, outdir: str, \
                learning_rate: float, epoch: int = 3, batch_size: int = 32, weight_decay: float = 0.01):
        self.dataset_name = dataset_name
        self.path_umls_semtype = path_umls_semtype
        self.model_name = model_name
        self.outdir = outdir
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
    def train_model(self):
        # Load and preprocess the data
        dataset_loader = DatasetLoader(dataset_name=self.dataset_name, model_name=self.model_name, \
                                       path_umls_semtype=self.path_umls_semtype)
        global classmap
        dataset, classmap, umls_label_code, _ = dataset_loader.load_dataset()

        # Remove columns not used in training
        if 'MedMentions' in self.dataset_name:
            dataset = dataset.remove_columns(['Full Text', 'Entity Codes', 'tokens', 'ner_tags', 'token_labels'])
            
        elif 'ncbi_disease' in self.dataset_name:
            dataset = dataset.remove_columns(['tokens', 'id', 'token_labels'])
            
        elif 'n2c2-2018' in self.dataset_name:
            dataset = dataset.remove_columns(['tokens', 'ner_tags', 'token_labels'])
            
        # Create an NER model from a base pretrained model
        model_loader = ModelLoader(self.model_name)
        NER_model, tokenizer, config = model_loader.load_model(num_labels = len(classmap.names))
        
        # Create a collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Specify training args
        logging_steps = len(dataset["train"]) // self.batch_size
        training_args = TrainingArguments(
                            output_dir=self.outdir,
                            num_train_epochs=self.epoch,
                            per_device_train_batch_size=self.batch_size,
                            per_device_eval_batch_size=self.batch_size,
                            load_best_model_at_end=True,
                            learning_rate=self.learning_rate,
                            weight_decay=self.weight_decay,
                            logging_steps=logging_steps,
                            log_level='info',
                            evaluation_strategy='epoch',
                            save_strategy='epoch',
                            disable_tqdm=False
                        )

        # Create a trainer
        trainer = Trainer(model=NER_model, 
                          args=training_args, 
                          data_collator=data_collator, 
                          compute_metrics=self.compute_metrics,
                          train_dataset=dataset["train"],
                          eval_dataset=dataset["validation"], 
                          tokenizer=tokenizer)

        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()

    def compute_metrics(self, eval_pred):
        y_pred, y_true = self.align_predictions(eval_pred.predictions, 
                                           eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred)}
    
    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                # Ignore label IDs = -100
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(classmap.int2str(int(label_ids[batch_idx][seq_idx])))
                    example_preds.append(classmap.int2str(int(preds[batch_idx][seq_idx])))

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        help="names of pretrained model, e.g. 'UFNLP/gatortron-base'")
    # resume training on a NER model if set it will overwrite pretrained_model
    parser.add_argument("--resume_from_model", type=str, default=None,
                        help="The NER model file or directory for continuous fine tuning.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data directory.")
    parser.add_argument("--new_model_dir", type=str, required=True,
                        help="directory for saving new model checkpoints")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="The batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--path_umls_semtype", type=str, required=True,
                        help="The directory containing UMLS concept types.")
    
    global_args = parser.parse_args()
    
    # Initiate a trainer instance
    model_trainer = ModelTrainer(dataset_name=global_args.data_dir,
                                 path_umls_semtype=global_args.path_umls_semtype,
                                 model_name=global_args.model_name, 
                                 outdir=global_args.new_model_dir, 
                                 epoch=global_args.num_train_epochs, 
                                 batch_size=global_args.batch_size, 
                                 weight_decay=global_args.weight_decay,
                                 learning_rate=global_args.learning_rate)
    
    # Start training
    model_trainer.train_model()
    
if __name__ == '__main__':
    main()