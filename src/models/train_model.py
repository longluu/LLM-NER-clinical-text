from src.data.data_loader import *
from src.models.model import *
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric
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
        dataset, classmap, umls_label_code, _ = dataset_loader.load_dataset()

        # Remove columns not used in training
        if self.dataset_name == 'ibm/MedMentions-ZS':
            dataset = dataset.remove_columns(['tokens', 'ner_tags', 'token_labels'])
            
        # Create an NER model from a base pretrained model
        model_loader = ModelLoader(self.model_name)
        NER_model, tokenizer, config = model_loader.load_model(num_labels = len(classmap.names))
        
        # Create a collator
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        # Define validation metrics
        global metric
        metric = load_metric("f1")
        
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


    def compute_metrics(self, eval_pred):  # custom method to take in logits and calculate accuracy of the eval set
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    
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