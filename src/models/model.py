from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
import torch
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from src.data.data_loader import *
from torch.nn.functional import cross_entropy
from typing import List
import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
    
class ModelLoader():
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def load_model(self, num_labels: int):
        # Choose the gpu to load model on (if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('The device to run the model:', device)
        
        # Load the model
        print('Load the pretrained model ...')
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        NER_model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels)
        print('The model has ' + str(NER_model.num_parameters()/1e6) + 'millions parameters.')
        
        return NER_model, tokenizer, config
    
    def evaluate_model(self, dataset_name: str, path_umls_semtype: str, metric_names: List[str]):
        # Load the fine-tuned model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        global NER_model
        NER_model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(device)

        # Load the data
        dataset_loader = DatasetLoader(dataset_name=dataset_name, path_umls_semtype=path_umls_semtype, model_name=self.model_name)
        dataset, classmap, umls_label_code, tokenizer = dataset_loader.load_dataset()
        
        if 'MedMentions' in dataset_name:
            dataset = dataset.remove_columns(['Full Text', 'Entity Codes', 'tokens', 'ner_tags', 'token_labels'])
        
        elif 'ncbi_disease' in dataset_name:
            dataset = dataset.remove_columns(['tokens', 'id', 'token_labels'])
            
        elif 'n2c2-2018' in dataset_name:
            dataset = dataset.remove_columns(['tokens', 'ner_tags', 'token_labels'])
            
        # Create a collator
        global data_collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Make predictions on test data
        prediction_results = dataset['test'].map(self.forward_pass_with_label, batched=True, batch_size=1)        
        
        # Compute the metrics
        pred_list = []
        label_list = []
        for predictions, labels in zip(prediction_results['predicted_label'], prediction_results['labels']):
            example_labels, example_preds = [], []
            for pred, label in zip(predictions, labels):
                if label != -100:
                    example_preds.append(classmap.int2str(int(pred)))
                    example_labels.append(classmap.int2str(int(label)))

            label_list.append(example_labels)
            pred_list.append(example_preds)
            
        return {'f1': f1_score(pred_list, label_list), \
                'precision': precision_score(pred_list, label_list), \
                'recall': recall_score(pred_list, label_list)}, \
                prediction_results, classmap
        
        
    def forward_pass_with_label(self, batch):
        # Convert dict of lists to list of dicts suitable for data collator
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]

        # Pad inputs and labels and put all tensors on device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = data_collator(features)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            # Pass data through model  
            output = NER_model(input_ids, attention_mask)

            # Logit.size: [batch_size, sequence_length, classes]
            # Predict class with largest logit value on classes axis
            predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()

        return {"predicted_label": predicted_label}