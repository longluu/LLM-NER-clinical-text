from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
import torch
from datasets import load_metric
from src.data.data_loader import *
from torch.nn.functional import cross_entropy
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
    
    def evaluate_model(self, model_name: str, dataset_name: str, path_umls_semtype: str):
        # Load the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        NER_model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Load the data
        dataset_loader = DatasetLoader(dataset_name=dataset_name, path_umls_semtype=path_umls_semtype, model_name=model_name)
        data_medmentions, classmap, umls_label_code, tokenizer = dataset_loader.load_dataset()
        data_medmentions = data_medmentions.remove_columns(['Full Text', 'Entity Codes', 'tokens', 'ner_tags', 'token_labels'])

        # Create a collator
        global data_collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Make predictions on test data
        predicted_labels = data_medmentions['test'].map(forward_pass_with_label, batched=True, batch_size=6)        
        
    def forward_pass_with_label(self, batch):
        # Convert dict of lists to list of dicts suitable for data collator
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]

        # Pad inputs and labels and put all tensors on device
        batch = data_collator(features)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            # Pass data through model  
            output = NER_model(input_ids, attention_mask)

            # Logit.size: [batch_size, sequence_length, classes]
            # Predict class with largest logit value on classes axis
            predicted_label = torch.argmax(output.logits, axis=-1).cpu().numpy()

        return {"predicted_label": predicted_label}