from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
import torch

    
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
        NER_model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels).to(device)
        print('The model has ' + str(NER_model.num_parameters()/1e6) + 'millions parameters.')
        
        return NER_model, tokenizer, config