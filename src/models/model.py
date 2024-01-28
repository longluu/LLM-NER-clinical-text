from transformers import AutoModelForTokenClassification, AutoConfig
from transformers.models.megatron_bert.configuration_megatron_bert import MegatronBertConfig
from transformers import BertTokenizer
    
    
class ModelLoader():
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def load_model(self, num_labels: int):
        # Load the base model
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = num_labels
        NER_model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=num_labels)           
        
        return NER_model, tokenizer, config