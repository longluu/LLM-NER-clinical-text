from itertools import islice
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from typing import List
import os
import pandas as pd
import csv


class DatasetLoader():
    def __init__(self, dataset_name: str, model_name: str, path_umls_semtype: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.path_umls_semtype = path_umls_semtype
    
    def load_dataset(self):
        # Load the model tokenizer
        global tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the dataset
        print('Loading and preprocessing the dataset ...')
        print(self.dataset_name)
        if 'MedMentions' in self.dataset_name:
            # Load the data
            dataset = load_dataset("csv", data_files={
                "train": self.dataset_name + 'train.csv',
                "validation": self.dataset_name + 'val.csv',
                "test": self.dataset_name + 'test.csv'})
            
            # Remove unused column
            dataset = dataset.remove_columns('UMLS Code')
            
            # Transform the dataset to a BIO format for LLM
            dataset = dataset.map(self.split_and_convert_to_BIO, batched=True)
            
            # Create a list of unique label code
            st21pv_types = ['T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 
                            'T038', 'T058', 'T062', 'T074', 'T082', 'T091', 'T092', 
                            'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204']
            unique_label_code = sum([['B-' + code, 'I-' + code] for code in st21pv_types], [])
            unique_label_code.append('O')

            # Encode the text labels in the dataset
            classmap = ClassLabel(num_classes=len(unique_label_code), names=unique_label_code)
            dataset = dataset.map(lambda y: {"token_labels": classmap.str2int(y["ner_tags"])})
            
            # Load the UMLS concept types and make it match the label code
            umls_semtype = self.load_umls_semtype(self.path_umls_semtype)
            umls_label_code = {}
            for code in umls_semtype.keys():
                umls_label_code['B-'+code] = umls_semtype[code]
                umls_label_code['I-'+code] = umls_semtype[code]
            umls_label_code['O'] = None
            
        elif 'ncbi_disease' in self.dataset_name:
            # Load the data
            dataset = load_dataset(self.dataset_name)
            
            # Rename the label column
            dataset = dataset.rename_column("ner_tags", "token_labels")
            
            # Set the mapping of labels
            classmap = ClassLabel(num_classes=3, names=['none', 'disease', 'disease_continued'])
            umls_label_code = None
            
        elif 'n2c2-2018' in self.dataset_name:
            # Get lists of all files
            train_dir = self.dataset_name + "train-bio/"
            train_files = [train_dir + file for file in os.listdir(train_dir) if os.path.isfile(train_dir + file)]
            test_dir = self.dataset_name + "test-bio/"
            test_files = [test_dir + file for file in os.listdir(test_dir) if os.path.isfile(test_dir + file)]
            
            # Get the data into pandas dataframe
            df_train, df_test = self.n2c2_txt_to_pandas(train_files, test_files)

            # Load test data
            test_ds = Dataset.from_pandas(df_test, split='test')
            
            # Load train data
            dataset_train = Dataset.from_pandas(df_train)

            # Split to train and validation
            dataset_split = dataset_train.train_test_split(test_size=0.2)
            train_ds = dataset_split.pop("train")
            validation_ds = dataset_split.pop("test")
            dataset = DatasetDict({"train": train_ds,
                                   "validation": validation_ds,
                                    "test": test_ds})
            
            # Set the mapping of labels
            classmap = ClassLabel(num_classes=19, names=['B-Drug', 'I-Drug', 'B-Strength', 'I-Strength', 'B-Duration', 'I-Duration',
                                                          'B-Route', 'I-Route', 'B-Form', 'I-Form', 'B-ADE', 'I-ADE',
                                                          'B-Dosage', 'I-Dosage', 'B-Reason', 'I-Reason', 'B-Frequency', 'I-Frequency', 'O'])
            dataset = dataset.map(lambda y: {"token_labels": classmap.str2int(y["ner_tags"])})
            
            umls_label_code = None                     
        
        # Re-tokenize with the model's tokenizer and align the labels
        dataset = dataset.map(self.encode_and_align_labels, batched=True)

        return dataset, classmap, umls_label_code, tokenizer

    def n2c2_txt_to_pandas(self, train_files: List[str], test_files: List[str]):
        # Iterate over train and test files to extract tokens and NER tags
        df_train = pd.DataFrame(columns=['tokens', 'ner_tags'])
        chunk_size = 450

        for file in train_files:
            # Read the BIO file
            df = pd.read_csv(file, sep=" ", header=None, quoting=csv.QUOTE_NONE)
            df = df.dropna()

            # Get tokens and ner_tags in lists
            tokens = df.iloc[:, 0].to_list()
            ner_tags = df.iloc[:, -1].to_list()

            # Split the tokens and ner_tags to input size of the BERT model
            tokens_list = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]  
            ner_tags_list = [ner_tags[i:i + chunk_size] for i in range(0, len(ner_tags), chunk_size)] 

            for (tokens, ner_tags) in zip(tokens_list, ner_tags_list):
                # Make a pandas dataframe containing chunks of tokens and ner tags
                df_new = pd.DataFrame(columns = ['tokens', 'ner_tags']) 
                df_new.at[0, 'tokens'] = tokens
                df_new.at[0, 'ner_tags'] = ner_tags

                # Concatenate with the train dataframe
                df_train = pd.concat([df_train, df_new], ignore_index=True)   

        df_test = pd.DataFrame(columns=['tokens', 'ner_tags'])
        for file in test_files:
            # Read the BIO file
            df = pd.read_csv(file, sep=" ", header=None, quoting=csv.QUOTE_NONE)
            df = df.dropna()

            # Get tokens and ner_tags in lists
            tokens = df.iloc[:, 0].to_list()
            ner_tags = df.iloc[:, -1].to_list()

            # Split the tokens and ner_tags to input size of the BERT model
            tokens_list = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]  
            ner_tags_list = [ner_tags[i:i + chunk_size] for i in range(0, len(ner_tags), chunk_size)] 

            for (tokens, ner_tags) in zip(tokens_list, ner_tags_list):
                # Make a pandas dataframe containing chunks of tokens and ner tags
                df_new = pd.DataFrame(columns = ['tokens', 'ner_tags']) 
                df_new.at[0, 'tokens'] = tokens
                df_new.at[0, 'ner_tags'] = ner_tags

                # Concatenate with the test dataframe
                df_test = pd.concat([df_test, df_new], ignore_index=True)     
                
        return df_train, df_test
    
    def split_and_convert_to_BIO(self, example):
        input_processed = {}

        # First convert the Full Text to list of tokens
        tokens = []
        for text in example['Full Text']:
            # Convert string of words to list of words
            text = text.replace("'", "").strip('][').split(', ')

            # Remove all <START> and <END>
            text = [item for item in text if (item != '<START>') and (item != '<END>')]

            tokens.append(text)

        # Convert Entity Codes to ner_tags
        ner_tags = []
        for text in example['Entity Codes']:
            # Convert the labels to BIO format
            text = text.replace('U-', 'B-').replace('L-', 'I-')

            # Convert string of words to list of words
            text = text.replace("'", "").strip('][').split(', ')

            # Remove all <START> and <END>
            text = [item for item in text if (item != '<START>') and (item != '<END>')]
            
            ner_tags.append(text)

        input_processed["tokens"] = tokens
        input_processed["ner_tags"] = ner_tags

        return input_processed
    
    def encode_and_align_labels(self, example):
        tokenized_inputs = tokenizer(example["tokens"], max_length=512, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(example[f"token_labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # Set the special tokens to -100.
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # CLS and SEP are labeled as -100

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        
        return tokenized_inputs
        
    def read_in_chunks(self, file_object, n=10000):
        """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k."""
        while True:
            data = list(islice(file_object, n))
            if not data:
                break
                
            yield data


    def load_umls_concepts(self, file_path: str):
        umls_concepts = {}
        with open(file_path) as f:
            for piece in self.read_in_chunks(f):
                for line in piece:
                    if line != "":
                        line_list = line.split("|")
                        if line_list[1] == 'ENG':
                            umls_concepts[line_list[0]] = line_list[14]
                            
        return umls_concepts


    def load_umls_semtype(self, file_path: str):
        umls_semtype = {}
        with open(file_path) as f:
            f = f.read().split("\n")
            for line in f:
                if line != "":
                    line_list = line.split("|")
                    umls_semtype[line_list[2]] = [line_list[1], line_list[3]]
                    
        return umls_semtype