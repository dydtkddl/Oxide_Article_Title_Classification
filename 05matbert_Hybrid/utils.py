# Clean text function
import torch
from sklearn.model_selection import train_test_split
import string
import re

# Define character set and mapping
char_set = string.ascii_letters + string.digits + string.punctuation + " "
char_vocab_size = len(char_set)
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for idx, char in enumerate(char_set)}
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("–", "-")
    text = re.sub(r'[^a-zA-Z0-9\s\-\_\(\)\[\]\/@]', '', text)
    return text


def predict(text, model, tokenizer, max_len, device, threshold=0.5):
    model.eval()
    
    text = clean_text(text)
    
    word_encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    char_encoding = char_tokenize(text, max_len)
    
    word_input_ids = word_encoding['input_ids'].to(device)
    char_input_ids = char_encoding['input_ids'].to(device)

    with torch.no_grad():
        outputs_is_nanoparticle, outputs_main_subject = model(
            word_input_ids,
            char_input_ids
        )
    
    outputs_is_nanoparticle = torch.sigmoid(outputs_is_nanoparticle).cpu().numpy()
    outputs_main_subject = torch.sigmoid(outputs_main_subject).cpu().numpy()
    
    is_nanoparticle_result = (outputs_is_nanoparticle > threshold).astype(int)
    main_subject_result = (outputs_main_subject > threshold).astype(int)
    
    return is_nanoparticle_result, main_subject_result,outputs_is_nanoparticle,outputs_main_subject

# Character tokenization function
def char_tokenize(text, max_len):
    char_tokens = list(text)
    char_ids = [char_to_index.get(c, char_to_index[' ']) for c in char_tokens]

    if len(char_ids) < max_len:
        padding_length = max_len - len(char_ids)
        char_ids += [char_to_index[' ']] * padding_length
    else:
        char_ids = char_ids[:max_len]

    return {
        'input_ids': torch.tensor(char_ids).unsqueeze(0)
    }
def to_label_text(predict_logit):
    names = [
        'ZnO',
        'SiO2',
        'TiO2',
        'Al2O3',
        'CuO',
        'Other'
    ]
    predict_names = []
    for i, logit in enumerate(predict_logit):
        if logit == 1:
            name = names[i]
            predict_names.append(name)
    return predict_names


   