import os
import torch
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
##
from model import HybridTextCNN
from utils import *
from parameters import  *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 모델 초기화
try:
    tokenizer = BertTokenizerFast.from_pretrained(BERT)
except:
    tokenizer = BertTokenizerFast.from_pretrained("../05-1matbert/" + BERT)
vocab_size = tokenizer.vocab_size
model = HybridTextCNN(6, 6, vocab_size, char_vocab_size, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, KERNEL_SIZES, NUM_FILTERS)
model.to(device)
# 저장된 모델 로드
model.load_state_dict(torch.load(os.path.join("./model_weights", "hybrid_text_cnn_matbertlr0.001_epoch20_matbert.pth")))
model.eval()
print("Model loaded!")

# 예측 예제
text = "A new study on the effects of SiO2 nanostar in solar cells"
text ='''Driving into the Future: Nano Graphene and Silicon Dioxide Enriched Kevlar Composites for Automotive Applications'''

texts = [
"Effect of nano-metal oxides (ZnO, Al_2O_3, CuO, and TiO_2) on the corrosion behavior of a nano-metal oxide/epoxy coating applied on the copper substrate in the acidic environment"
]
for text in texts:
  is_nanoparticle_result, main_subject_result, out1,out2 = predict(text, model, tokenizer, MAX_LEN, device,  threshold=0.4)
  nanoparticlePredicted = to_label_text(is_nanoparticle_result.tolist()[0])
  MainSubjectPredicted = to_label_text(main_subject_result.tolist()[0])
  print("Is Nanoparticle:", nanoparticlePredicted,out1)
  print("Main Subject:", MainSubjectPredicted,out2)