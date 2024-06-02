# Define model
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
class HybridTextCNN(nn.Module):
    def __init__(self, nanoparticle_classes, subject_classes, vocab_size, char_vocab_size, embedding_dim, char_embedding_dim, kernel_sizes, num_filters):
        super(HybridTextCNN, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, char_embedding_dim)) for k in kernel_sizes
        ])
        self.fc_is_nanoparticle = nn.Linear(len(kernel_sizes) * num_filters * 2, nanoparticle_classes)
        self.fc_main_subject = nn.Linear(len(kernel_sizes) * num_filters * 2, subject_classes)

    def conv_and_pool(self, x, convs):
        x = x.unsqueeze(1)
        conv_results = [F.relu(conv(x)).squeeze(3) for conv in convs]
        pool_results = [F.max_pool1d(result, result.size(2)).squeeze(2) for result in conv_results]
        return torch.cat(pool_results, 1)

    def forward(self, word_input_ids, char_input_ids):
        word_embedded = self.word_embedding(word_input_ids)
        word_features = self.conv_and_pool(word_embedded, self.word_convs)
        char_embedded = self.char_embedding(char_input_ids)
        char_features = self.conv_and_pool(char_embedded, self.char_convs)
        features = torch.cat([word_features, char_features], 1)
        is_nanoparticle_output = self.fc_is_nanoparticle(features)
        main_subject_output = self.fc_main_subject(features)
        return is_nanoparticle_output, main_subject_output