import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import string
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from CustomDataset import CustomDataset  # Import your CustomDataset class
from model import HybridTextCNN  # Import your HybridTextCNN model class
from utils import *  # Import your utility functions
from EarlyStopping import EarlyStopping

# Define character set and mapping
char_set = string.ascii_letters + string.digits + string.punctuation + " "
char_vocab_size = len(char_set)
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for idx, char in enumerate(char_set)}

def main(args):
    print("HY")

    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    EMBEDDING_DIM = args.embedding_dim
    CHAR_EMBEDDING_DIM = args.char_embedding_dim
    KERNEL_SIZES = args.kernel_sizes
    NUM_FILTERS = args.num_filters
    LR = args.lr
    BERT = args.bert
    DATAPATH = args.data_path
    THRESHOLD= args.threshold
    print(
        "MAX_LEN : ", MAX_LEN, "\n",
        "BATCH_SIZE : ", BATCH_SIZE, "\n",
        "EPOCHS : ", EPOCHS, "\n",
        "EMBEDDING_DIM : ", EMBEDDING_DIM, "\n",
        "CHAR_EMBEDDING_DIM : ", CHAR_EMBEDDING_DIM, "\n",
        "KERNEL_SIZES : ", KERNEL_SIZES, "\n",
        "NUM_FILTERS : ", NUM_FILTERS, "\n",
        "LR : ", LR, "\n",
        "DATAPATH : ", DATAPATH, "\n",
        "BERT : ", BERT,
        "THRESHOLD : ", THRESHOLD,
    )

    # Load data and tokenizer
    try:
        tokenizer = BertTokenizerFast.from_pretrained(BERT)
    except:
        tokenizer = BertTokenizerFast.from_pretrained("../05-1matbert/" + BERT)

    df = pd.read_excel(DATAPATH)
    texts = df['title'].tolist()
    is_nanoparticle_labels = [ast.literal_eval(sub) for sub in df['is_nanoparticle'].tolist()]
    main_subject_labels = [ast.literal_eval(sub) for sub in df['main_subject'].tolist()]

    # Train-validation-test split
    train_texts, temp_texts, train_is_nanoparticle_labels, temp_is_nanoparticle_labels, train_main_subject_labels, temp_main_subject_labels = train_test_split(
        texts, is_nanoparticle_labels, main_subject_labels, test_size=0.2, random_state=42)
    val_texts, test_texts, val_is_nanoparticle_labels, test_is_nanoparticle_labels, val_main_subject_labels, test_main_subject_labels = train_test_split(
        temp_texts, temp_is_nanoparticle_labels, temp_main_subject_labels, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_texts, train_is_nanoparticle_labels, train_main_subject_labels, tokenizer, max_len=MAX_LEN)
    val_dataset = CustomDataset(val_texts, val_is_nanoparticle_labels, val_main_subject_labels, tokenizer, max_len=MAX_LEN)
    test_dataset = CustomDataset(test_texts, test_is_nanoparticle_labels, test_main_subject_labels, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    num_nanoparticle_classes = 6
    num_subject_classes = 6
    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridTextCNN(num_nanoparticle_classes, num_subject_classes, vocab_size, char_vocab_size, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, KERNEL_SIZES, NUM_FILTERS)
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    # Training
    train_losses = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    val_accuracies = []

    log_file = open("./training_log/lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.txt"% (LR, EPOCHS, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS), "w")

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs_is_nanoparticle, outputs_main_subject = model(
                batch['word_input_ids'].to(device),
                batch['char_input_ids'].to(device)
            )
            loss_is_nanoparticle = criterion(outputs_is_nanoparticle, batch['is_nanoparticle_labels'].to(device))
            loss_main_subject = criterion(outputs_main_subject, batch['main_subject_labels'].to(device))
            loss = loss_is_nanoparticle + loss_main_subject
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                outputs_is_nanoparticle, outputs_main_subject = model(
                    batch['word_input_ids'].to(device),
                    batch['char_input_ids'].to(device)
                )
                loss_is_nanoparticle = criterion(outputs_is_nanoparticle, batch['is_nanoparticle_labels'].to(device))
                loss_main_subject = criterion(outputs_main_subject, batch['main_subject_labels'].to(device))
                loss = loss_is_nanoparticle + loss_main_subject
                total_val_loss += loss.item()

                all_val_labels.append(batch['main_subject_labels'].cpu().numpy())
                all_val_preds.append(torch.sigmoid(outputs_main_subject).cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check early stopping
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        all_val_labels = np.concatenate(all_val_labels)
        all_val_preds = (np.concatenate(all_val_preds) >= THRESHOLD).astype(int)

        # Calculate metrics
        precision = precision_score(all_val_labels, all_val_preds, average='macro')
        recall = recall_score(all_val_labels, all_val_preds, average='macro')
        f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        accuracy = accuracy_score(all_val_labels, all_val_preds)

        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        val_accuracies.append(accuracy)

        # Log metrics
        log_file.write(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}\n")

    log_file.close()

    # Create plot directory if not exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot train and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylim(0, 1)
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('plots/loss_plot_lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.png' % (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS))
    plt.close()
    # Plot validation metrics
    plt.figure()
    plt.plot(range(1, len(val_precisions) + 1), val_precisions, label='Precision')
    plt.plot(range(1, len(val_recalls) + 1), val_recalls, label='Recall')
    plt.plot(range(1, len(val_f1s) + 1), val_f1s, label='F1 Score')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.ylim(0, 1)
    plt.title('Validation Metrics Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('plots/validation_metrics_plot_lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.png' % (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS))
    plt.close()

    # Evaluate on test set
    model.eval()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for batch in test_loader:
            outputs_is_nanoparticle, outputs_main_subject = model(
                batch['word_input_ids'].to(device),
                batch['char_input_ids'].to(device)
            )
            all_test_labels.append(batch['main_subject_labels'].cpu().numpy())
            all_test_preds.append(torch.sigmoid(outputs_main_subject).cpu().numpy())

    all_test_labels = np.concatenate(all_test_labels)
    all_test_preds = (np.concatenate(all_test_preds) >= THRESHOLD).astype(int)

    # Calculate test metrics
    precision = precision_score(all_test_labels, all_test_preds, average='macro')
    recall = recall_score(all_test_labels, all_test_preds, average='macro')
    f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    accuracy = accuracy_score(all_test_labels, all_test_preds)

    # Log test metrics
    with open("./test_results/lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.txt"% (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS), "w") as f:
        f.write(f"Test Precision: {precision}\n")
        f.write(f"Test Recall: {recall}\n")
        f.write(f"Test F1 Score: {f1}\n")
        f.write(f"Test Accuracy: {accuracy}\n")

    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1 Score: {f1}")
    print(f"Test Accuracy: {accuracy}")

    # Plot test metrics
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title('Test Metrics')
    plt.grid()
    plt.savefig('plots/test_metrics_plot_lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.png' % (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS))
    plt.close()

    # Save model
    save_path = "./model_weights"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file_name = "./training_log/lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.txt"% (LR, EPOCHS, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS)
    new_log_file_name = "./training_log/lr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.txt"% (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS)
    os.rename(log_file_name, new_log_file_name)
    torch.save(model.state_dict(), os.path.join(save_path, "hybrid_text_cnn_matbertlr%s_epoch%s_%s_threshold%s_MAX_LEN%s_BATCH_SIZE%s_KERNEL_SIZES%s_NUM_FILTERS%s.pth" % (LR, epoch, BERT,THRESHOLD, MAX_LEN, BATCH_SIZE, KERNEL_SIZES, NUM_FILTERS)))
    print("Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hybrid text CNN model.")
    parser.add_argument("--data_path", type=str, default="../04datapurify/purify_data.xlsx", help="Path to the dataset file")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of the input sequences")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of the word embeddings")
    parser.add_argument("--char_embedding_dim", type=int, default=64, help="Dimension of the character embeddings")
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5], help="Kernel sizes for the CNN")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters per kernel size")
    parser.add_argument("--lr", type=float, default =1e-3, help="Learning rate")
    parser.add_argument("--bert", type=str, default="bert-base-uncased", help="BERT name")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold")
    
    
    args = parser.parse_args()
    main(args)
