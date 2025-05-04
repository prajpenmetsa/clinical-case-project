import opendatasets as od
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import argparse
import os
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import KeyedVectors
import re
from transformers import BertModel, BertTokenizer

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model configuration
EMBED_SIZE = 300  # GloVe embedding size
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 256
BIDIRECTIONAL = True

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch_size, hidden_size)
        
        return context_vector, attention_weights

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers, 
                 bidirectional=True, dropout=0.3, embedding_matrix=None):
        super(LSTMAttentionClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embedding.weight.requires_grad = True
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = Attention(hidden_size * 2 if bidirectional else hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len)
        
        # Embedding layer
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # Make sure all lengths are > 0 to avoid the pack_padded_sequence error
        valid_lengths = torch.clamp(lengths, min=1)
        
        # Pack padded sequences for more efficient LSTM processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM layer
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # Apply attention
        context_vector, attention_weights = self.attention(output)
        # context_vector shape: (batch_size, hidden_size * num_directions)
        
        # Fully connected layers with dropout
        output = self.dropout(F.relu(self.fc1(context_vector)))
        output = self.fc2(output)
        
        return output, attention_weights

def download_dataset():
    """Download the dataset from Kaggle if it doesn't exist"""
    if not os.path.exists('./kuc-hackathon-winter-2018'):
        print("Downloading dataset from Kaggle...")
        od.download('https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018')
    else:
        print("Dataset already downloaded.")


def load_biobert_embeddings():
    """Initialize BioBERT model and tokenizer"""
    print("Loading BioBERT model...")
    try:
        tokenizer = BertTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
        model = BertModel.from_pretrained('monologg/biobert_v1.1_pubmed')
        model.eval()  # Set to evaluation mode
        if torch.cuda.is_available():
            model = model.cuda()
        print("BioBERT model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading BioBERT model: {e}")
        return None, None

def get_biobert_embedding(word, tokenizer, model):
    """Get BioBERT embedding for a single word"""
    try:
        # Make sure the word is properly tokenized for BERT
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the embedding from the last hidden state
        # Take the average of all token embeddings (this is a simple approach)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Move to CPU and convert to numpy
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error extracting BioBERT embedding for '{word}': {e}")
        return None

def preprocess_text(text):
    """Preprocess the text: lowercase, remove special chars, tokenize"""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    return tokens

def load_data(limit=None):
    """Load and prepare the data"""
    try:
        kaggle_train = pd.read_csv('./kuc-hackathon-winter-2018/drugsComTrain_raw.csv')
        kaggle_test = pd.read_csv('./kuc-hackathon-winter-2018/drugsComTest_raw.csv')
        
        # Clean data - remove rows with empty reviews or conditions
        kaggle_train = kaggle_train.dropna(subset=['review', 'condition'])
        kaggle_test = kaggle_test.dropna(subset=['review', 'condition'])
        
        main_x = list(pd.concat([kaggle_train['review'], kaggle_test['review']], axis=0, ignore_index=True))
        main_y = list(pd.concat([kaggle_train['condition'], kaggle_test['condition']], axis=0))
        
        if limit:
            main_x = main_x[:limit]
            main_y = main_y[:limit]
        
        # Create mapping dictionaries for conditions
        vocab = list(Counter(main_y).keys())
        i_to_x = {i: vocab[i] for i in range(len(vocab))}
        x_to_i = {vocab[i]: i for i in range(len(vocab))}
        
        return main_x, main_y, vocab, i_to_x, x_to_i
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def build_vocabulary(texts, min_freq=2):
    """Build vocabulary from texts with minimum frequency"""
    word_counts = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        word_counts.update(tokens)
    
    # Filter words by frequency
    words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create word-to-index mapping
    word_to_idx = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(words):
        word_to_idx[word] = i + 2
    
    return word_to_idx

def load_embeddings(word_to_idx, embed_size=300, use_biobert=True):
    """Load embeddings (BioBERT only)"""
    print(f"Loading BioBERT embeddings (dimension: {embed_size})...")
    
    # Initialize BioBERT
    biobert_tokenizer, biobert_model = load_biobert_embeddings()
    
    if biobert_tokenizer is None or biobert_model is None:
        print("Error: Could not load BioBERT model. Please check your internet connection and try again.")
        return np.random.uniform(-0.25, 0.25, (len(word_to_idx), embed_size)), {'biobert': 0, 'random': len(word_to_idx)}
    
    embedding_matrix = np.zeros((len(word_to_idx), embed_size))
    biobert_found = 0
    random_initialized = 0
    
    # Initialize embedding for <pad> and <unk> tokens
    embedding_matrix[0] = np.zeros(embed_size)  # <pad>
    embedding_matrix[1] = np.random.uniform(-0.25, 0.25, embed_size)  # <unk>
    random_initialized += 1  # Counting <unk> as randomly initialized
    
    # Track which words we found in BioBERT
    biobert_words = set()
    
    # Process all words with BioBERT
    print("Getting embeddings from BioBERT...")
    for word, idx in tqdm(word_to_idx.items(), desc="Processing BioBERT embeddings"):
        if word in ['<pad>', '<unk>']:
            continue
                
        try:
            embedding = get_biobert_embedding(word, biobert_tokenizer, biobert_model)
            if embedding is not None:
                # Truncate or pad BioBERT embedding to match desired size
                if len(embedding) > embed_size:
                    embedding = embedding[:embed_size]
                elif len(embedding) < embed_size:
                    embedding = np.pad(embedding, (0, embed_size - len(embedding)))
                
                embedding_matrix[idx] = embedding
                biobert_found += 1
                biobert_words.add(word)
        except Exception as e:
            print(f"Error getting BioBERT embedding for '{word}': {e}")
            # Use random initialization for this word
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embed_size)
            random_initialized += 1
    
    # For any remaining words, use random initialization
    for word, idx in word_to_idx.items():
        if np.all(embedding_matrix[idx] == 0) and idx not in [0]:  # Skip <pad>
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embed_size)
            random_initialized += 1
    
    # Print coverage statistics
    total_words = len(word_to_idx)
    print("\n===== EMBEDDING COVERAGE STATISTICS =====")
    print(f"Total vocabulary size: {total_words}")
    print(f"BioBERT coverage: {biobert_found} words ({biobert_found/total_words*100:.2f}%)")
    print(f"Random initialization: {random_initialized} words ({random_initialized/total_words*100:.2f}%)")
    print("========================================\n")
    
    return embedding_matrix, {'biobert': biobert_found, 'random': random_initialized}

def tokenize_text(texts, word_to_idx, max_length):
    """Convert texts to token indices and calculate lengths"""
    token_ids = []
    lengths = []
    
    for text in texts:
        # Make sure text is not empty
        if not text or len(text.strip()) == 0:
            # For empty texts, use a single <unk> token
            indices = [word_to_idx['<unk>']]
            length = 1
        else:
            tokens = preprocess_text(text)
            # If no tokens (e.g., after preprocessing), use a single <unk>
            if not tokens:
                tokens = ['<unk>']
            
            # Convert tokens to indices, use <unk> for unknown words
            indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
            
            # Truncate or pad to max_length
            if len(indices) > max_length:
                indices = indices[:max_length]
                length = max_length
            else:
                length = len(indices)
                indices = indices + [word_to_idx['<pad>']] * (max_length - len(indices))
        
        token_ids.append(indices)
        lengths.append(length)
    
    return token_ids, lengths

def prepare_datasets(main_x, main_y, x_to_i, word_to_idx):
    """Split the data and prepare datasets for training"""
    x_train, x_test, y_train, y_test = train_test_split(main_x, main_y, test_size=.21, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)
    
    # Tokenize texts
    x_train_tokens, x_train_lengths = tokenize_text(x_train, word_to_idx, MAX_SEQ_LENGTH)
    x_val_tokens, x_val_lengths = tokenize_text(x_val, word_to_idx, MAX_SEQ_LENGTH)
    x_test_tokens, x_test_lengths = tokenize_text(x_test, word_to_idx, MAX_SEQ_LENGTH)
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train_tokens, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val_tokens, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_tokens, dtype=torch.long)
    
    x_train_lengths = torch.tensor(x_train_lengths, dtype=torch.long)
    x_val_lengths = torch.tensor(x_val_lengths, dtype=torch.long)
    x_test_lengths = torch.tensor(x_test_lengths, dtype=torch.long)
    
    y_train_tensor = torch.tensor([x_to_i[j] for j in y_train], dtype=torch.long)
    y_val_tensor = torch.tensor([x_to_i[j] for j in y_val], dtype=torch.long)
    y_test_tensor = torch.tensor([x_to_i[j] for j in y_test], dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, x_train_lengths, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, x_val_lengths, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, x_test_lengths, y_test_tensor)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader, x_test, y_test

def train_model(model, train_dataloader, val_dataloader, model_metadata, save_path='./best_lstm_model.pt'):
    """Train the model and save the best checkpoint"""
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    best_model = {'accuracy': -1, 'epoch': -1, 'model': {}, 'optimizer': {}}
    
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}')
        
        # Training
        model.train()
        losses = []
        accuracies = []
        f1_scores = []
        
        for inputs, lengths, labels in tqdm(train_dataloader, desc="Training"):
            try:
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                # Forward pass
                out, _ = model(inputs, lengths)
                loss = loss_func(out, labels)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                pred = torch.max(out, dim=1)[1]
                acc = (pred == labels).float().mean()
                accuracies.append(acc.item())
                
                f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                f1_scores.append(f1)
                
                losses.append(loss.item())
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        print(f'Train Loss: {sum(losses)/len(losses):.4f}')
        print(f'Train Accuracy: {sum(accuracies)/len(accuracies):.4f}')
        print(f'Train F1 score: {sum(f1_scores)/len(f1_scores):.4f}')
        
        # Validation
        model.eval()
        val_accuracies = []
        val_losses = []
        val_f1 = []
        
        with torch.no_grad():
            for inputs, lengths, labels in tqdm(val_dataloader, desc="Validation"):
                try:
                    inputs = inputs.to(device)
                    lengths = lengths.to(device)
                    labels = labels.to(device)
                    
                    pred, _ = model(inputs, lengths)
                    loss = loss_func(pred, labels)
                    
                    pred_class = torch.max(pred, dim=1)[1]
                    acc = (pred_class == labels).float().mean()
                    val_accuracies.append(acc.item())
                    
                    f1 = f1_score(labels.cpu().numpy(), pred_class.cpu().numpy(), average='weighted')
                    val_f1.append(f1)
                    
                    val_losses.append(loss.item())
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if len(val_losses) > 0:
            val_loss = sum(val_losses)/len(val_losses)
            val_acc = sum(val_accuracies)/len(val_accuracies)
            print(f'Dev Loss: {val_loss:.4f}')
            print(f'Dev Accuracy: {val_acc:.4f}')
            print(f'Dev F1 score: {sum(val_f1)/len(val_f1):.4f}')
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Save the best model
            if best_model['accuracy'] < val_acc:
                best_model['accuracy'] = val_acc
                best_model['epoch'] = epoch+1
                best_model['model'] = model.state_dict()
                best_model['optimizer'] = optimizer.state_dict()
        else:
            print("No valid validation results for this epoch")
    
    # Save the best model with additional metadata
    if best_model['epoch'] > -1:
        save_dict = {
            'accuracy': best_model['accuracy'],
            'epoch': best_model['epoch'],
            'model': best_model['model'],
            'optimizer': best_model['optimizer'],
        }
        
        # Add all metadata
        save_dict.update(model_metadata)
        
        torch.save(save_dict, save_path)
        
        print(f"Best model saved at epoch {best_model['epoch']} with accuracy {best_model['accuracy']:.4f}")
    else:
        print("No model was saved due to validation errors")
    
    return best_model
def evaluate_model(model, test_dataloader):
    """Evaluate the model on the test set"""
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    test_accuracies = []
    test_f1_scores = []
    test_losses = []
    all_attentions = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, lengths, labels in tqdm(test_dataloader, desc="Testing"):
            try:
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                pred, attention_weights = model(inputs, lengths)
                loss = loss_func(pred, labels)
                
                pred_class = torch.max(pred, dim=1)[1]
                acc = (pred_class == labels).float().mean()
                test_accuracies.append(acc.item())
                
                f1 = f1_score(labels.cpu().numpy(), pred_class.cpu().numpy(), average='weighted')
                test_f1_scores.append(f1)
                
                test_losses.append(loss.item())
                
                # Save attention weights for analysis
                all_attentions.append(attention_weights.cpu().numpy())
                all_preds.extend(pred_class.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error in test batch: {e}")
                continue
    
    if len(test_losses) > 0:
        print(f'Test Loss: {sum(test_losses)/len(test_losses):.4f}')
        print(f'Test Accuracy: {sum(test_accuracies)/len(test_accuracies):.4f}')
        print(f'Test F1 score: {sum(test_f1_scores)/len(test_f1_scores):.4f}')
        
        return {
            'loss': sum(test_losses)/len(test_losses),
            'accuracy': sum(test_accuracies)/len(test_accuracies),
            'f1': sum(test_f1_scores)/len(test_f1_scores),
            'attention_weights': all_attentions,
            'predictions': all_preds,
            'labels': all_labels
        }
    else:
        print("No valid test results")
        return {
            'loss': None,
            'accuracy': None,
            'f1': None,
            'attention_weights': [],
            'predictions': [],
            'labels': []
        }

def predict_condition(model, word_to_idx, symptoms_text, i_to_x):
    """Predict the medical condition based on symptoms"""
    model.eval()
    
    # Tokenize the symptoms text
    tokens, lengths = tokenize_text([symptoms_text], word_to_idx, MAX_SEQ_LENGTH)
    
    inputs = torch.tensor(tokens, dtype=torch.long).to(device)
    lengths = torch.tensor(lengths, dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs, attention_weights = model(inputs, lengths)
        predicted_class = torch.max(outputs, dim=1)[1].item()
    
    # Get the condition name
    predicted_condition = i_to_x[predicted_class]
    
    # Get confidence scores (softmax probabilities)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    top_5_probabilities, top_5_indices = torch.topk(probabilities, 5)
    
    top_conditions = [
        (i_to_x[idx.item()], prob.item()) 
        for idx, prob in zip(top_5_indices, top_5_probabilities)
    ]
    
    # Return attention weights for visualization
    return predicted_condition, top_conditions, attention_weights

def train_eval_predict(args):
    """Combined function to train, evaluate, and predict in one go"""
    print("-" * 50)
    print("Starting combined train-evaluate-predict workflow with BioBERT embeddings")
    print("-" * 50)
    
    try:
        # Download dataset if needed
        download_dataset()
        
        # Force BioBERT only settings
        args.use_biobert = True
        args.use_glove = False
        
        # Load data
        main_x, main_y, vocab, i_to_x, x_to_i = load_data(limit=args.limit)
        
        # Build vocabulary from texts
        word_to_idx = build_vocabulary(main_x)
        print(f"Vocabulary size: {len(word_to_idx)}")
        
        # Load BioBERT embeddings
        embedding_matrix, embedding_stats = load_embeddings(
            word_to_idx, 
            EMBED_SIZE, 
            use_biobert=True
        )
        
        # Print embedding coverage statistics
        print("\nEmbedding coverage summary:")
        print(f"- BioBERT: {embedding_stats['biobert']} words")
        print(f"- Random: {embedding_stats['random']} words")
        
        # Prepare datasets
        train_dataloader, val_dataloader, test_dataloader, x_test_raw, y_test_raw = prepare_datasets(
            main_x, main_y, x_to_i, word_to_idx
        )
        
        # Initialize model
        model = LSTMAttentionClassifier(
            vocab_size=len(word_to_idx),
            embedding_dim=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_classes=len(vocab),
            num_layers=NUM_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT,
            embedding_matrix=embedding_matrix
        ).to(device)
        
        # Check if we should train from scratch or load existing model
        if args.train or not os.path.exists(args.model_path):
            print("\n" + "="*50)
            print("TRAINING PHASE")
            print("="*50)
            
            # Save embedding statistics in the model
            model_metadata = {
                'word_to_idx': word_to_idx,
                'embedding_stats': embedding_stats,
                'use_glove': False,
                'use_biobert': True
            }
            
            # Train model
            best_model = train_model(
                model, 
                train_dataloader, 
                val_dataloader, 
                model_metadata,
                save_path=args.model_path
            )
            
            # Load best model for evaluation if training was successful
            if os.path.exists(args.model_path):
                checkpoint = torch.load(args.model_path)
                model.load_state_dict(checkpoint['model'])
        else:
            print(f"Loading existing model from {args.model_path}")
            checkpoint = torch.load(args.model_path)
            word_to_idx = checkpoint.get('word_to_idx', word_to_idx)
            model.load_state_dict(checkpoint['model'])
            
            # Print embedding statistics if available
            if 'embedding_stats' in checkpoint:
                print("\nEmbedding coverage from loaded model:")
                print(f"- BioBERT: {checkpoint['embedding_stats'].get('biobert', 0)} words")
                print(f"- Random: {checkpoint['embedding_stats'].get('random', 0)} words")
        
        # Rest of the function remains the same - evaluation and prediction phases
        # Evaluation phase
        print("\n" + "="*50)
        print("EVALUATION PHASE")
        print("="*50)
        
        results = evaluate_model(model, test_dataloader)
        if results['accuracy'] is not None:
            print(f"Model test accuracy: {results['accuracy']:.4f}")
            print(f"Model test F1 score: {results['f1']:.4f}")
        
        # Prediction phase - either use provided symptoms or select random test examples
        print("\n" + "="*50)
        print("PREDICTION PHASE")
        print("="*50)
        
        if args.predict:
            # Use provided symptoms text
            symptoms_text = args.predict
            print(f"Predicting condition for: '{symptoms_text}'")
            
            predicted_condition, top_conditions, _ = predict_condition(
                model, word_to_idx, symptoms_text, i_to_x
            )
            
            print(f"\nBased on the symptoms provided, the predicted condition is: {predicted_condition}")
            print("\nTop 5 possible conditions with confidence scores:")
            for condition, score in top_conditions:
                print(f"- {condition}: {score:.4f} ({score*100:.2f}%)")
        else:
            # Select 5 random test examples for prediction demonstration
            print("Selecting 5 random test examples for prediction demonstration:")
            import random
            random_indices = random.sample(range(len(x_test_raw)), min(5, len(x_test_raw)))
            
            for idx in random_indices:
                symptoms_text = x_test_raw[idx]
                actual_condition = y_test_raw[idx]
                
                # Truncate text for display if too long
                display_text = symptoms_text[:150] + "..." if len(symptoms_text) > 150 else symptoms_text
                print(f"\nReview: '{display_text}'")
                print(f"Actual condition: {actual_condition}")
                
                predicted_condition, top_conditions, _ = predict_condition(
                    model, word_to_idx, symptoms_text, i_to_x
                )
                
                print(f"Predicted condition: {predicted_condition}")
                print("Top 3 predictions:")
                for i, (condition, score) in enumerate(top_conditions[:3]):
                    print(f"  {i+1}. {condition}: {score:.4f} ({score*100:.2f}%)")
                
                print("-" * 30)
    except Exception as e:
        print(f"Error in train_eval_predict: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Medical Condition Classification with LSTM+Attention')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict condition from symptoms')
    parser.add_argument('--model_path', type=str, default='./best_lstm_model.pt', help='Path to save/load model')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    parser.add_argument('--combined', action='store_true', help='Run train, eval, and predict in one go')
    parser.add_argument('--print_embedding_stats', action='store_true', help='Print detailed embedding statistics')
    
    args = parser.parse_args()
    # Force use_biobert to True and use_glove to False
    args.use_biobert = True
    args.use_glove = False
    
    if args.combined:
        train_eval_predict(args)
    else:
        # Original workflow with separate steps
        if args.train:
            # Download dataset if needed
            download_dataset()
            
            # Load data
            main_x, main_y, vocab, i_to_x, x_to_i = load_data(limit=args.limit)
            
            # Build vocabulary from texts
            word_to_idx = build_vocabulary(main_x)
            print(f"Vocabulary size: {len(word_to_idx)}")
            
            # Load BioBERT embeddings
            embedding_matrix, embedding_stats = load_embeddings(
                word_to_idx, 
                EMBED_SIZE, 
                use_biobert=True
            )
            
            # Create model metadata
            model_metadata = {
                'word_to_idx': word_to_idx,
                'embedding_stats': embedding_stats,
                'use_glove': False, 
                'use_biobert': True
            }
            
            # Initialize model
            model = LSTMAttentionClassifier(
                vocab_size=len(word_to_idx),
                embedding_dim=EMBED_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_classes=len(vocab),
                num_layers=NUM_LAYERS,
                bidirectional=BIDIRECTIONAL,
                dropout=DROPOUT,
                embedding_matrix=embedding_matrix
            ).to(device)
            
            # Prepare datasets
            train_dataloader, val_dataloader, test_dataloader, _, _ = prepare_datasets(
                main_x, main_y, x_to_i, word_to_idx
            )
            
            # Train model
            best_model = train_model(
                model, 
                train_dataloader, 
                val_dataloader, 
                model_metadata,
                save_path=args.model_path
            )
        if args.eval:
            # Load data if not already loaded
            if 'main_x' not in locals():
                main_x, main_y, vocab, i_to_x, x_to_i = load_data(limit=args.limit)
                word_to_idx = build_vocabulary(main_x)
            
            # Load model
            try:
                checkpoint = torch.load(args.model_path)
                
                # Check if word_to_idx is in checkpoint
                if 'word_to_idx' in checkpoint:
                    word_to_idx = checkpoint['word_to_idx']
                
                # Print embedding statistics if requested
                if args.print_embedding_stats and 'embedding_stats' in checkpoint:
                    print("\nEmbedding Statistics:")
                    stats = checkpoint['embedding_stats']
                    print(f"- GloVe coverage: {stats.get('glove', 0)} words")
                    print(f"- BioBERT coverage: {stats.get('biobert', 0)} words")
                    print(f"- Random initialization: {stats.get('random', 0)} words")
                    print(f"- Total vocabulary size: {len(word_to_idx)}")
                
                # Initialize model
                model = LSTMAttentionClassifier(
                    vocab_size=len(word_to_idx),
                    embedding_dim=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    num_classes=len(vocab),
                    num_layers=NUM_LAYERS,
                    bidirectional=BIDIRECTIONAL,
                    dropout=DROPOUT
                ).to(device)
                
                model.load_state_dict(checkpoint['model'])
                print(f"Model loaded from {args.model_path}")
                print(f"Best accuracy: {checkpoint['accuracy']}, achieved at epoch {checkpoint['epoch']}")
                
                # Prepare datasets if not already prepared
                if 'test_dataloader' not in locals():
                    _, _, test_dataloader, _, _ = prepare_datasets(
                        main_x, main_y, x_to_i, word_to_idx
                    )
                
                # Evaluate model
                results = evaluate_model(model, test_dataloader)
            except Exception as e:
                print(f"Error loading or evaluating model: {e}")
        
        if args.predict:
            # Load data if not already loaded
            if 'main_x' not in locals():
                main_x, main_y, vocab, i_to_x, x_to_i = load_data(limit=args.limit)
                
            # Load model
            try:
                checkpoint = torch.load(args.model_path)
                
                # Check if word_to_idx is in checkpoint
                if 'word_to_idx' in checkpoint:
                    word_to_idx = checkpoint['word_to_idx']
                else:
                    word_to_idx = build_vocabulary(main_x)
                
                # Initialize model
                model = LSTMAttentionClassifier(
                    vocab_size=len(word_to_idx),
                    embedding_dim=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    num_classes=len(vocab),
                    num_layers=NUM_LAYERS,
                    bidirectional=BIDIRECTIONAL,
                    dropout=DROPOUT
                ).to(device)
                
                model.load_state_dict(checkpoint['model'])
                print(f"Model loaded from {args.model_path}")
                
                # Predict condition
                predicted_condition, top_conditions, attention_weights = predict_condition(
                    model, word_to_idx, args.predict, i_to_x
                )
                
                print(f"\nBased on the symptoms provided, the predicted condition is: {predicted_condition}")
                print("\nTop 5 possible conditions with confidence scores:")
                for condition, score in top_conditions:
                    print(f"- {condition}: {score:.4f} ({score*100:.2f}%)")
                
            except Exception as e:
                print(f"Error loading model or making prediction: {e}")
                
if __name__ == "__main__":
    main()