import opendatasets as od
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import argparse
import os

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model configuration
EMBED_SIZE = 300
BERT_MODEL = 'prajjwal1/bert-mini'
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512

class DrugClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DrugClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.lin1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.lin2 = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output1 = self.lin1(pooled_output)
        output2 = self.lin2(output1)
        return output2

def download_dataset():
    """Download the dataset from Kaggle if it doesn't exist"""
    if not os.path.exists('./kuc-hackathon-winter-2018'):
        print("Downloading dataset from Kaggle...")
        od.download('https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018')
    else:
        print("Dataset already downloaded.")

def load_data(limit=None):
    """Load and prepare the data"""
    kaggle_train = pd.read_csv('./kuc-hackathon-winter-2018/drugsComTrain_raw.csv')
    kaggle_test = pd.read_csv('./kuc-hackathon-winter-2018/drugsComTest_raw.csv')
    
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

def prepare_datasets(main_x, main_y, x_to_i, tokenizer):
    """Split the data and prepare datasets for training"""
    x_train, x_test, y_train, y_test = train_test_split(main_x, main_y, test_size=.21, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)
    
    # Prepare training data
    x_train_encoded = tokenizer(x_train, truncation=True, max_length=MAX_SEQ_LENGTH, padding=True, return_tensors='pt')
    y_train_encoded = torch.tensor([x_to_i[j] for j in y_train], dtype=torch.long)
    
    # Prepare validation data
    x_val_encoded = tokenizer(x_val, truncation=True, max_length=MAX_SEQ_LENGTH, padding=True, return_tensors='pt')
    y_val_encoded = torch.tensor([x_to_i[j] for j in y_val], dtype=torch.long)
    
    # Prepare test data
    x_test_encoded = tokenizer(x_test, truncation=True, max_length=MAX_SEQ_LENGTH, padding=True, return_tensors='pt')
    y_test_encoded = torch.tensor([x_to_i[j] for j in y_test], dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(
        x_train_encoded['input_ids'], 
        x_train_encoded['attention_mask'], 
        x_train_encoded['token_type_ids'], 
        y_train_encoded
    )
    
    val_dataset = TensorDataset(
        x_val_encoded['input_ids'], 
        x_val_encoded['attention_mask'], 
        x_val_encoded['token_type_ids'], 
        y_val_encoded
    )
    
    test_dataset = TensorDataset(
        x_test_encoded['input_ids'], 
        x_test_encoded['attention_mask'], 
        x_test_encoded['token_type_ids'], 
        y_test_encoded
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def train_model(model, train_dataloader, val_dataloader, save_path='./best_model.pt'):
    """Train the model and save the best checkpoint"""
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_model = {'accuracy': -1, 'epoch': -1, 'model': {}, 'optimizer': {}}
    
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}')
        
        # Training
        model.train()
        losses = []
        accuracies = []
        f1_scores = []
        
        for input_ids, attention_mask, token_type_ids, labels in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_func(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.max(out, dim=1)[1]
            acc = (pred == labels).float().mean()
            accuracies.append(acc.item())
            
            f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
            f1_scores.append(f1)
            
            losses.append(loss.item())
        
        print(f'Train Loss: {sum(losses)/len(losses):.4f}')
        print(f'Train Accuracy: {sum(accuracies)/len(accuracies):.4f}')
        print(f'Train F1 score: {sum(f1_scores)/len(f1_scores):.4f}')
        
        # Validation
        model.eval()
        val_accuracies = []
        val_losses = []
        val_f1 = []
        
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in tqdm(val_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                pred = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(pred, labels)
                
                pred_class = torch.max(pred, dim=1)[1]
                acc = (pred_class == labels).float().mean()
                val_accuracies.append(acc.item())
                
                f1 = f1_score(labels.cpu().numpy(), pred_class.cpu().numpy(), average='weighted')
                val_f1.append(f1)
                
                val_losses.append(loss.item())
                
        val_acc = sum(val_accuracies)/len(val_accuracies)
        print(f'Dev Loss: {sum(val_losses)/len(val_losses):.4f}')
        print(f'Dev Accuracy: {val_acc:.4f}')
        print(f'Dev F1 score: {sum(val_f1)/len(val_f1):.4f}')
        
        # Save the best model
        if best_model['accuracy'] < val_acc:
            best_model['accuracy'] = val_acc
            best_model['epoch'] = epoch+1
            best_model['model'] = model.state_dict()
            best_model['optimizer'] = optimizer.state_dict()
    
    # Save the best model
    torch.save({
        'accuracy': best_model['accuracy'],
        'epoch': best_model['epoch'],
        'model': best_model['model'],
        'optimizer': best_model['optimizer']
    }, save_path)
    
    print(f"Best model saved at epoch {best_model['epoch']} with accuracy {best_model['accuracy']:.4f}")
    return best_model

def evaluate_model(model, test_dataloader):
    """Evaluate the model on the test set"""
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    test_accuracies = []
    test_f1_scores = []
    test_losses = []
    
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in tqdm(test_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_func(pred, labels)
            
            pred_class = torch.max(pred, dim=1)[1]
            acc = (pred_class == labels).float().mean()
            test_accuracies.append(acc.item())
            
            f1 = f1_score(labels.cpu().numpy(), pred_class.cpu().numpy(), average='weighted')
            test_f1_scores.append(f1)
            
            test_losses.append(loss.item())
    
    print(f'Test Loss: {sum(test_losses)/len(test_losses):.4f}')
    print(f'Test Accuracy: {sum(test_accuracies)/len(test_accuracies):.4f}')
    print(f'Test F1 score: {sum(test_f1_scores)/len(test_f1_scores):.4f}')

def predict_condition(model, tokenizer, symptoms_text, i_to_x):
    """Predict the medical condition based on symptoms"""
    model.eval()
    
    # Tokenize the symptoms text
    inputs = tokenizer(
        symptoms_text, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH, 
        padding=True, 
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
    
    return predicted_condition, top_conditions

def main():
    parser = argparse.ArgumentParser(description='Medical Condition Classification')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict condition from symptoms')
    parser.add_argument('--model_path', type=str, default='./best_model.pt', help='Path to save/load model')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    
    args = parser.parse_args()
    
    # Download dataset if needed
    download_dataset()
    
    # Load data
    main_x, main_y, vocab, i_to_x, x_to_i = load_data(limit=args.limit)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    
    # Initialize model
    model = DrugClassifier(len(vocab)).to(device)
    
    if args.train:
        # Prepare datasets
        train_dataloader, val_dataloader, test_dataloader = prepare_datasets(
            main_x, main_y, x_to_i, tokenizer
        )
        
        # Train model
        best_model = train_model(model, train_dataloader, val_dataloader, save_path=args.model_path)
        
        if args.eval:
            # Load best model
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model'])
            
            # Evaluate
            evaluate_model(model, test_dataloader)
    
    elif args.eval:
        # Prepare datasets
        _, _, test_dataloader = prepare_datasets(main_x, main_y, x_to_i, tokenizer)
        
        # Load model
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        
        # Evaluate
        evaluate_model(model, test_dataloader)
    
    if args.predict:
        # Load model if not already loaded
        if not args.train:
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model'])
        
        # Make prediction
        predicted_condition, top_conditions = predict_condition(
            model, tokenizer, args.predict, i_to_x
        )
        
        print(f"\nBased on the symptoms provided, the predicted condition is: {predicted_condition}")
        print("\nTop 5 possible conditions with confidence scores:")
        for condition, score in top_conditions:
            print(f"- {condition}: {score:.4f} ({score*100:.2f}%)")

if __name__ == "__main__":
    main()