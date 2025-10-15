import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoImageProcessor
from timm import create_model
from tqdm import tqdm

IMG_DIR = './data/img/'
ANNOTATIONS_PATH = './data/annotations.txt'
CHECKPOINT_DIR = './checkpoints/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_TEXT_LENGTH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256

# Load and process annotations file
def load_annotations(annotations_path):
    image_text_pairs = []
    with open(annotations_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                try:
                    image_name, text = line.strip().split('\t', 1)
                    image_text_pairs.append((image_name, text))
                except:
                    image_name = line.strip().split('\t', 1)[0]
                    image_text_pairs.append((image_name, ''))
    return image_text_pairs

# Custom Dataset (unchanged)
class ImageTextDataset(Dataset):
    def __init__(self, annotations, img_dir, tokenizer, image_processor, max_length=128):
        self.annotations = annotations
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        self.image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_name, text = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized_text.input_ids.squeeze(0)
        attention_mask = tokenized_text.attention_mask.squeeze(0)
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text  # Keep original text for evaluation
        }

# New Chained Vision-Language Model
class ChainedVisionLanguageModel(nn.Module):
    def __init__(self, 
                 vision_models=['swinv2_tiny_window16_256', 'swinv2_small_window16_256', 'swinv2_base_window12_192'],
                 text_model_name='gpt2'):
        super(ChainedVisionLanguageModel, self).__init__()
        
        # Chain of 3 Vision models (Swin Transformers)
        self.vision_models = nn.ModuleList()
        
        # First Swin model
        self.vision_models.append(create_model(
            vision_models[0],
            pretrained=True,
            num_classes=0  # Remove classifier head
        ))
        
        # Second Swin model
        self.vision_models.append(create_model(
            vision_models[1],
            pretrained=True,
            num_classes=0  # Remove classifier head
        ))
        
        # Third Swin model
        self.vision_models.append(create_model(
            vision_models[2],
            pretrained=True,
            num_classes=0  # Remove classifier head
        ))
        
        # Feature dimensions of each vision model
        vision_feature_dims = [model.num_features for model in self.vision_models]
        
        # Projection layers between vision models
        self.vision_projections = nn.ModuleList()
        for i in range(len(vision_models) - 1):
            self.vision_projections.append(
                nn.Linear(vision_feature_dims[i], vision_feature_dims[i+1])
            )
        
        # Chain of 4 GPT2 models
        self.text_models = nn.ModuleList()
        for _ in range(4):
            self.text_models.append(GPT2LMHeadModel.from_pretrained(text_model_name))
        
        # Get the embedding dimension of GPT2
        gpt_embedding_dim = self.text_models[0].config.n_embd
        
        # Final projection from vision to text domain
        self.final_vision_projection = nn.Linear(
            vision_feature_dims[-1],
            gpt_embedding_dim
        )
        
        # Projection layers between GPT models
        self.text_projections = nn.ModuleList()
        for _ in range(3):  # 3 projections for 4 GPT models
            self.text_projections.append(
                nn.Linear(gpt_embedding_dim, gpt_embedding_dim)
            )
    
    def forward(self, images, input_ids, attention_mask):
        # Process through chain of vision models
        vision_features = images
        
        for i, vision_model in enumerate(self.vision_models):
            vision_features = vision_model(vision_features)
            
            # Apply projection for all but the last vision model
            if i < len(self.vision_models) - 1:
                vision_features = self.vision_projections[i](vision_features)
        
        # Project final vision features to text model dimensions
        projected_vision_features = self.final_vision_projection(vision_features)
        
        # Create image embeddings
        image_embeddings = projected_vision_features.unsqueeze(1)  # [batch_size, 1, gpt_embedding_dim]
        
        # Process through chain of GPT models
        # Get token embeddings from first GPT model
        token_embeddings = self.text_models[0].transformer.wte(input_ids)  # [batch_size, seq_len, gpt_embedding_dim]
        
        # Combine image embeddings with token embeddings
        combined_embeddings = torch.cat([image_embeddings, token_embeddings], dim=1)  # [batch_size, 1+seq_len, gpt_embedding_dim]
        
        # Adjust attention mask to account for the added image token
        extended_attention_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        extended_attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)
        
        # Process through first GPT model
        outputs = self.text_models[0](
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask,
            labels=None
        )
        
        # Hidden states from first GPT model
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs[0]
        
        # Process through the remaining GPT models (GPT 2, 3, and 4)
        for i in range(1, len(self.text_models)):
            # Project hidden states
            projected_hidden_states = self.text_projections[i-1](hidden_states)
            
            # Process through next GPT model
            outputs = self.text_models[i](
                inputs_embeds=projected_hidden_states,
                attention_mask=extended_attention_mask,
                labels=None,
                output_hidden_states=True
            )
            
            # Update hidden states
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs[0]
        
        # Final logits from the last GPT model
        logits = self.text_models[-1].lm_head(hidden_states)
        
        # Return logits, excluding the image token prediction
        return logits[:, 1:, :]

# Modified training function
def train(model, train_loader, optimizer, criterion, device, epoch, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()  # Zero gradients at the start
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(images, input_ids, attention_mask)
        
        # Calculate loss (shift logits and labels for next-token prediction)
        shift_logits = logits[:, :-1, :]  # Remove last token prediction
        shift_labels = input_ids[:, 1:]   # Remove first token (we predict from second token onward)
        
        # Calculate loss only on valid tokens (where attention_mask is 1)
        loss = criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Step the optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    
    # Make sure to perform the last update step if needed
    if len(train_loader) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_loader)

# Evaluation function (unchanged)
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(images, input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            loss = criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Modified text generation function
def generate_text(model, image, tokenizer, device, max_length=50):
    model.eval()
    
    # Prepare image
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Start with BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    generated_text = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(image, input_ids, attention_mask)
            
            # Get next token prediction (last token in sequence)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Add to generated sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
            
            # Check if we've generated an EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            generated_text.append(next_token_id.item())
    
    return tokenizer.decode(generated_text)

# Visualization function (unchanged)
def visualize_results(model, dataset, tokenizer, device, num_samples=5):
    model.eval()
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4*num_samples))
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        
        # Get image and text
        image = sample['image']
        original_text = sample['text']
        
        # Generate text
        generated_text = generate_text(model, image, tokenizer, device)
        
        # Display image
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        # Display text comparison
        comparison_text = f"Original: {original_text}\n\nGenerated: {generated_text}"
        axes[i, 1].text(0.05, 0.5, comparison_text, fontsize=10, wrap=True)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png')
    plt.show()

# Modified main function
def main():
    print(f"Using device: {DEVICE}")
    
    # Load annotations
    annotations = load_annotations(ANNOTATIONS_PATH)
    print(f"Loaded {len(annotations)} image-text pairs")
    
    # Split into train and validation
    train_val_split = int(0.9 * len(annotations))
    train_annotations = annotations[:train_val_split]
    val_annotations = annotations[train_val_split:]
    
    # Initialize tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token by default
    image_processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    
    # Create datasets and dataloaders
    train_dataset = ImageTextDataset(train_annotations, IMG_DIR, tokenizer, image_processor, MAX_TEXT_LENGTH)
    val_dataset = ImageTextDataset(val_annotations, IMG_DIR, tokenizer, image_processor, MAX_TEXT_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model with 3 Swin models and 4 GPT models
    model = ChainedVisionLanguageModel(
        vision_models=['swinv2_tiny_window16_256', 'swinv2_small_window16_256', 'swinv2_base_window12_192'],
        text_model_name='gpt2'
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Initialize optimizer and loss function
    # Use a lower learning rate due to the increased model complexity
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Use gradient accumulation for larger effective batch size
    gradient_accumulation_steps = 4
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, epoch, gradient_accumulation_steps)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"Saved best model checkpoint (Epoch {epoch+1})")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.show()
    
    # Load best model for visualization
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize results
    visualize_results(model, val_dataset, tokenizer, DEVICE)

# Test function
def test_model():
    # Load annotations and setup
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256')
    
    # Load annotations
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    # Create test dataset
    test_dataset = ImageTextDataset(annotations, IMG_DIR, tokenizer, image_processor, MAX_TEXT_LENGTH)
    
    # Initialize model
    model = ChainedVisionLanguageModel(
        vision_models=['swinv2_tiny_window16_256', 'swinv2_small_window16_256', 'swinv2_base_window12_192'],
        text_model_name='gpt2'
    ).to(DEVICE)
    
    # Load trained model
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on a few samples
    num_test_samples = 10
    test_indices = np.random.choice(len(test_dataset), num_test_samples, replace=False)
    
    results = []
    
    for idx in test_indices:
        sample = test_dataset[idx]
        image = sample['image'].to(DEVICE)
        original_text = sample['text']
        
        # Generate text
        generated_text = generate_text(model, image, tokenizer, DEVICE)
        
        results.append({
            'image_path': os.path.join(IMG_DIR, annotations[idx][0]),
            'original_text': original_text,
            'generated_text': generated_text
        })
    
    # Display results
    for i, result in enumerate(results):
        print(f"Sample {i+1}:")
        print(f"  Image: {result['image_path']}")
        print(f"  Original: {result['original_text']}")
        print(f"  Generated: {result['generated_text']}")
        print("-" * 50)
    
    # Create visualization
    visualize_results(model, test_dataset, tokenizer, DEVICE, num_samples=5)

if __name__ == '__main__':
    # For training
    main()
    
    # For testing a trained model
    # test_model()