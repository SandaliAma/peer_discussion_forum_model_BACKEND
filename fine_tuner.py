"""
Complete Fine-Tuning System for SinhaLM on Mathematics Problems
Trains on your specific mathematics dataset
"""

import os
import json
import torch
from datetime import datetime
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
import logging

import config
from math_rag_system import MathProblemProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATASET PREPARATION
# ============================================================================

class FineTuningDatasetBuilder:
    """Prepare dataset for fine-tuning"""
    
    @staticmethod
    def prepare_training_data(problems: List[Dict]) -> Dataset:
        """
        Convert problems to instruction-following format
        Format: Question -> Step-by-step solution -> Final answer
        """
        
        logger.info("Preparing training data...")
        
        training_examples = []
        
        for problem in problems:
            # Create instruction format
            instruction = f"""ප්‍රශ්නය: {problem['question']}

මෙම ගණිත ගැටලුව පියවරෙන් පියවර විසඳන්න.

විසඳුම:
"""
            
            # Full solution with steps
            full_text = instruction + problem['solution'] + f"\n\nඅවසාන පිළිතුර: {problem['final_answer']}"
            
            training_examples.append({
                'text': full_text,
                'topic': problem['topic'],
                'sub_topic': problem['sub_topic'],
                'question_length': len(problem['question']),
                'solution_length': len(problem['solution'])
            })
        
        dataset = Dataset.from_list(training_examples)
        logger.info(f"Created dataset with {len(dataset)} examples")
        
        return dataset
    
    @staticmethod
    def split_dataset(dataset: Dataset, train_ratio: float = config.TRAIN_TEST_SPLIT):
        """Split dataset into train and validation"""
        
        split_dataset = dataset.train_test_split(
            test_size=1-train_ratio,
            shuffle=True,
            seed=42
        )
        
        logger.info(f"Train examples: {len(split_dataset['train'])}")
        logger.info(f"Validation examples: {len(split_dataset['test'])}")
        
        return split_dataset['train'], split_dataset['test']

# ============================================================================
# FINE-TUNING TRAINER
# ============================================================================

class MathSinhaLMFineTuner:
    """Fine-tune SinhaLM on mathematics problems"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def prepare_model(self):
        """Load base model and SinhaLM, prepare for fine-tuning"""
        
        logger.info("="*70)
        logger.info("PREPARING MODEL FOR FINE-TUNING")
        logger.info("="*70)
        
        # Load tokenizer
        logger.info(f"\n[1/4] Loading tokenizer: {config.BASE_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"\n[2/4] Loading base model: {config.BASE_MODEL_NAME}")
        logger.info("This will take 2-3 minutes on first run...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            device_map=config.DEVICE,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load existing SinhaLM adapter
        logger.info(f"\n[3/4] Loading SinhaLM adapter: {config.SINHALM_MODEL_PATH}")
        
        if os.path.exists(config.SINHALM_MODEL_PATH):
            self.model = PeftModel.from_pretrained(base_model, config.SINHALM_MODEL_PATH)
            logger.info("✓ SinhaLM adapter loaded")
        else:
            logger.warning("SinhaLM adapter not found, using base model only")
            self.model = base_model
        
        # Add NEW LoRA layers for mathematics
        logger.info(f"\n[4/4] Adding new LoRA layers for mathematics...")
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"\n Model Statistics:")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        logger.info("\n✓ Model prepared for fine-tuning")
    
    def prepare_dataset(self, train_dataset: Dataset, val_dataset: Dataset):
        """Tokenize datasets"""
        
        logger.info("\n" + "="*70)
        logger.info("PREPARING DATASETS")
        logger.info("="*70)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.FINETUNE_MAX_LENGTH,
                padding="max_length"
            )
        
        logger.info("\nTokenizing training data...")
        self.train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        logger.info("Tokenizing validation data...")
        self.val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation"
        )
        
        logger.info("✓ Datasets prepared")
    
    def train(self):
        """Start fine-tuning"""
        
        logger.info("\n" + "="*70)
        logger.info("STARTING FINE-TUNING")
        logger.info("="*70)
        
        # Calculate training steps
        total_steps = (len(self.train_dataset) // (config.FINETUNE_BATCH_SIZE * config.FINETUNE_GRADIENT_ACCUMULATION)) * config.FINETUNE_EPOCHS
        
        logger.info(f"\n Training Configuration:")
        logger.info(f"   Training examples: {len(self.train_dataset)}")
        logger.info(f"   Validation examples: {len(self.val_dataset)}")
        logger.info(f"   Epochs: {config.FINETUNE_EPOCHS}")
        logger.info(f"   Batch size: {config.FINETUNE_BATCH_SIZE}")
        logger.info(f"   Gradient accumulation: {config.FINETUNE_GRADIENT_ACCUMULATION}")
        logger.info(f"   Effective batch size: {config.FINETUNE_BATCH_SIZE * config.FINETUNE_GRADIENT_ACCUMULATION}")
        logger.info(f"   Total training steps: {total_steps}")
        logger.info(f"   Learning rate: {config.FINETUNE_LEARNING_RATE}")
        
        # Estimate time
        logger.info(f"\n  Estimated time on CPU:")
        logger.info(f"   ~{total_steps * 3 / 60:.1f} - {total_steps * 5 / 60:.1f} minutes")
        logger.info(f"   ({total_steps * 3 / 3600:.1f} - {total_steps * 5 / 3600:.1f} hours)")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.FINETUNE_OUTPUT_DIR,
            num_train_epochs=config.FINETUNE_EPOCHS,
            per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
            per_device_eval_batch_size=config.FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=config.FINETUNE_GRADIENT_ACCUMULATION,
            learning_rate=config.FINETUNE_LEARNING_RATE,
            warmup_steps=config.FINETUNE_WARMUP_STEPS,
            logging_steps=config.FINETUNE_LOGGING_STEPS,
            save_steps=config.FINETUNE_SAVE_STEPS,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=config.FINETUNE_SAVE_STEPS,
            fp16=False,  # CPU doesn't support fp16
            dataloader_num_workers=0,
            optim="adamw_torch",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("\n Starting training...\n")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        train_result = self.trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info(" TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n  Total training time: {duration / 60:.1f} minutes ({duration / 3600:.2f} hours)")
        logger.info(f" Final training loss: {train_result.training_loss:.4f}")
        
        # Save model
        self._save_model()
        
        return train_result
    
    def _save_model(self):
        """Save the fine-tuned model"""
        
        logger.info(f"\n Saving fine-tuned model to {config.FINETUNE_OUTPUT_DIR}")
        
        self.model.save_pretrained(config.FINETUNE_OUTPUT_DIR)
        self.tokenizer.save_pretrained(config.FINETUNE_OUTPUT_DIR)
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'base_model': config.BASE_MODEL_NAME,
            'sinhalm_path': config.SINHALM_MODEL_PATH,
            'output_path': config.FINETUNE_OUTPUT_DIR,
            'config': {
                'epochs': config.FINETUNE_EPOCHS,
                'batch_size': config.FINETUNE_BATCH_SIZE,
                'learning_rate': config.FINETUNE_LEARNING_RATE,
                'lora_r': config.LORA_R,
                'lora_alpha': config.LORA_ALPHA
            }
        }
        
        with open(os.path.join(config.FINETUNE_OUTPUT_DIR, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("✓ Model saved successfully")

# ============================================================================
# MAIN FINE-TUNING PIPELINE
# ============================================================================

def run_finetuning():
    """Complete fine-tuning pipeline"""
    
    print("\n" + "="*70)
    print("SINHALM MATHEMATICS FINE-TUNING PIPELINE")
    print("="*70)
    
    # Step 1: Load problems
    print("\n[STEP 1/5] Loading mathematics problems...")
    processor = MathProblemProcessor()
    problems = processor.process_all(config.MATH_PROBLEMS_JSON)
    
    stats = processor.get_statistics(problems)
    print(f"✓ Loaded {stats['total']} problems")
    
    if stats['total'] < 10:
        print("\n  WARNING: Very few training examples!")
        print("   Recommend at least 50-100 examples for good fine-tuning")
        proceed = input("   Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Fine-tuning cancelled.")
            return
    
    # Step 2: Prepare dataset
    print("\n[STEP 2/5] Preparing training dataset...")
    dataset_builder = FineTuningDatasetBuilder()
    dataset = dataset_builder.prepare_training_data(problems)
    train_dataset, val_dataset = dataset_builder.split_dataset(dataset)
    print("✓ Dataset prepared")
    
    # Step 3: Initialize fine-tuner
    print("\n[STEP 3/5] Initializing fine-tuner...")
    fine_tuner = MathSinhaLMFineTuner()
    
    # Step 4: Prepare model
    print("\n[STEP 4/5] Preparing model...")
    fine_tuner.prepare_model()
    fine_tuner.prepare_dataset(train_dataset, val_dataset)
    print("✓ Model and datasets ready")
    
    # Step 5: Train
    print("\n[STEP 5/5] Fine-tuning...")
    print("\n  This will take time on CPU. You can monitor progress below.")
    print("    Press Ctrl+C to stop training (model will be saved)\n")
    
    try:
        train_result = fine_tuner.train()
        
        print("\n" + "="*70)
        print(" FINE-TUNING COMPLETE!")
        print("="*70)
        print(f"\n Model saved to: {config.FINETUNE_OUTPUT_DIR}")
        print("\nYou can now use the fine-tuned model by running:")
        print("   python run_system.py")
        print("   Choose option 1 (Quick Test)")
        print("   When prompted, select 'Use fine-tuned model'")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        print("Saving current model state...")
        fine_tuner._save_model()
        print("Model saved. You can resume training or use the current checkpoint.")
    
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_finetuning()