import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import yaml
import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from logutil import init_logger, get_logger
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. ĐỌC CONFIG VÀ KHỞI TẠO LOGGER NGAY LẬP TỨC (TRƯỚC KHI IMPORT MODEL)
with open("internvl_config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_out_dir = config['training']['output_dir']
output_dir = f'{base_out_dir}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)
init_logger(output_dir)
logger = get_logger()

# 2. BÂY GIỜ MỚI IMPORT CÁC MODULE CỦA BẠN (Đã an toàn)
from wad_dataset import build_dataset  
from model.modeling_internvl_chat import InternVLChatModel
from model.conversation import get_conv_template

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)

class CollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model # Truyền model vào thay vì dùng biến global
    
    def __call__(self, batch):
        label_ids_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        samples_batch = []

        for sample in batch:
            question = sample['question']
            answer = sample['answer']
            pixel_values = sample['pixel_values']
            samples_batch.append(sample)

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep) 
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pv.shape[0] for pv in pixel_values]
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            
            input_ids = self.tokenizer.encode(query, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            
            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(attention_mask) == len(label_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(torch.cat(pixel_values, dim=0))

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        return input_ids_tensor, label_ids_tensor, attention_mask_tensor, torch.cat(pixel_values_batch), samples_batch

def test_model(model, tokenizer, val_loader_with_shuffle, shuffle=False):
    model.eval()
    with torch.no_grad():
        total_test_batches = 0
        for batch in tqdm(val_loader_with_shuffle):
            _, _, _, _, samples = batch 
            for sample in samples:
                pixel_values = torch.cat(sample['pixel_values'], dim=0).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=512, do_sample=False)
                question = f"{sample['question']}"
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                logger.info(f'\nUser: {question}\nAssistant: {response}\nGround truth:{sample["answer"]}\n\n')
            total_test_batches += 1
            if total_test_batches == 2:
                break

def eval_model(model, val_loader, step, epoch, epochs):
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_eval_batchs = 0
        for batch in tqdm(val_loader, desc=f"Validation after {step} batches training in epoch {epoch + 1}/{epochs}"):
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, _ = batch
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()
            
            outputs = model(
                input_ids=input_ids_batch, pixel_values=pixel_values_batch, labels=label_ids_batch, image_flags=image_flags_batch, return_dict=True
            )
            loss = outputs.loss
            total_eval_loss += loss.item()
            total_eval_batchs += 1
            if total_eval_batchs == 200:
                break
        logger.info(f"Validation loss after {step} batches training in epoch {epoch + 1}/{epochs}: {total_eval_loss / total_eval_batchs:.4f}")

import os
import torch

def train_model(model, tokenizer, train_loader, val_loader, val_loader_with_shuffle, config, output_dir, resume_dir=None, start_epoch=0, start_step=0):
    epochs = config['training']['num_epochs']
    lr = float(config['training']['learning_rate'])
    accum_steps = config['training']['gradient_accumulation_steps']
    weight_decay = float(config['training']['weight_decay'])
    warmup_steps = config['training']['warmup_steps']
    max_grad_norm = float(config['training']['max_grad_norm'])

    logger.info(f"Total params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable params for LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Training config: LR={lr}, Accum_steps={accum_steps}, Weight_decay={weight_decay}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_training_steps = (len(train_loader) * epochs) // accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    # ==========================================
    # [MỚI] 1. LOAD OPTIMIZER & SCHEDULER TỪ CHECKPOINT
    # ==========================================
    if resume_dir and os.path.exists(resume_dir):
        logger.info(f"Resuming training from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        
        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")
        
        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(torch.load(opt_path))
            lr_scheduler.load_state_dict(torch.load(sch_path))
            logger.info("Loaded Optimizer and Scheduler states successfully!")
        else:
            logger.warning("No Optimizer/Scheduler states found in checkpoint. Starting with fresh states.")

    # ==========================================
    # [MỚI] 2. BẮT ĐẦU TỪ START_EPOCH THAY VÌ 0
    # ==========================================
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        
        accumulated_loss_for_log = 0.0 
        set_seed(42 + epoch)  # Đảm bảo mỗi epoch có seed khác nhau nhưng vẫn reproducible
        batch_iterator = iter(train_loader) # Biến loader thành băng chuyền
        
        # ==========================================
        # [MỚI] 3. TUA NHANH BATCH (CHỈ TUA Ở EPOCH ĐẦU TIÊN KHI RESUME)
        # ==========================================
        if epoch == start_epoch and start_step > 0:
            logger.info(f" Skipping {start_step} batches to resume state...")
            for _ in range(start_step):
                next(batch_iterator) # Bốc data vứt đi, không đưa vào train
            i = start_step # Cập nhật lại biến đếm
        else:
            i = 0
            
        # ==========================================
        # [MỚI] 4. DÙNG BATCH_ITERATOR ĐỂ CHẠY
        # ==========================================
        for batch in tqdm(batch_iterator, desc=f"Training Epoch {epoch + 1}/{epochs}", total=len(train_loader), initial=i):
            i += 1
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, _ = batch
            
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()

            outputs = model(
                input_ids=input_ids_batch, 
                pixel_values=pixel_values_batch, 
                labels=label_ids_batch, 
                image_flags=image_flags_batch, 
                return_dict=True
            )
            
            loss = outputs.loss / accum_steps
            loss.backward()
            
            accumulated_loss_for_log += outputs.loss.item()
            
            if i % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                avg_loss = accumulated_loss_for_log / accum_steps
                logger.info(f"Step {i//accum_steps} | Avg Loss (last {accum_steps} batches): {avg_loss:.4f}")
                accumulated_loss_for_log = 0.0 
                
                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()
            
            if i % config['training']['eval_steps'] == 0:
                eval_model(model, val_loader, i, epoch, epochs)
                test_model(model, tokenizer, val_loader_with_shuffle, shuffle=True)
                model.train()
                
            if i % config['training']['save_steps'] == 0: 
                step_save_dir = f"{output_dir}/epoch_{epoch+1}_step_{i}/"
                os.makedirs(step_save_dir, exist_ok=True)
                logger.info(f"Saving model, tokenizer, opt, scheduler at step {i} to {step_save_dir}")
                
                model.language_model.save_pretrained(step_save_dir)
                tokenizer.save_pretrained(step_save_dir) 
                # ==========================================
                # [MỚI] 5. LƯU THÊM OPTIMIZER VÀ SCHEDULER
                # ==========================================
                torch.save(optimizer.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(step_save_dir, "scheduler.pt"))
        
        epoch_save_dir = f"{output_dir}/epoch_{epoch+1}/"
        os.makedirs(epoch_save_dir, exist_ok=True)
        logger.info(f"Saving model and tokenizer for epoch {epoch+1} to {epoch_save_dir}")
        model.language_model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        # Lưu Optimizer cho cuối epoch
        torch.save(optimizer.state_dict(), os.path.join(epoch_save_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(epoch_save_dir, "scheduler.pt"))


if __name__ == "__main__":

    model_name_or_path = config['model']['name']
    batch_size = config['training']['batch_size']
        
    # 2. Cấu hình Quantization 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['quantization']['enabled'],
        bnb_4bit_compute_dtype=torch.bfloat16 if config['model']['quantization']['compute_dtype'] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config['model']['quantization']['double_quant'],
        bnb_4bit_quant_type=config['model']['quantization']['type']
    )
    # 3. Load model
    logger.info(f"Loading model {model_name_or_path} in 4-bit...")
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=config['model']['trust_remote_code']
    )

    model.config.use_cache = False
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    # 4. Đóng băng Vision Model
    if config['model']['vision']['freeze_encoder']:
        model.vision_model.requires_grad_(False)
    
    # 5. Cấu hình LoRA
    logger.info("Applying LoRA...")
    model.language_model = prepare_model_for_kbit_training(model.language_model)

    if hasattr(model.language_model, "get_input_embeddings"):
        model.language_model.get_input_embeddings().to(torch.bfloat16)

    hf_repo_id = "huyvanzzz/internvl2.5_config1"
    # peft_config = LoraConfig(
    #     r=config['model']['lora']['r'],
    #     lora_alpha=config['model']['lora']['alpha'],
    #     target_modules=config['model']['lora']['target_modules'],
    #     lora_dropout=config['model']['lora']['dropout'],
    #     bias=config['model']['lora']['bias'],
    #     task_type=config['model']['lora']['task_type']
    # )
    
    # model.language_model = get_peft_model(model.language_model, peft_config)
    # model.language_model.print_trainable_parameters()
    # model.train()
    
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        hf_repo_id,
        is_trainable=True # BẮT BUỘC ĐỂ TRUE nếu bạn muốn train tiếp. Nếu chỉ chạy test thì để False.
    )

    model.language_model.print_trainable_parameters()
    model.train()

    # 6. Load Dataset
    logger.info("Building dataset...")
    train_dataset, val_dataset = build_dataset(config)
    
    collate_fn_wrapper = CollaterFn(tokenizer, model)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=False
    )

    val_loader_with_shuffle = DataLoader(
        val_dataset, batch_size=1, collate_fn=collate_fn_wrapper, shuffle=True
    )

    # 7. Bắt đầu train
    logger.info("STARTING TRAINING...")
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        val_loader_with_shuffle=val_loader_with_shuffle,
        config=config,
        output_dir=output_dir,
        resume_dir=None,
        start_epoch=1,
        start_step=1999,
    )