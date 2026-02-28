import torch
import os
import yaml  # Thêm thư viện đọc yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn.utils.rnn import pad_sequence
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from logutil import init_logger, get_logger
import datetime

output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
init_logger(output_dir)
logger = get_logger()

# Import dataset build function của bạn
from wad_dataset import build_dataset  
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from model.modeling_internvl_chat import InternVLChatModel
from transformers import AutoModel, AutoTokenizer
from model.conversation import get_conv_template


IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)

class CollaterFn:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
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

            template = get_conv_template(model.template)
            template.system_message = model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep) 
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pixel_values.shape[0]]
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
            
            input_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(answer)
            
            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids +[eos_token_id]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(attention_mask) == len(label_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(pixel_values)

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        return input_ids_tensor, label_ids_tensor, attention_mask_tensor, torch.cat(pixel_values_batch), samples_batch

def test_model(model, val_loader_with_shuffle, shuffle=False):
    # (Giữ nguyên như cũ)
    model.eval()
    with torch.no_grad():
        total_test_batches = 0
        for batch in tqdm(val_loader_with_shuffle if shuffle else val_loader):
            _, _, _, _, samples = batch # Bỏ pixel_values_batch thừa
            for sample in samples:
                pixel_values = sample['pixel_values'].cuda()
                generation_config = dict(max_new_tokens=512, do_sample=False)
                question = f"{sample['question']}"
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                logger.info(f'\nUser: {question}\nAssistant: {response}\nGround truth:{sample["answer"]}\n\n')
            total_test_batches += 1
            if total_test_batches == 5:
                break

def eval_model(model, val_loader, step, epoch, epochs):
    # (Giữ nguyên như cũ)
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_eval_batchs = 0
        for batch in tqdm(val_loader, desc=f"Validation after {step} batches training in epoch {epoch + 1}/{epochs}"):
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, _ = batch
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.cuda()
            outputs = model(
                input_ids=input_ids_batch, pixel_values=pixel_values_batch, labels=label_ids_batch, return_dict=True
            )
            loss = outputs.loss
            total_eval_loss += loss.item()
            total_eval_batchs += 1
            if total_eval_batchs == 60:
                break
        logger.info(f"Validation loss after {step} batches training in epoch {epoch + 1}/{epochs}: {total_eval_loss / total_eval_batchs}")


def train_model(model, train_loader, val_loader, val_loader_with_shuffle, epochs, lr=1e-6):
    logger.info(f"total params for Lora training: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"total trainable params for Lora training: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    NUM_ACCUMULATION_STEPS = 8
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        i = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            i += 1
            accumulated_avg_loss = 0
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, _ = batch
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.cuda()

            outputs = model(
                input_ids=input_ids_batch, pixel_values=pixel_values_batch, labels=label_ids_batch, return_dict=True
            )
            
            loss = outputs.loss / NUM_ACCUMULATION_STEPS
            accumulated_avg_loss += loss.item()
            loss.backward()
            
            if i % NUM_ACCUMULATION_STEPS == 0:
                logger.info(f"Batch {i} of epoch {epoch + 1}/{epochs}, average training loss of previous {NUM_ACCUMULATION_STEPS} batches: {accumulated_avg_loss}")
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 200 == 0:
                eval_model(model, val_loader, i, epoch, epochs)
                test_model(model, val_loader_with_shuffle, shuffle=True)
                model.train()
        
        lr_scheduler.step()
        os.makedirs(f"{output_dir}/epoch_{epoch+1}/", exist_ok=True)
        logger.info(f"Saving model {output_dir}/epoch_{epoch+1}/pytorch_model.finetuned.by.us.bin")
        # Chỉ lưu trọng số LoRA để tiết kiệm dung lượng
        model.language_model.save_pretrained(f"{output_dir}/epoch_{epoch+1}/") 


if __name__ == "__main__":
    # 1. Đọc file config yaml
    with open("internvl_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name_or_path = config['model']['name']
    epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']

    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        
        # 2. Cấu hình Quantization 4-bit (Tránh OOM)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 3. Load model (KHÔNG GỌI .cuda() ở đây vì bitsandbytes tự động đẩy lên GPU)
        logger.info(f"Loading model {model_name_or_path} in 4-bit...")
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Dùng AutoTokenizer thay vì Qwen2Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        # 4. Đóng băng Vision Model (Chỉ dạy ngôn ngữ)
        model.vision_model.requires_grad_(False)
        
        # 5. Cấu hình LoRA từ file YAML
        logger.info("Applying LoRA...")
        peft_config = LoraConfig(
            r=config['model']['lora']['r'],
            lora_alpha=config['model']['lora']['alpha'],
            target_modules=config['model']['lora']['target_modules'],
            lora_dropout=config['model']['lora']['dropout'],
            bias=config['model']['lora']['bias'],
            task_type=config['model']['lora']['task_type']
        )
        
        model.language_model = get_peft_model(model.language_model, peft_config)
        print(model.language_model)
        model.language_model.print_trainable_parameters()
        model.train() 

        # 6. Load Dataset bằng hàm build_dataset của bạn
        logger.info("Building dataset...")
        train_dataset, val_dataset = build_dataset(config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=CollaterFn(tokenizer),
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=CollaterFn(tokenizer),
            shuffle=False
        )

        val_loader_with_shuffle = DataLoader(
            val_dataset,
            batch_size=1,
            collate_fn=CollaterFn(tokenizer),
            shuffle=True
        )

        # 7. Bắt đầu train!
        train_model(model, train_loader, val_loader, val_loader_with_shuffle, epochs=epochs)