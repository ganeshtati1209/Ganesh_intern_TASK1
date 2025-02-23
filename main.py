import torch
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS device detected. Reducing batch size to avoid memory errors.")
else:
    device = "cpu"

MODEL_NAME = "gpt2"
OUTPUT_DIR = "./poetry-gpt2-finetuned"
LOG_DIR = "./logs"
EPOCHS = 3
BATCH_SIZE = 2
FREEZE_LAYERS = 8

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("merve/poetry")

dataset = dataset["train"].train_test_split(test_size=0.2)


def tokenize_function(examples):

    text = examples["content"]


    tokenized_inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",  # ✅ Fix: Set to fixed size
        max_length=256,
        return_tensors="np"  # ✅ Fix: Convert to NumPy to avoid PyArrow issues
    )


    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=1)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(device)

for i, layer in enumerate(model.transformer.h):
    if i < FREEZE_LAYERS:
        for param in layer.parameters():
            param.requires_grad = False


model.gradient_checkpointing_enable()


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),
    save_on_each_node=True,
    disable_tqdm=False,
    logging_first_step=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)


trainer.train()


trainer.save_model(OUTPUT_DIR)


def generate_poetry_response(question, max_length=100):

    poetic_prompt = f"Q: {question}\nA (Poetry Style):"


    input_ids = tokenizer.encode(poetic_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    poetry = tokenizer.decode(output[0], skip_special_tokens=True)

    poetry = poetry.replace(poetic_prompt, "").strip()

    return poetry.split("\n")[0]


if __name__ == "__main__":
    print("\nFine-tuned  GPT-2 with Poetic texts")
    while True:
        user_input = input("\nEnter a question (or type 'exit' to stop): ")
        if user_input.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        poetry_response = generate_poetry_response(user_input)
        print("\nResponse:\n", poetry_response)
