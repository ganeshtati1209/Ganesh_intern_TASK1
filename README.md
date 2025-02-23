# 📝 Fine-Tuning GPT-2 for Poetry-Style Text Generation

This project fine-tunes **GPT-2**, an OpenAI transformer model, on a **custom dataset (`merve/poetry`)** to generate **contextually relevant poetic responses** based on a given prompt.

## 🚀 Features
- Fine-tunes **GPT-2** on the `merve/poetry` dataset.
- Generates **poetry-style text** based on user questions.
- **Efficient layer-wise fine-tuning** to reduce memory usage.
- Interactive **prompt-response system**.

---

## 📂 Project Structure
```
📁 project-folder
│── 📄 main.py          # Script to train and generate text
│── 📄 requirements.txt # Dependencies required to run the project
│── 📄 .env             # Environment variables (Model path, API keys if needed)
│── 📜 README.md        # This file (Guide to run the project)
```

---

## 🛠️ Setup Instructions

### ✅ **1. Clone the Repository**
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### ✅ **2. Create & Activate a Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### ✅ **3. Install Dependencies**
```sh
pip install -r requirements.txt
```

### ✅ **4. Set Up Environment Variables**
- **Create a `.env` file** in the project folder:
  ```sh
  touch .env
  ```
- **Add the following content to `.env`**:
  ```
  MODEL_NAME=gpt2
  OUTPUT_DIR=./poetry-gpt2-finetuned
  ```
- The **`.env` file stores configurations**, ensuring the code runs with the right settings.

---

## 🎯 Training the Model
To fine-tune the GPT-2 model on poetry data:
```sh
python main.py
```
- This will **train the model** using `merve/poetry` dataset.
- The **fine-tuned model is saved** in `OUTPUT_DIR`.

---

## 💬 Generating Poetic Responses
After training, you can interact with the model:
```sh
python main.py
```
- **Enter a question**, and the model will generate a **poetic response**.
- **Example Usage:**
  ```
  🔵 Fine-tuned Poetry GPT-2
  💬 Enter a question (or type 'exit' to stop): What is love?
  📝 Poetic Response:
  Love is a melody, soft and bright.
  ```

---

## 🛑 Stopping Execution
To exit:
```sh
exit
```
or manually stop the script.

---

## 📌 Troubleshooting
**1️⃣ Out of Memory Errors?**
- Reduce batch size in `main.py`:
  ```python
  BATCH_SIZE = 1
  ```
- Train on a **smaller dataset**:
  ```python
  dataset = dataset["train"].shuffle(seed=42).select(range(300))
  ```

**2️⃣ Training Takes Too Long?**
- Reduce the number of **epochs**:
  ```python
  EPOCHS = 2
  ```

---

## 📜 License
This project is open-source under the **MIT License**.

---

## ⭐ Contributing
Feel free to fork and contribute by submitting a **pull request**!

---

## 🔗 References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Dataset: merve/poetry](https://huggingface.co/datasets/merve/poetry)
- [OpenAI GPT-2 Model](https://huggingface.co/gpt2)

