Intern_Ganesh_Task1
📝 Fine-Tuning GPT-2 for Poetry-Style Text Generation
This project fine-tunes GPT-2, an OpenAI transformer model, on a custom dataset (merve/poetry) to generate contextually relevant poetic responses based on a given prompt.

🚀 Features
Fine-tunes GPT-2 on the merve/poetry dataset.
Generates poetry-style text based on user questions.
Efficient layer-wise fine-tuning to reduce memory usage.
Interactive prompt-response system.
📂 Project Structure
📁 project-folder
│── 📄 main.py          # Script to train and generate text
│── 📄 requirements.txt # Dependencies required to run the project
│── 📄 .env             # Environment variables (Model path, API keys if needed)
│── 📜 README.md        # This file (Guide to run the project)


---

## 🛠️ Setup Instructions

### ✅ **1. Clone the Repository**
```sh
git clone https://github.com/ganeshtati1209/Intern_Ganesh_Task1.git
cd YOUR_REPO_NAME
✅ 2. Create & Activate a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
✅ 3. Install Dependencies
pip install -r requirements.txt
🎯 Training the Model
To fine-tune the GPT-2 model on poetry data:

python main.py
This will train the model using merve/poetry dataset.
The fine-tuned model is saved in OUTPUT_DIR.
💬 Generating Poetic Responses
After training, you can interact with the model:

python main.py
Enter a question, and the model will generate a poetic response.
Example Usage:
🔵 Fine-tuned Poetry GPT-2
💬 Enter a question (or type 'exit' to stop): What is love?
📝 Poetic Response:
Love is a melody, soft and bright.
🛑 Stopping Execution
To exit:

exit
or manually stop the script.

📌 Troubleshooting
1️⃣ Out of Memory Errors?

Reduce batch size in main.py:
BATCH_SIZE = 1
Train on a smaller dataset:
dataset = dataset["train"].shuffle(seed=42).select(range(300))
2️⃣ Training Takes Too Long?

Reduce the number of epochs:
EPOCHS = 2
📜 License
This project is open-source .

⭐ Contributing
Feel free to fork and contribute by submitting a pull request!

🔗 References
Hugging Face Transformers
Dataset: merve/poetry
OpenAI GPT-2 Model
