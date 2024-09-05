
# **Character-Level Text Generation Using RNN**

## **Objective**

The goal of this project is to build a **character-level Recurrent Neural Network (RNN)** capable of generating text in the style of William Shakespeare. The model is trained on a dataset of Shakespeare’s works, learning to predict the next character in a sequence. Once trained, the model can generate new, coherent text sequences that resemble Shakespeare's writing style.

---

## **Dataset**

The dataset consists of the complete works of **William Shakespeare**, obtained from a public domain source. It includes plays, sonnets, and poems, and is processed to be suitable for training a character-level RNN. Each character is treated as an individual token, and the model learns character-by-character to predict the next one in a sequence.


The dataset is tokenized at the character level, meaning each individual character (letters, punctuation, and spaces) is treated as a separate token, allowing the model to learn patterns and structure at a fine-grained level.

---

## **Model Architecture**

This project uses a **Recurrent Neural Network (RNN)** to generate text. The model is designed to capture sequential dependencies between characters and generate text that mimics the structure and style of Shakespeare's writing.

### Key components of the model:

- **Embedding Layer**: Maps each character in the input sequence to a dense vector of fixed size. This helps the model learn a more efficient representation of the input.
  
- **Recurrent Layers (LSTM or GRU)**: Two main RNN architectures, **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)**, are typically used in this type of text generation task due to their ability to capture long-term dependencies and prevent vanishing gradients.
  
- **Dense Layer**: The final layer outputs a probability distribution over all possible characters for the next step, allowing the model to predict the next character in the sequence.

---

## **Training Process**

The model is trained using **next-character prediction**. For each input sequence, the model learns to predict the next character based on the preceding characters. The loss function is calculated using **categorical cross-entropy**, and the model is optimized using **Adam** optimizer.

### Key Training Steps:

1. **Sequence Preparation**: 
   - The text is split into sequences of a fixed length (e.g., 100 characters).
   - For each sequence, the model is trained to predict the next character in the sequence.
  
2. **Batching and Shuffling**:
   - The dataset is shuffled and batched for training, ensuring the model sees different parts of the text in different batches.
  
3. **Stateful RNN Training**:
   - The model is trained to maintain its internal state between batches, allowing it to generate continuous sequences of text.

---

## **Model Evaluation**

The model's performance is evaluated based on its ability to generate coherent, human-like text. After training, the model is provided with a seed sequence (e.g., the first few characters of a Shakespearean sentence) and is asked to generate the subsequent characters.

### Evaluation Metrics:

- **Perplexity**: Perplexity is used as the main metric for evaluating the model's performance on character-level predictions. Lower perplexity indicates that the model is better at predicting the next character.
- **Text Coherency**: The model's ability to generate readable, coherent text that follows the style of the training data.
  
---

## **Text Generation**

Once the model is trained, it can be used to generate text by providing it with a starting seed (a sequence of characters). The model predicts the next character one at a time, updating its internal state with each prediction and generating text in an **autoregressive** manner.

### Customization Parameters:

- **Seed Text**: The initial sequence of characters that the model uses to start generating text.
- **Temperature**: This parameter controls the randomness of predictions. Lower temperatures make the model more conservative, while higher temperatures introduce more diversity and creativity.
- **Length of Output**: The number of characters to generate in a single run (e.g., 500 characters).

---

## **Future Improvements**

The project can be extended with the following future improvements:

- **Transformer Models**: Incorporate transformer architectures (such as GPT or BERT) for more efficient and powerful text generation.
- **Pretrained Models**: Utilize pretrained models and fine-tune them on specific datasets to reduce training time and improve results.
- **Real-Time Text Generation**: Implement real-time text generation with dynamic prompts for interactive use cases.

---


## **License**

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
