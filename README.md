# Tom: A Multimodal Voice Assistant Chatbot 🤖🎙️

Tom is a versatile voice assistant chatbot that combines text and code explanation capabilities with image understanding through the use of OpenAI's **GPT** model for text, **Gemini** for image processing, and **Whisper** for speech-to-text and text-to-speech conversion. 🌟

## Features

- **Text and Code Explanation**: Utilizes **LLAMA** model to provide detailed explanations for text and code inputs. 💬
- **Image Understanding**: Employs **Gemini API** to comprehend images captured via screenshots or webcams, providing rich context for responses. 📸
- **Speech Interaction**: Incorporates **Whisper** for speech-to-text conversion, enabling seamless voice interaction. 🗣️
- **Text-to-Speech Output**: Employs **Whisper** for converting response text into speech, making interactions natural and accessible. 🔊

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/tamim-18/tom_a_multimodal_chatbot
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure proper setup of **OpenAI** and **Google API** keys in the `.env` file.

## Usage

1. Start the voice assistant:

    ```bash
    python3.11 final.py
    ```

2. Upon running, the voice assistant listens for the wake word followed by a user prompt. Use the wake word "**Tom**" to initiate interaction.

3. You can provide prompts either through speech or text input.

4. Tom will process the prompt, perform necessary actions (such as taking screenshots, analyzing images, etc.), and provide a response.

5. The response will be both displayed as text and spoken aloud using text-to-speech conversion.

## Video Demo 📹

Check out our video demo to see Tom in action!📣


https://github.com/tamim-18/tom_a_multimodal_chatbot/assets/61451847/e37eda82-65fa-4c9c-b473-2dee4dc1b331




## Example Workflow

1. User: "Tom, what are you seeing on my webcam?"
2. Tom: *Describes the scene captured by the webcam*
3. Tom: *Spoken response is also provided using text-to-speech*

## Contributing

Contributions are welcome! If you have any ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request. 🚀


