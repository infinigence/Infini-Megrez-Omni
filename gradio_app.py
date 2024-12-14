# -*- encoding: utf-8 -*-
# File: app.py
# Description: None


import threading
from copy import deepcopy
from typing import Dict, List

import gradio as gr
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v")
AUDIO_EXTENSIONS = (".mp3", ".wav")

DEFAULT_SAMPLING_PARAMS = {
    "top_p": 0.8,
    "top_k": 100,
    "temperature": 0.7,
    "do_sample": True,
    "num_beams": 1,
    "repetition_penalty": 1.2,
}
MAX_NEW_TOKENS = 1024


def main(model_path: str, port: int):

    if gr.NO_RELOAD:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            .eval()
            .cuda()
        )
        iterable_streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30,
        )

    def history2messages(history: List[Dict]) -> List[Dict]:
        """
        Transform gradio history to chat messages.
        """
        messages = []
        cur_message = dict()
        for item in history:
            if item["role"] == "assistant":
                if len(cur_message) > 0:
                    messages.append(deepcopy(cur_message))
                    cur_message = dict()
                messages.append(deepcopy(item))
                continue

            if "role" not in cur_message:
                cur_message["role"] = "user"
            if "content" not in cur_message:
                cur_message["content"] = dict()

            if "metadata" not in item:
                item["metadata"] = {"title": None}
            if item["metadata"]["title"] is None:
                cur_message["content"]["text"] = item["content"]
            elif item["metadata"]["title"] == "image":
                cur_message["content"]["image"] = item["content"][0]
            elif item["metadata"]["title"] == "audio":
                cur_message["content"]["audio"] = item["content"][0]
        if len(cur_message) > 0:
            messages.append(cur_message)
        return messages

    def check_messages(history, message, audio):
        audios = []
        images = []

        for file_msg in message["files"]:
            if file_msg.endswith(AUDIO_EXTENSIONS) or file_msg.endswith(VIDEO_EXTENSIONS):
                audios.append(file_msg)
            elif file_msg.endswith(IMAGE_EXTENSIONS):
                images.append(file_msg)
            else:
                filename = file_msg.split("/")[-1]
                raise gr.Error(f"Unsupported file type: {filename}. It should be an image or audio file.")

        if len(audios) > 1:
            raise gr.Error("Please upload only one audio file.")

        if len(images) > 1:
            raise gr.Error("Please upload only one image file.")

        if audio is not None:
            if len(audios) > 0:
                raise gr.Error("Please upload only one audio file or record audio.")
            audios.append(audio)

        # Append the message to the history
        for image in images:
            history.append({"role": "user", "content": (image,), "metadata": {"title": "image"}})

        for audio in audios:
            history.append({"role": "user", "content": (audio,), "metadata": {"title": "audio"}})

        if message["text"] is not None:
            history.append({"role": "user", "content": message["text"]})

        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def bot(
        history: list,
        top_p: float,
        top_k: int,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int = MAX_NEW_TOKENS,
        regenerate: bool = False,
    ):
        sampling_params = {
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }

        if regenerate:
            history = history[:-1]

        msgs = history2messages(history)
        th = threading.Thread(
            target=model.chat,
            kwargs=dict(
                input_msgs=msgs,
                sampling=True,
                streamer=iterable_streamer,
                max_new_tokens=max_new_tokens,
                **sampling_params,
            ),
        )
        th.start()

        response = ""
        for subtext in iterable_streamer:
            response += subtext
            yield history + [{"role": "assistant", "content": response}]

        th.join()
        return response

    def change_state(state):
        return gr.update(visible=not state), not state

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", height=800)

        sampling_params_group_hidden_state = gr.State(False)

        with gr.Row(equal_height=True):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                scale=4,
            )
            chat_input = gr.MultimodalTextbox(
                file_count="multiple",
                show_label=False,
                scale=10,
                file_types=["image", "audio"],
                # stop_btn=True,
            )
            with gr.Column(scale=1, min_width=150):
                with gr.Row(equal_height=True):
                    regenerate_btn = gr.Button("Regenerate", variant="primary")
                    clear_btn = gr.ClearButton(
                        [chat_input, audio_input, chatbot],
                    )

        with gr.Row():
            sampling_params_toggle_btn = gr.Button("Sampling Parameters")

        with gr.Group(visible=False) as sampling_params_group:
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0, maximum=1.2, value=DEFAULT_SAMPLING_PARAMS["temperature"], label="Temperature"
                )
                repetition_penalty = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
                    label="Repetition Penalty",
                )

            with gr.Row():
                top_p = gr.Slider(minimum=0, maximum=1, value=DEFAULT_SAMPLING_PARAMS["top_p"], label="Top-p")
                top_k = gr.Slider(minimum=0, maximum=1000, value=DEFAULT_SAMPLING_PARAMS["top_k"], label="Top-k")

            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=MAX_NEW_TOKENS,
                    value=MAX_NEW_TOKENS,
                    label="Max New Tokens",
                    interactive=True,
                )

        sampling_params_toggle_btn.click(
            change_state,
            sampling_params_group_hidden_state,
            [sampling_params_group, sampling_params_group_hidden_state],
        )

        chat_msg = chat_input.submit(
            check_messages,
            [chatbot, chat_input, audio_input],
            [chatbot, chat_input],
        )
        bot_msg = chat_msg.then(
            bot,
            inputs=[chatbot, top_p, top_k, temperature, repetition_penalty, max_new_tokens],
            outputs=chatbot,
            api_name="bot_response",
        )
        bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

        regenerate_btn.click(
            bot,
            inputs=[chatbot, top_p, top_k, temperature, repetition_penalty, max_new_tokens, gr.State(True)],
            outputs=chatbot,
        )

    demo.launch(server_port=port)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=7680)
    args = parser.parse_args()

    main(args.model_path, args.port)
