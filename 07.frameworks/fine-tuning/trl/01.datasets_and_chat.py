"""01 · TRL dataset formats and chat templates.

TRL trainers accept a few standard dataset shapes. Getting the shape right is
half the battle. We build each and render a chat template so you can SEE the
exact string the model trains on.

  - "language modeling"   : {"text": "..."}                      (SFT raw text)
  - "prompt-completion"   : {"prompt": "...", "completion": "..."}(SFT, masked prompt)
  - "preference"          : {"prompt","chosen","rejected"}        (DPO/ORPO/KTO)
  - "conversational"      : {"messages": [{role, content}, ...]}  (chat SFT)

Docs: https://huggingface.co/docs/trl/dataset_formats
Tiny model `sshleifer/tiny-gpt2`. CPU. Exit code 0.
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from datasets import Dataset
from transformers import AutoTokenizer

MODEL = "sshleifer/tiny-gpt2"

# A minimal ChatML-ish template (tiny-gpt2 ships none). Real instruct models
# define their own tokenizer.chat_template; never hand-roll for those.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token

    # 1. language-modeling
    lm = Dataset.from_dict({"text": ["The capital of France is Paris."]})
    print("[language-modeling]", lm[0])

    # 2. prompt-completion (SFT can mask the prompt -> loss only on completion)
    pc = Dataset.from_dict({"prompt": ["Capital of France?"],
                            "completion": [" Paris."]})
    print("[prompt-completion]", pc[0])

    # 3. preference (for DPO/ORPO/KTO)
    pref = Dataset.from_dict({
        "prompt": ["Capital of France?"],
        "chosen": [" Paris."],
        "rejected": [" London."],
    })
    print("[preference]", pref[0])

    # 4. conversational -> render with a chat template
    tok.chat_template = CHAT_TEMPLATE
    convo = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]
    rendered = tok.apply_chat_template(convo, tokenize=False)
    print("\n[conversational] rendered training string:")
    print(rendered)

    # The generation prompt (what you feed at inference) stops before assistant.
    gen_prompt = tok.apply_chat_template(
        convo[:-1], tokenize=False, add_generation_prompt=True)
    print("[conversational] inference prompt:")
    print(gen_prompt)

    print("\nKey idea: TRL auto-detects the format from the columns present; "
          "for chat data the tokenizer's chat_template defines the exact text.")


if __name__ == "__main__":
    main()
