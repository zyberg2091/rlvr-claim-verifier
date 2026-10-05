"""Reasoning probe from the original notebook, with configurable file paths."""

import argparse
import json
import os
import re
from pathlib import Path

# Illustrations use throwaway names (Sam/Alex) that appear in NO puzzle, so the
# fixed prompt can never prime the model on an actual gold block.
SYSTEM_PROMPT = """Knights and Knaves. Each character makes one statement.
Characters with the role "<<TRUTH_ROLE>>" always speak the truth. Characters with the role "<<LIAR_ROLE>>" always lie.

You are given the names of the characters and one block of someone's reasoning about the puzzle.
You are not given the puzzle's statements. Use only the text of this block.
Do not try to solve the puzzle. Never check whether the block is right.

For EACH character listed, answer one question:

Does the block do BOTH of these for that character?
  (A) say that THAT character's OWN statement is true or false
  (B) connect that truth claim to something -- derive it from something the block states,
      or draw something from it

About (A)
- Any plain wording that says the statement is true or false counts, for example "accurate",
  "mistaken", "honest", "dishonest", and negated forms such as "is not a lie".
- Saying the statement itself is contradictory, or would contradict itself, counts as
  calling it false, for example "Alex's statement is contradictory" or "what Alex said
  would contradict itself". A contradiction said of a situation is NOT a truth value for
  any statement, for example "that leads to a contradiction" or "this cannot happen".
- A truth value inside an assumption counts, for example "assume what he said is accurate".
- It must be about the statement as a unit. If the block itself says the truth value is for
  only one part of the statement, it does not count.
- It must be about that character's OWN statement.
  In "Because Sam is a saint, what Alex said must be false", Sam gets a ROLE and ALEX gets a
  truth value. Only Alex has (A). Sam does not.
- The character may be pointed to by name, by a pronoun whose owner is clear in the block,
  or as part of a group whose members are clear in the block. If the block does not make
  clear whose statement it means, nobody has (A).
- Go by what the block says about who said what. Do not correct it.
- Naming or repeating what a character said, without a truth value for it, is not (A).

About (B)
- What the truth claim is connected to does not matter. It may be that character's own role,
  another character's role, or any other conclusion. The character's own role does NOT have
  to appear.
- Connectives = so, therefore, thus, hence, because, since, means, makes, which, that, if,
  suppose, then. A dash or a comma also counts when one part is given as the reason for the other.
- NOT connectives = separately, also, meanwhile, anyway.
- The connective must link the truth claim itself to something. A connective somewhere else
  in the block does not count.
- Claims that are only listed side by side, with commas or "and", are not connected.
- A truth claim with nothing connected before or after it gives false.
- A truth claim that comes only from that character's own role, as in "Sam is a saint, so his
  statement is true", is not connected when nothing follows from it. If the block draws
  anything from it, even inside an assumption, as in "If Sam is a saint, he tells the truth,
  so Alex is a devil", it is connected.ough.

Wrong reasoning still gives true. A correct answer with no truth claim still gives false.

Reply with one JSON object: one key per listed character, using the names exactly as listed,
value true or false. Nothing else.

The examples below use the roles saint and devil. Your puzzle may use other role names.

Example 1
Characters: Sam, Alex, Jordan
Block: We already found that Sam is a saint. That makes what Alex said true, and only a saint says true things, so Alex is a saint.
{"Sam": false, "Alex": true, "Jordan": false}
Sam has a role but no truth value about what Sam said.

Example 2
Characters: Sam, Alex, Jordan
Block: Since Jordan's claim is mistaken, neither Alex nor Sam can be a devil.
{"Sam": false, "Alex": false, "Jordan": true}
Jordan is true even though the block never gives Jordan's role.

Example 3
Characters: Sam, Alex, Jordan
Block: Alex talks about Sam. Sam is a devil. Meanwhile, Jordan is being honest.
{"Sam": false, "Alex": false, "Jordan": false}
Jordan has a truth value, but nothing is connected to it.

Example 4
Characters: Sam, Alex, Jordan
Block: Since Sam is a saint, his statement is true. Alex's statement is false, so Alex is a devil.
{"Sam": false, "Alex": true, "Jordan": false}
Sam's truth value only repeats the rule for saints, and nothing follows from it."""


USER_TEMPLATE = """Characters: {names}

Block:
{block}"""


def render_user(names, block):
    lines = ", ".join(names)
    return USER_TEMPLATE.format(names=lines, block=block)


def build_messages(characters, block):
    """Chat-format messages for transformers (system goes inside the list)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user(characters, block)},
    ]


JSON_OBJ = re.compile(r'\{[^{}]*\}')


def extract(reply, expected_keys):
    want = set(expected_keys)
    for m in JSON_OBJ.finditer(reply):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and set(obj) == want and all(isinstance(v, bool) for v in obj.values()):
            return obj
    return None


def run_probe(paths):
    import torch
    from transformers import pipeline

    # Initialize the pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3.6-27B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    with open(paths.probe_input, 'r') as f:
      # Use json.load() to read directly from the file object
      file = json.load(f)

    Path(paths.probe_output).parent.mkdir(parents=True, exist_ok=True)

    OUT_PATH  = paths.probe_output
    PROMPT    = SYSTEM_PROMPT
    RERUN_IDS = None
    results = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else []
    done = {r["id"] for r in results}

    for puzzle in file['puzzles']:
      names = list(puzzle['statements'].keys())
      roles = [puzzle['roles']['truth'], puzzle['roles']['liar']]

      for block in puzzle['examples']:

        if block["id"] in done: continue

        if RERUN_IDS is not None and block["id"] not in RERUN_IDS: continue

        blk = block['text']

        messages = build_messages(names, blk)
        messages[0]["content"] = PROMPT
        messages[0]["content"] = messages[0]["content"].replace("<<TRUTH_ROLE>>", roles[0]).replace("<<LIAR_ROLE>>", roles[1])

        assert "<<" not in messages[0]["content"]

        THINKING = True   # flip for the other arm

        chat_prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=THINKING,
        )

        out = pipe(chat_prompt,
              max_new_tokens=4000 if THINKING else 300,
              do_sample=False)

        reply = out[0]["generated_text"][len(chat_prompt):]
        raw = reply

        truncated = THINKING and "</think>" not in reply

        if "</think>" in reply:
            reply = reply.rsplit("</think>", 1)[1].strip()

        answer = None if truncated else extract(reply, names)

        results.append({"id": block["id"], "answer": answer,
                        "truncated": truncated, "raw": raw})
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=1)

    return results


def main():
    from paths import load_paths, mount_colab, disconnect_colab

    parser = argparse.ArgumentParser(description="Run the original reasoning probe.")
    parser.add_argument("--paths", default=None, help="Path to a JSON file with input and output paths.")
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive in Colab before running.")
    parser.add_argument("--disconnect", action="store_true", help="Disconnect the Colab runtime after a successful run.")
    args = parser.parse_args()
    if args.mount_drive:
        mount_colab()
    paths = load_paths(args.paths)
    if not Path(paths.probe_input).is_file():
        raise FileNotFoundError(f"Probe input not found: {paths.probe_input}. Set probe_input in the path configuration.")
    run_probe(paths)
    if args.disconnect:
        disconnect_colab()


if __name__ == "__main__":
    main()
