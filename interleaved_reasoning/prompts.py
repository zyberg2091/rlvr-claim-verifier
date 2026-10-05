"""Original prompt text and input construction."""

import re

EXAMPLES = """User: A very special island is inhabited only by wizards and goblins. Wizards always tell the truth, and goblins always lie. You meet 2 inhabitants: Quinn and Rowan. Quinn said that Rowan is a goblin. Rowan said "Quinn is a wizard and Rowan is a wizard". So who is a wizard and who is a goblin?. Assistant: <think>Suppose Quinn is a goblin. Then Quinn's statement is false, so Rowan is a wizard. A wizard tells the truth, but Rowan's statement says Quinn is a wizard, which would be false. That is a contradiction, so Quinn is a wizard.</think>
<answer>Quinn is a wizard
<sub_ans>Quinn: wizard</sub_ans></answer>
<think>Quinn is a wizard, so Quinn's statement is true, and Rowan is a goblin. Check: Rowan's statement says Rowan is a wizard, which is false, and a goblin always lies, so this fits.</think>
<answer>Rowan is a goblin
<sub_ans>Rowan: goblin</sub_ans></answer>
<think>So, the final answer is:</think>
<answer>Quinn is a wizard, and Rowan is a goblin.
<final_answer>{"Quinn": "wizard", "Rowan": "goblin"}</final_answer></answer>
User: A very special island is inhabited only by sailors and pirates. Sailors always tell the truth, and pirates always lie. You meet 3 inhabitants: Tessa, Milo, and Iris. Tessa said "Tessa is a sailor or Tessa is a pirate". Milo said that Tessa is a pirate. Iris said "Milo is a pirate if and only if Tessa is a sailor". So who is a sailor and who is a pirate?. Assistant: <think>Tessa says "Tessa is a sailor or Tessa is a pirate". Everyone is either a sailor or a pirate, so this statement is always true. A pirate never says something true, so Tessa is a sailor.</think>
<answer>Tessa is a sailor
<sub_ans>Tessa: sailor</sub_ans></answer>
<think>Milo says Tessa is a pirate. Tessa is a sailor, so this statement is false. A sailor never says something false, so Milo is a pirate.</think>
<answer>Milo is a pirate
<sub_ans>Milo: pirate</sub_ans></answer>
<think>Iris says "Milo is a pirate if and only if Tessa is a sailor". Milo is a pirate and Tessa is a sailor, so both sides are true and the statement is true. A pirate never says something true, so Iris is a sailor.</think>
<answer>Iris is a sailor
<sub_ans>Iris: sailor</sub_ans></answer>
<think>So, the final answer is:</think>
<answer>Tessa is a sailor, Milo is a pirate, and Iris is a sailor.
<final_answer>{"Tessa": "sailor", "Milo": "pirate", "Iris": "sailor"}</final_answer></answer>
"""


TABLE1 = """You are a helpful assistant. You reason through problems step by step before providing an answer. You
conduct your reasoning within <think></think> and share partial answers within <answer></answer>
as soon as you become confident about the intermediate results. You continue this pattern of
<think></think><answer></answer> until you reach the final answer. User: prompt. Assistant:
"""

SUB_ANS_SENTENCE = ("Write one <think></think> before every <answer></answer>, and never write two <answer></answer> blocks in a row. "
                    "Each <answer></answer> states one character's role in a short sentence, followed on the next line by the same role as a tag "
                    "inside the same <answer></answer>, using the names and role names from the puzzle. "
                    "For example, for a character named Tom who is not in this puzzle: "
                    "<think>Tom's statement is true, so Tom tells the truth.</think><answer>Tom is {a_truth}\n<sub_ans>Tom: {truth}</sub_ans></answer>")

FINAL_SENTENCE = ("The last <answer></answer> also comes after a <think></think>. It states the final answer, followed on the next line by "
                  "the full assignment of the puzzle's characters as a JSON object inside <final_answer></final_answer>.")

def build_template(table1):
    text = " ".join(table1.split())
    assert "PASTE TABLE 1" not in text, "Paste Table 1 of the paper into TABLE1 first."
    cut = text.find("User:")
    sentences = re.split(r"(?<=\.)\s+", text[:cut].strip())
    assert cut > 0 and len(sentences) == 4, "Table 1 should have 4 instruction sentences, then 'User:'."
    return " ".join(sentences[:3] + [SUB_ANS_SENTENCE, sentences[3], FINAL_SENTENCE]) + " " + EXAMPLES + \
           text[cut:].replace("prompt", "{question}", 1)

TEMPLATE = build_template(TABLE1)
TEMPERATURE = 0.8

def generate_input(problem, tokenizer):
    """
    Generate reasoning with SHORT think-answer pairs (1-2 sentences each)
    """

    truth, liar = re.search(r"inhabited only by (\w+) and (\w+)\.", problem).groups()
    truth, liar = truth[:-1], liar[:-1]
    a_truth = ("an " if truth[0] in "aeiou" else "a ") + truth
    prompt = (TEMPLATE.replace("{a_truth}", a_truth).replace("{truth}", truth).replace("{liar}", liar)
                      .replace("{question}", problem))

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    return enc["input_ids"]
