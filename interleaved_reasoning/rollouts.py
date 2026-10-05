"""Rollout generation and rewards from the interleaved reasoning notebook."""

import json
import logging
import os
import pickle
import re
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import StoppingCriteria

from .prompts import TEMPERATURE, generate_input


class StopAtFinalAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        return "</final_answer>" in decoded

def extract_format(generated_subanswers, name_role_subanswers_gen):
    answers = [a.strip() for a in generated_subanswers]
    if len(answers) < 2:
        return -1

    *steps, last = answers

    for a in steps:
        tags = re.findall(r"<sub_ans>(.*?)</sub_ans>", a, re.S)
        sentence = re.sub(r"<sub_ans>.*?</sub_ans>", "", a, flags=re.S).strip()
        if len(tags) != 1 or not re.fullmatch(r"\s*\w+\s*:\s*\w+\s*", tags[0]) or not sentence:
            return -1   # each step answer needs a sentence and exactly one tag holding one "Name: role" pair
    if "<sub_ans>" in last or not re.search(r"<final_answer>.*</final_answer>\s*$", last, re.S):
        return -1       # the last answer ends with the final answer and holds no tag

    if len(name_role_subanswers_gen) != len(steps):
        return -1
    return 1


def extract_final(final_generated_answer, gt_final_pair):

  try:
    final_generated_answer = json.loads(final_generated_answer)

  except (json.JSONDecodeError, TypeError):
    return False

  # must be a non-empty dict of str -> str; anything else is a malformed answer, not a crash
  if not (isinstance(final_generated_answer, dict) and final_generated_answer
          and all(isinstance(k, str) and isinstance(v, str) for k, v in final_generated_answer.items())):
    return False

  print('gt_final_pair_new: ', gt_final_pair)

  print('final_generated_answer: ', final_generated_answer)


  try:
    gt_final_pair = json.loads(gt_final_pair)  # last ROW, all entities

  except (json.JSONDecodeError, TypeError, IndexError):
      return False

  if not gt_final_pair or not final_generated_answer:
      return False

  gt_final = {pair["name"].lower() : pair["role"].lower() for pair in list(gt_final_pair)}

  gen_final = {key.lower(): value.lower() for key, value in final_generated_answer.items()}

  print(f'ground truth final naswer: {gt_final} and generated final answer {gen_final}')


  gt_keys = list(sorted(gt_final.keys()))
  gen_keys = list(sorted(gen_final.keys()))



  if gt_keys == gen_keys:
    for key in gt_keys:
      if gen_final[key] != gt_final[key]:
        return False

  else:
    return False

  return True

def norm(s):
    name, _, role = s.partition(':')
    return f"{name.strip().lower()}:{role.strip().lower()}"

def conditional_reward_system(gt_clean_subanswers, generated_subanswers, name_role_subanswers_gen, name_role_subanswers_gt,
                              final_generated_answer, final_answer, gt_final_pair, sub_ans_token_positions, final_ans_index,
                              Acc_curr, Acc_prev, mode, T, gamma):

    rewards = torch.zeros(T, device=device)

    R_base = 0.5
    epsilon = 0.05

    matched = []  # sub_ans -> step index


    r_format = extract_format(generated_subanswers, name_role_subanswers_gen)


    if r_format != 1:
        r_final = 0                      # paper: final reward only when the format is correct
    elif extract_final(final_generated_answer, gt_final_pair):
        r_final = 2
    else:
        try:
            json.loads(final_generated_answer)
            r_final = -1.5               # paper: wrong answer
        except (json.JSONDecodeError, TypeError):
            r_final = -2                 # paper: answer cannot be parsed

    bool_format = 1 if r_format==1 else 0

    bool_final = 1 if r_final>0 else 0


    matches=0

    len_subans_gt = len(name_role_subanswers_gt)



    if len_subans_gt == 0:
        # place format + final first, unconditionally
        rewards[final_ans_index] += r_final
        rewards[T - 1]           += r_format

        return rewards

    # Parse JSON strings and dedupe to unique claims
    parsed_gt = set()
    for pair in name_role_subanswers_gt:
        for d in json.loads(pair):
            parsed_gt.add(f"{d['name'].strip().lower()}:{d['role'].strip().lower()}")

    gt_pairs       = list(parsed_gt)             # unique claims, lowercased
    gen_pairs      = [norm(p) for p in name_role_subanswers_gen]
    len_subans_gt  = len(gt_pairs)


    for k, (gen_pair, t_pos) in enumerate(zip(gen_pairs, sub_ans_token_positions), start=1):
        if gen_pair in gt_pairs:
            gt_pairs.remove(gen_pair)
            matches += 1
            matched.append((k, t_pos))





    if mode == 'training_phase':
      bool_accuracy = 1 if (Acc_curr > Acc_prev - epsilon) else 0

    else:
      bool_accuracy = 1

    gate = bool_format * bool_final * bool_accuracy


    ### calc reward (time discounted)


    matched_pos = []
    matched_int_rewards = []

    for k, t_pos in matched:
      if matches==len_subans_gt:
        r_intermediate = gate * R_base / matches
      else:
        r_intermediate  = gate * R_base * (1.0 / k) / len_subans_gt
      rewards[t_pos]+= r_intermediate
      matched_pos.append(t_pos)
      matched_int_rewards.append(r_intermediate)



    # (2) final-answer reward at the </final_answer> token

    final_ans_token_pos = final_ans_index
    rewards[final_ans_token_pos] += r_final


    rewards[T-1]+= r_format

    print(f'rewards for the task in rollout {i+1}: ',  rewards)

    print(f'indexes to watch for format, final_nas_token_pos and matched_pos in rewards: {T-1, final_ans_token_pos, matched_pos}')

    print(f'r_final: {r_final}, r_format: {r_format}, matched_intermediate_rewards: {matched_int_rewards}')

    print('\n')

    return rewards


def collect_rollouts(policy_model, tokenizer, df, final_df, paths):
    global i

    trajectories_file = paths.trajectories_file

    if os.path.exists(trajectories_file):
        with open(trajectories_file, 'rb') as f:
            trajectories = pickle.load(f)

    else:
        trajectories = {}

    # Suppress bitsandbytes warnings
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="bitsandbytes")

    policy_model.eval()   # in train mode, gradient checkpointing switches off the cache this loop relies on
    gamma = 0.999
    k=4

    # Extract already processed task IDs for fast O(1) lookup
    processed_task_ids = {k[0] for k in trajectories.keys()}

    task_id_list, prompts_list, final_answers_list_gt, row_claims = final_df['task_id'].to_list(), final_df['quiz_text'].to_list(), final_df['final_answer'].to_list(), final_df['row_claims'].to_list()

    gt_final_answers = [claims[-1] for claims in row_claims]

    for task_id, input_text, final_answer_gt, gt_final_pair in zip(task_id_list, prompts_list, final_answers_list_gt, gt_final_answers):

      # Skip if already processed
      if task_id in processed_task_ids:

        continue

      MAX_RETRIES = 4

      match_val = re.search(r"<final>(.*?)</final>", final_answer_gt, re.S)
      final_answer_gt = match_val.group(1).strip() if match_val is not None else ''

      for i in range(k):

        gt_subanswers = df[df['task_id']==task_id]['sub_answer'].dropna().to_list()
        name_role_subanswers_gt = df[df['task_id']==task_id]['row_claims'].dropna().to_list()

        all_retry = True

        for retry in range(1, MAX_RETRIES+1):
          # Generate initial prompt input_ids once per retry
          initial_prompt_input_ids = generate_input(input_text, tokenizer)

          actions, log_probs_old, values = [], [], []
          past_key_values = None # Initialize past_key_values for efficient generation
          current_model_input_ids = initial_prompt_input_ids # First input to the model will be the full prompt

          if retry == MAX_RETRIES:

            all_retry = False
            break

          # Loop for generating tokens - INCREASED LIMIT HERE
          for _ in range(600):
            with torch.no_grad():
                # Pass current_model_input_ids and past_key_values to the model
                logits, value, past_key_values = policy_model(current_model_input_ids, past_key_values, True)

            # Logits for the *last* token in the current_model_input_ids
            probs = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)

            dist = torch.distributions.Categorical(probs)

            action = dist.sample()                           # sampling a token_id

            log_prob = dist.log_prob(action)

            actions.append(action) # Collect the generated token
            log_probs_old.append(log_prob.detach()) # Store log_prob of the sampled action
            values.append(value.detach()) # Store value prediction for the state before action

            # For the next step, the input to the model is only the newly generated token
            current_model_input_ids = action.unsqueeze(0) # Prepare for the next iteration

            # Check for final answer token to stop generation early
            # We need to concatenate initial_prompt_input_ids with all generated actions for the stopping criteria
            # This temporary tensor is only for checking, not for model input
            temp_full_sequence_ids = torch.cat([initial_prompt_input_ids.squeeze(0), torch.cat(actions)], dim=0).unsqueeze(0)
            if StopAtFinalAnswer(tokenizer)(temp_full_sequence_ids, None):
                break

          # Decode all collected actions to get the full generated text
          generated_text = tokenizer.decode(
              torch.cat(actions),
              skip_special_tokens= True
          )

          end = generated_text.rfind("</final_answer>")
          start = generated_text.rfind("<final_answer>", 0, max(end, 0))
          if end != -1 and start != -1:
              generated_text = generated_text[:end + len("</final_answer>")]   # drop anything after the tag, like the "</" of a "></" token
              before = generated_text[:start]
              if before.count("<answer>") == before.count("</answer>"):   # final answer written outside any <answer>
                  generated_text = before + "<answer>" + generated_text[start:]
              if generated_text.count("<answer>") > generated_text.count("</answer>"):
                  generated_text = generated_text + "</answer>"

          generated_subanswers = [a for a in re.findall(r"<answer>(.*?)</answer>", generated_text, re.S) if a.strip()]
          name_role_subanswers_generated = re.findall(r"<sub_ans>(.*?)</sub_ans>", generated_text, re.S)

          # Clean the ground truth subanswers to extract text from tags if they exist
          gt_clean_subanswers = []
          for sa in gt_subanswers:
              if not pd.isna(sa):
                  match = re.search(r"<answer>(.*?)</answer>", str(sa), re.S)
                  gt_clean_subanswers.append(match.group(1).strip() if match else str(sa).strip())

          # Fallback if ground truth is empty or missing
          if not gt_clean_subanswers:
              gt_clean_subanswers = [''] * len(generated_subanswers)

          match_val = re.search(r"<final_answer>(.*?)</final_answer>", generated_text, re.S)
          final_generated_answer = match_val.group(1).strip() if match_val is not None else ''

          if '<think>' not in generated_text or '</think>' not in generated_text:

            continue

          if '<answer>' not in generated_text or '</answer>' not in generated_text:

            continue

          if '<final_answer>' not in generated_text or '</final_answer>' not in generated_text:

            continue

          if '<sub_ans>' not in generated_text or '</sub_ans>' not in generated_text:

            continue

          action_ids = [a.item() for a in actions]
          sub_ans_indexes = []
          final_ans_index = []

          for m in range(len(action_ids)):
            text_so_far = tokenizer.decode(action_ids[:m+1], skip_special_tokens=True)
            prev_text = tokenizer.decode(action_ids[:m], skip_special_tokens=True) if m > 0 else ""

            # Use count() to robustly check if the tag was completed by token m
            if text_so_far.count("</sub_ans>") > prev_text.count("</sub_ans>"):
              sub_ans_indexes.append(m)

            if text_so_far.count("</final_answer>") > prev_text.count("</final_answer>"):
              final_ans_index.append(m)

          if len(sub_ans_indexes) != len(name_role_subanswers_generated):

            continue

          if len(final_ans_index) != 1:

            continue

          break

        if all_retry:

            T = len(values)
            # calculation reward/return for each task for advantage estimation i.e per trajectory
            rewards = conditional_reward_system(gt_clean_subanswers, generated_subanswers, name_role_subanswers_generated, name_role_subanswers_gt,
                                               final_generated_answer, final_answer_gt, gt_final_pair, sub_ans_indexes, final_ans_index[0],
                                               None, None, "no_accuracy_gate", T, gamma)

            # Corrected: Initialize returns as (T, 1) to match values shape
            returns = torch.zeros(T, 1).to(device) # Move returns to the same device as values
            returns[T - 1] = rewards[T - 1]

            for t in reversed(range(T - 1)):
                returns[t] = rewards[t] + gamma * returns[t + 1]

            values  = torch.stack(values)

            advantages = (returns - values).detach()

            # pick the first task, do one rollout, verify shapes and values
            assert len(action_ids) == T
            assert len(sub_ans_indexes) == len(name_role_subanswers_generated)
            assert 0 <= final_ans_index[0] < T
            assert all(0 <= p < T for p in sub_ans_indexes)
            assert all(isinstance(p, str) for p in name_role_subanswers_gt)   # catches NaN
            assert -3.5 <= rewards.sum().item() <= 4.5
            assert returns[0].abs().item() > 0  # confirms terminal reward propagated

            trajectories[(task_id, i+1)] = {
                "actions": torch.stack(actions),
                "log_probs_old": torch.stack(log_probs_old),
                "advantages": advantages,
                "returns": returns,
                'rewards': rewards,
                'generated_subanswers': generated_subanswers,
                'name_role_subanswers_generated': name_role_subanswers_generated,
                'final_generated_answer': final_generated_answer,
                'sub_ans_token_positions': sub_ans_indexes,
                'final_ans_index': final_ans_index,
                'T': T,
                'generated_text': generated_text,
            }

        # in the failure branch:
        else:

            continue

      with open(trajectories_file, 'wb') as f:   # save after every task, so a disconnect loses at most one task
        pickle.dump(trajectories, f)

    with open(trajectories_file, 'wb') as f:
        pickle.dump(trajectories, f)

    with open(paths.rollouts_jsonl, "w") as f:
        for (task_id, r), tr in trajectories.items():
            f.write(json.dumps({"task_id": task_id, "rollout": r, "text": tr["generated_text"]}) + "\n")

    return trajectories
