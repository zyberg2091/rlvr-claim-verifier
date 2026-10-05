"""PPO training and final adapter saving from the interleaved notebook."""

import os
import random
import re

import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from .prompts import TEMPERATURE, generate_input


def ppo_loss(ratio, advantage, value_pred, return_val, entropy):
  eps = 0.2
  c_v = 0.5
  c_h = 0.01

  l_policy = -torch.min((ratio * advantage), (torch.clamp(ratio, 1 - eps, 1 + eps) * advantage))
  l_policy = l_policy.mean() # Reduce policy loss to scalar

  value_t = value_pred
  # Ensure return_val has the same shape as value_t (torch.Size([1]))
  if return_val.dim() == 0: # if it's a scalar
      return_val = return_val.unsqueeze(0) # make it torch.Size([1])

  l_value = F.mse_loss(value_t, return_val)


  return l_policy + c_v * l_value - c_h * entropy


def train_ppo(policy_model, tokenizer, df, trajectories, paths):
    task_ids = []

    for task_id in list(trajectories.keys()):
      if task_id[0] not in task_ids:
        task_ids.append(task_id[0])

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    learning_rate = 2e-5
    # Use 8-bit optimizer to save memory
    optimizer = bnb.optim.AdamW8bit(policy_model.parameters(), lr=learning_rate, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01)

    epochs = 3
    k=4
    ACCUM_STEPS = 8

    #RESUME SETTINGS
    resume_from_checkpoint = paths.resume_from_checkpoint

    start_epoch = 0
    skip_tasks = 0

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):

        policy_model.lm.load_adapter(resume_from_checkpoint, "default")
        policy_model.value_head.load_state_dict(torch.load(os.path.join(resume_from_checkpoint, 'value_head.pt')))

        # Automatically determine epoch and skip_tasks from folder name
        match = re.search(r"epoch_(\d+)(?:_task_(\d+))?", resume_from_checkpoint)
        if match:
            start_epoch = int(match.group(1)) - 1
            skip_tasks = int(match.group(2)) if match.group(2) else 0

    for ppo_epoch in range(start_epoch, epochs):

      # Deterministic shuffle per epoch
      random.seed(42 + ppo_epoch)
      shuffled_ids = task_ids.copy()
      random.shuffle(shuffled_ids)

      # Skip already processed tasks for the resumed epoch
      tasks_to_process = shuffled_ids[skip_tasks:] if ppo_epoch == start_epoch else shuffled_ids
      processed = skip_tasks if ppo_epoch == start_epoch else 0

      optimizer.zero_grad()
      policy_model.train()

      for task_id in tasks_to_process:

        input_text = df[df['task_id'] == task_id]['quiz_text'].to_list()[0]

        if not any((task_id, j + 1) in trajectories for j in range(k)):
          continue

        for i in range(k): # optimizing for each rollout
          if (task_id, i + 1) not in trajectories:
            continue
          torch.cuda.empty_cache()
          initial_prompt_input_ids = generate_input(input_text, tokenizer)
          current_prompt_input_ids = initial_prompt_input_ids

          traj = trajectories[(task_id, i+1)]
          actions, log_prob_olds, advantages, returns  = traj['actions'], traj['log_probs_old'], traj['advantages'], traj['returns']

          # Reshape actions and related tensors from (gen_len, 1) to (1, gen_len)
          actions = actions.transpose(0, 1) # Shape becomes (1, gen_len)
          log_prob_olds = log_prob_olds.transpose(0, 1) # Shape becomes (1, gen_len)
          advantages = advantages.transpose(0, 1) # Shape becomes (1, gen_len)
          returns = returns.transpose(0, 1) # Shape becomes (1, gen_len)

          prompt_len = initial_prompt_input_ids.shape[1]
          gen_len = actions.shape[1] # now actions is (1, gen_len)

          full_sequence = torch.cat(
              [current_prompt_input_ids, actions],
              dim=1)

          logits, value_pred = policy_model(full_sequence, past_key_values=None, use_cache=False)

          # Slice logits aligned to generated tokens
          gen_logits = logits[:, prompt_len-1 : prompt_len+gen_len-1, :].contiguous()

          # Free up full logits immediately
          del logits

          # Calculate log_probs and entropy manually to avoid OOM from Categorical
          log_softmax_gen_probs = F.log_softmax(gen_logits / TEMPERATURE, dim=-1)   # (B, T_gen, V)

          # Free gen_logits as it's no longer needed
          del gen_logits

          log_prob_new = log_softmax_gen_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

          probs = torch.exp(log_softmax_gen_probs) # reuse log_softmax to get probs
          entropy = -(probs * log_softmax_gen_probs).sum(dim=-1).mean()

          # Free large probability tensors
          del probs, log_softmax_gen_probs

          ratio = torch.exp(log_prob_new - log_prob_olds)

          # Slice value_pred to match the generated tokens
          # Align values the same way
          value_pred_gen = value_pred[:, prompt_len-1 : prompt_len+gen_len-1]
          value_pred_gen = value_pred_gen.squeeze(-1)   # [1, T_gen]
          del value_pred

          loss = ppo_loss(ratio, advantages, value_pred_gen, returns, entropy)

          # Backpropagate immediately to free the computation graph for this rollout
          scaled_loss = loss / (k * ACCUM_STEPS)

          # Free temporary scalars/tensors before backward
          del loss, ratio, log_prob_new, value_pred_gen, full_sequence, current_prompt_input_ids
          torch.cuda.empty_cache()

          scaled_loss.backward()

        processed += 1

        if processed % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        if processed % 50 == 0:

            output_dir = os.path.join(paths.checkpoint_dir, f'ppo_checkpoint_epoch_{ppo_epoch+1}_task_{processed}')
            os.makedirs(output_dir, exist_ok=True)
            policy_model.lm.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(policy_model.value_head.state_dict(), os.path.join(output_dir, 'value_head.pt'))

      if processed % ACCUM_STEPS != 0:   # step on the gradients left over from the last tasks of the epoch
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

      output_dir = os.path.join(paths.checkpoint_dir, f'ppo_checkpoint_epoch_{ppo_epoch+1}')
      os.makedirs(output_dir, exist_ok=True)
      policy_model.lm.save_pretrained(output_dir)
      tokenizer.save_pretrained(output_dir)
      torch.save(policy_model.value_head.state_dict(), os.path.join(output_dir, 'value_head.pt'))

    return policy_model


def save_final_model(policy_model, tokenizer, paths):
    final_save_path = paths.final_model_dir
    os.makedirs(final_save_path, exist_ok=True)

    # Safely save only the trained LoRA adapter weights and tokenizer
    policy_model.lm.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
