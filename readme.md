# rlvr-claim-verifier

**Status:** in progress (October 2026). All work so far is on Knights-and-Knaves puzzles.

**One line:** checks whether each intermediate claim a model makes during RL with verifiable rewards is supported by the reasoning it wrote before that claim.

---

## Starting point: Xie et al. (2025)

- **Taken from the paper:** the interleaved format, where `<think>` reasoning alternates with intermediate answers in `<answer>` tags, and the conditional intermediate reward. An intermediate answer earns a reward for an exact match only if the format is valid and the final answer is correct.
- **Changed:**
  - Each intermediate answer includes a `Name: role` tag. The reward checks this tag instead of matching the whole answer text.
  - The paper's batch-accuracy condition is implemented in the code, but it was disabled for all reward calculations.
  - Training is smaller, with 4 rollouts per puzzle and a 600-token limit. There is no KL penalty, and saved rollouts are reused.
- **Unchanged:** as in the paper, the reward checks only the answer, not the reasoning before it.
- The full comparison is in Section 7.
  
## 1. The problem

- This project uses **Knights-and-Knaves puzzles**. Each character either always tells the truth or always lies. The model must determine each character’s role from their statements.
- The model alternates reasoning in `<think>` blocks with intermediate answers in `<answer>` blocks. Each intermediate role claim has a tag such as `<sub_ans>Ella: deceiver</sub_ans>`.
- The reward does not check whether the reasoning written before the claim supports it. 
- **Example:** In `task_10000` (rollout 2, step 2), the claim `Elijah: deceiver` is correct and receives an intermediate reward of 0.1667. But the step reaches this answer by assuming that "Ella cannot be a deceiver", even though Ella is a deceiver in the solution.
- We want to investigate whether the reasoning written before a rewarded claim actually justifies it.

## 2. Why it needs solving

- A correct claim and a justified claim are different. Matching the answer key does not establish that the preceding reasoning supports the claim.
- To reward checkable reasoning, we need to distinguish claims supported by the written reasoning from correct claims that lack that support.
- The main goal is to make this distinction explicit. We will also measure how filtering intermediate rewards affects final-answer accuracy.

## 3. The design

The verifier has three layers:

- **Engagement layer: built.** An LLM parser reads each step. For each character, it checks whether the reasoning calls that character’s own statement true or false and connects that judgment to another part of the argument. It returns one engagement label per character. The parser does not receive the puzzle or answer key.
- **Axiom layer: designed.** These parts of the reasoning are converted into logical rules using only what the text explicitly says. The answer key is not used to construct the rules.
- **Enumeration layer: designed.** Every possible combination of roles is checked against the rules collected before the claim. Combinations that break any rule are removed.

**What the results mean**

- **Inconsistent rules: the rules contradict each other.** There is no way to assign everyone a role without breaking a rule. These rules cannot be used to support a claim.
- **Supported: the rules establish the claimed role.** The rules do not contradict each other, and the character must have the claimed role to satisfy them.
- **Insufficient support: the rules do not establish the claimed role.** They still allow the character to have the opposite role.

We also compare the rules with the known solution:

- **Conflict with the solution: a rule disagrees with the correct answer.** The error could come from the model’s reasoning or from how we extracted the rule.

**Example**

Suppose the correct solution is **Alice = liar, Bob = truth-teller**, and the model claims **“Bob is a truth-teller.”**

Each row below shows a different set of extracted rules:

| Extracted rules | Remaining possibilities | Result |
|---|---|---|
| Alice is a liar; they cannot both be liars. | Alice is a liar and Bob is a truth-teller. | **Supported:** Bob must have the claimed role. |
| Alice is a liar; nothing else is established. | Bob could have either role. | **Insufficient support:** Bob’s role is still undecided. |
| Bob is a liar; Bob is a truth-teller. | None: Bob cannot have both roles. | **Inconsistent rules.** |
| Alice is a truth-teller; Bob is a truth-teller. | Both are truth-tellers. | **Conflict with the solution:** the rule about Alice is wrong. |

The last case shows why the solution check is separate: the rules support the claim about Bob, but contain an error about Alice.

**Design boundaries**

- The claim being checked must not be used as evidence for itself.
- The engagement criterion requires an explicit judgment about a statement being true or false. It may miss valid reasoning expressed only through roles.
- Words such as “because” or “so” do not by themselves establish a valid argument.
- The `Name: role` tag gives the reward function and verifier a specific claim to check without requiring an exact match on the whole answer sentence.

## 4. Progress and next steps

**Done**

- Built the PPO + LoRA training pipeline with an adapted version of the paper’s format and reward (`train.py`, `interleaved_reasoning/`).

- We used Qwen2.5-7B-Instruct to generate **684 rollouts from 171 puzzles** before any RL training. After filtering for format, **497 rollouts remained, containing 1,186 tagged steps**.

- Xie et al. match the whole answer text. We check the `Name: role` tag that the prompt asks the model to include in each answer. This tag is compared with the puzzle's correct name-role pairs. As in their paper, a claim is rewarded only if it matches the correct answer. The reasoning before the claim is not checked.

- Defined the engagement criterion and updated the parser prompt (`probe.py`).

- Tested the parser on difficult cases written with AI assistance. It got 187 of 188 labels right, with three disputed labels settled by hand. Three boundary cases were left out because their labels were still undecided.

- Ran the parser on **841 of the 1,186 steps**. Checked 58 steps by hand: 50 picked at random and 8 where the parser disagreed with the proposed labels.
 
- Added a results notebook to check saved rollout and parser results (`check_kk_rollout_results.ipynb`).

**Next**

- Finish processing the remaining steps with the parser.
- Implement the axiom and enumeration layers, which currently have only hand-worked examples.
- Measure how often the engagement criterion misses valid reasoning because it does not explicitly call a statement true or false.
- Compare an LLM judge with the full three-layer verifier.
- **Later:** Test whether training with the verifier makes the reasoning more explicit or simply teaches the model to game its checks. Evaluate this through human review and tests that change parts of the reasoning, keeping these checks separate from those used during training. Repeat the comparison across several seeds.
- Extend the method beyond Knights-and-Knaves to tasks where reasoning steps can be represented as constraints, including math problems, planning, scheduling and other logic tasks. For each domain, the main challenge is defining which parts of the reasoning can be converted into constraints and checked with the verifier.

## 5. Results so far

- **Answer text alone:** Only 3 of the 1,186 answer sentences contain truth words such as "true", "false" or "lying". These are also the only 3 marked as engaged. The verifier reads the think text to assess engagement in the reasoning behind the answers.

- **Parser tests:** The parser matched **187 of 188 expected character labels**: 31/31 gold labels, 131/132 stress labels, and 25/25 labels from cases resembling real rollouts. Three disputed labels were settled by hand. Three boundary labels remained undecided and were left out of the score. The cases were written with AI assistance to test how well the parser follows the rule on difficult examples. This score does not measure accuracy on real rollouts.

- **Preliminary rollout results:** At least one character is marked as engaged in **189 of 200 randomly sampled reasoning steps**. These labels were proposed by a model, with 58 steps checked by hand. On the 50 randomly chosen steps checked by hand, the parser agrees with 113 of 118 reviewed character labels: **95.8%** (95% interval: 90.5% to 98.2%).

- **Known limitation:** Valid reasoning expressed only through roles may fail the engagement criterion because it does not explicitly call a statement true or false. Failing this criterion does not automatically mean the reasoning is wrong.

## 6. Code

| File | Purpose |
| --- | --- |
| `train.py` | Entry point: load data and model, collect rollouts, train, and save. |
| `interleaved_reasoning/` | Model, tokenizer, LoRA and value head (`model.py`), data loading (`data.py`), prompts (`prompts.py`), rewards and rollouts (`rollouts.py`), PPO (`ppo.py`), and generation (`generation.py`). |
| `probe.py` | Labels whether each character is engaged in a supplied reasoning block and saves the parser responses. |
| `check_kk_rollout_results.ipynb` | Checks saved rollout and parser results; requires a separate results bundle (written with AI help). |
| `paths.json`, `paths.py` | Input, output and checkpoint locations. |
| `original_notebooks/` | Original training, probe and results notebooks. |

**Run**

```bash
python -m pip install -r requirements.txt
python train.py --stage rollouts   # collect rollouts
python train.py --stage train      # PPO training on saved rollouts
python probe.py                    # run the parser on its configured input
```

Edit `paths.json` first, or copy it to `paths.local.json` and pass `--paths paths.local.json` to either script. Relative paths are resolved from the configuration file's directory.

Data, saved results and model weights are not included. Model runs need a compatible GPU setup. Training uses Qwen2.5-7B-Instruct with quantization; the parser currently loads Qwen3.6-27B. Dependency versions are not pinned, so use the working notebook environment where available.

**Required inputs**

- Set `dataset_csv` to a CSV containing `task_id`, `quiz_text`, `reasoning_step`, `sub_answer`, `row_claims`, and `final_answer`, using the original puzzle and claim formats.
- Set `probe_input` to the prepared probe JSON. It must contain `puzzles`, with character names in `statements`, role names in `roles.truth` and `roles.liar`, and `examples` containing `id` and `text`. The default input is the constructed test set. The script does not automatically convert saved rollouts into this format.
- For the results notebook, set `verifier_data_dir` in `paths.json` to the separate results bundle containing `data/`, `tests/`, `parser_outputs/`, and `labels_and_reviews/`. Run the notebook from the repository root so it can import `paths.py`.

**Validation and current limitations**

- The reported model runs were carried out in the original notebooks. The Python scripts reorganize that code. Python syntax and both command-line help interfaces have been checked, but the scripts have not been run end to end on a GPU.
- Rerunning the results notebook does not reproduce all the summaries in its cached outputs. It also does not yet recalculate the 841-step count or the 95.8% agreement with labels checked by hand.
- The parser skips IDs already present in `probe_output`, including failed or truncated responses. Use a new output filename when changing the prompt or deliberately rerunning a set.
- Training resume restores adapter and value-head weights, but not optimizer or random state. Resuming an end-of-epoch checkpoint repeats that epoch. Periodic checkpoints include the value head; the final export contains the adapter and tokenizer.

## 7. What we took from Xie et al. (2025) and what we changed

| Part of the setup | Xie et al. (2025) | This project |
|---|---|---|
| Reasoning format | Alternating `<think>` and `<answer>` blocks | Same |
| Intermediate answers | Short answer text | Short answer text with a `Name: role` tag; the prompt includes two worked examples |
| How intermediate answers are checked | The answer text must exactly match the expected answer | The tag must exactly match a correct name-role pair |
| What the check reads | Only the answer, not the reasoning | Same |
| Conditions for intermediate reward | Valid format, a correct final answer, and improved batch accuracy | Valid format and a correct final answer; the batch-accuracy condition is implemented but disabled |
| Reward schedule | Earlier correct answers receive more reward through time discounting | Also time-discounted, with a base reward of 0.5 |
| Rollouts per puzzle | 8 | 4 |
| Generation limit | 2,548 tokens | 600 tokens |
| KL penalty and reference model | Used | Not used |
| Learning rate | 1e-6 | 2e-5 |
| Rollout collection | Fresh rollouts for each update | Saved rollouts are reused |
| Format enforcement | Format reward | Format reward, retries for missing tags, and filtering after generation |
| Puzzles | Knights-and-Knaves puzzles | 171 Knights-and-Knaves puzzles; 25 with more than one valid answer excluded |

Any comparison with the paper's results needs to account for these differences.

## 8. References

1. Roy Xie et al. (2025). [Interleaved Reasoning for Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2505.19640). Source for the interleaved format and conditional intermediate reward.
2. John Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Reference for the PPO training method.
