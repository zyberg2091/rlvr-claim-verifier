# rlvr-claim-verifier

**Status:** in progress (October 2026). All work so far is on Knights-and-Knaves puzzles.

**One line:** checks whether each intermediate claim a model makes during RL with verifiable rewards is supported by the reasoning it wrote before that claim.

---

## 1. The problem

- This project uses **Knights-and-Knaves puzzles**. Each character either always tells the truth or always lies. The model must determine each character’s role from their statements.
- The model alternates reasoning in `<think>` blocks with intermediate answers in `<answer>` blocks. Each intermediate role claim has a tag such as `<sub_ans>Ella: deceiver</sub_ans>`.
- The intermediate reward, adapted from Xie et al. (2025), checks whether a claim matches the answer key. It is awarded only when the final answer is also correct.
- The reward does not check whether the reasoning written before the claim supports it.
- Among the **497 retained rollouts**, 27 assigned both roles to the same character at different steps. In **seven of these rollouts**, the correct claim received intermediate reward despite the conflicting claim elsewhere in the rollout.
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

- Generated **684 rollouts from 171 puzzles** using Qwen2.5-7B-Instruct. After format filtering, **497 rollouts remained, containing 1,186 tagged steps**.

- Checked the reward calculation. In seven rollouts containing conflicting role claims, the correct claim received intermediate reward.

  **Example from saved notebook output:** In `task_10000`, rollout 2, the model claims `elijah: guardian` and later `elijah: deceiver`; the latter receives an intermediate reward of 0.1667.

- Defined the engagement criterion and updated the parser prompt (`probe.py`).

- Tested the parser on difficult cases written with AI assistance.

- Processed **841 of the 1,186 steps** with the parser. Also checked 50 randomly selected steps by hand.

- Added a results notebook to check saved rollout and parser results (`check_kk_rollout_results.ipynb`).

**Next**

- Finish processing the remaining steps and report the parser’s agreement with hand-checked labels.
- Implement the axiom and enumeration layers, which currently have only hand-worked examples.
- Measure how often the engagement criterion misses valid reasoning because it does not explicitly call a statement true or false.
- Compare an LLM judge with the full three-layer verifier.
- **Later:** Test whether training with the verifier makes the reasoning more explicit or simply teaches the model to game its checks. Evaluate this through human review and tests that change parts of the reasoning, keeping these checks separate from those used during training. Repeat the comparison across several seeds.

## 5. Results so far

- **Reward behaviour:** In seven rollouts containing conflicting role claims, the correct claim still received intermediate reward.

- **Answer text alone:** Saved answer-only labels marked engagement in **3 of 1,186 steps**. This is a separate check from the unfinished full-step parser run. This measures explicit engagement under our definition, not whether the answer text contains any valid reasoning.

- **Parser tests:** The parser matched **187 of 188 expected character labels**: 31/31 gold labels, 131/132 stress labels, and 25/25 labels from cases resembling real rollouts. Three labels flagged for review were excluded from this score. These cases were written with AI assistance. The results show performance on these constructed tests; they do not establish accuracy on real rollouts.

- **Preliminary rollout results:** Model-proposed labels marked at least one character as engaged in **190 of 200 randomly sampled reasoning steps**. Fifty steps have also been checked by hand, but the agreement from those checks is still to be reported.

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

- The reported model runs were carried out in the original notebooks. The Python scripts reorganize that code. Python syntax and both command-line help interfaces have been checked, but the scripts have not been run end to end on a GPU in this review.
- The results notebook includes cached outputs. Many reporting statements have been removed, so rerunning cells does not display all of those summaries. It also does not recompute the 841-step progress count or agreement for the 50 hand-reviewed steps.
- The parser skips IDs already present in `probe_output`, including failed or truncated responses. Use a new output filename when changing the prompt or deliberately rerunning a set.
- Training resume restores adapter and value-head weights, but not optimizer or random state. Resuming an end-of-epoch checkpoint repeats that epoch. Periodic checkpoints include the value head; the final export contains the adapter and tokenizer.

## 7. Differences from Xie et al. (2025)

- **Rollouts per puzzle:** We generate 4; the paper uses 8.
- **Generation limit:** We use 600 tokens; the paper uses 2,548.
- **Training settings:** We use no KL penalty or reference model. The learning rate is `2e-5`, compared with `1e-6` in the paper.
- **Rollout collection:** Training reuses saved rollouts; it does not collect fresh rollouts between PPO updates.
- **Intermediate reward:** The base reward is 0.5. We do not apply the paper’s batch-accuracy condition.
- **Puzzle filtering:** We exclude a fixed list of 25 puzzles identified as having more than one valid answer.
- **Answer format:** Each intermediate answer includes a `Name: role` tag, and the prompt contains two worked examples. The reported analysis uses a separately filtered set of rollouts. During collection, outputs missing required tags are retried; additional format violations can receive a negative format reward and still be saved.

These differences need to be considered when comparing our results with the paper.

## 8. References

1. Roy Xie et al. (2025). [Interleaved Reasoning for Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2505.19640). Source for the interleaved format and conditional intermediate reward.
2. John Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Reference for the PPO training method.
