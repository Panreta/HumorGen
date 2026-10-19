Goal: Supervised Fine-Tuning the LLM with RL method to understand humor.

Paper refering:
1. HumorGen: Cognitive Synergy for Humor Generation in Large Language Models via Persona-Based Distillation
2. INNOVATIVE THINKING, INFINITE HUMOR: HUMOR RESEARCH OF LARGE LANGUAGE MODELS THROUGH STRUCTURED THOUGHT LEAPS

HumorGen from the 1st one, and LoL from the 2nd.
**HumorGen** has the better way to _create training jokes_: six persona prompts plus Elo ranking.
(psychological measure) **LoL** has the better way to _keep improving after SFT_: teach the model to judge humor, then use that judge for online DPO and best-of-n selection.

## The pipeline

Model: Qwen2.5-1.5B or 7B with LoRA, and about 1,000 prompts. 
Dataset: SemEval 2026 MWAHAHA headlines as HumorGen does. 
Human-voted judgment data: Use Oogiri-GO or SemEval 2021 Task 7, which LoL uses.

General Allocation:
Tingyu Zhang: Generate the structure, train the model and interpret the jobs into Software Engineer language.
Supriya: Data preparation and result measurement.Humur Quantification and training details.

**Step 0: Project setup**

- **Goal:** everyone can work in parallel from day 1.
- **Owner:** Both.
- **Do:** create the repo, agree on the file formats (`candidates.jsonl`, `judgments.jsonl`, `ratings.json`, `sft.jsonl`, `dpo.jsonl`), and hand-write a 20-prompt toy version of every file.
- **Done when:** Brainstorm and discussion part.

**Step 1: Collect inputs**

- **Goal:** have prompts to write jokes for, and real human votes to teach judging.
- **Owner:** Supriya.
- **Do:** download the SemEval 2026 headlines (the prompts) and Oogiri-GO / SemEval 2021 Task 7 (human votes). Convert the votes into "which is funnier, 1 or 2?" rows.
- **Output:** `prompts.jsonl`, `human_pairs.jsonl`.
- **Done when:** about 1,000 prompts and a few thousand human-vote pairs are stored and spot-checked.

**Step 2: Generate candidate jokes**

- **Goal:** for every prompt, get many diverse jokes in different comedy styles.
- **Owner:** Supriya.
- **Do:** call a strong LLM API with each of the 6 persona prompts, 4 times each, which gives 24 jokes per prompt. Parse the `<JOKE>` tags and handle retries and rate limits.(Not necessary to use LLM, just easier)
- **Output:** `candidates.jsonl`, with prompt, persona and joke on each line.
- **Done when:** every prompt has 24 parsed jokes and failures have been retried.

**Step 3: Judge and rank the jokes**

- **Goal:** know which jokes are good and which are bad, for every prompt.
- **Owner:** Supriya, using the ranking method you specify.
- **Do:** build a judge service in which an LLM sees two jokes and returns "A" or "B". Compare jokes within each prompt and turn the wins into scores with Bradley–Terry.
- **Output:** `judgments.jsonl`, `ratings.json`.
- **Done when:** every joke has a score, and a manual look at the top and bottom 3 jokes for 10 prompts looks reasonable.

**Step 4: Build training datasets**

- **Goal:** turn the ranked jokes into training files.
- **Owner:** Supriya builds the files; Tingyu write the selection rules.
- **Do:**
    - SFT generation rows: prompt → each of the top-10 jokes.
    - SFT judgment rows: two jokes pair→ which is funnier, from the human votes plus judge results.
    - DPO rows: one of the top-5 jokes as chosen, one of the bottom-5 as rejected.
- **Output:** `sft.jsonl`, `dpo.jsonl`.
- **Done when:** Supriya  have each read 50 random rows and found no formatting or label errors.

**Step 5: Train the first model (SFT)**

- **Goal:** one model that can both write jokes and judge jokes.
- **Owner:** Tingyu.
- **Do:** fine-tune a small model (e.g. Qwen2.5-1.5B or 7B with LoRA) on `sft.jsonl`. Expose it through a `/generate` endpoint.
- **Output:** the `sft-v1` checkpoint.
- **Done when:** its jokes are clearly better than the base model's on 20 prompts.

**Step 6: Check whether the model is a good judge**

- **Goal:** Decide whether the model can judge its own jokes in Step 7.
- **Owner:** Supriya runs the test; Discuss to make the decision.
- **Do:** test the model's judging accuracy on held-out human-vote pairs.
- **Done when:** Get an accuracy number. If it is well above 50%, use the model itself as the judge. Otherwise keep the external LLM judge from Step 3.

Summary: Learning part, paper generating in the same time.

---

**Step 7: Self-improvement training (online DPO)**

- **Goal:** improve the model using its own new jokes, not only the teachers'.
- **Owner:** Tingyu, with Supriya's judge service handling the judging.
- **Do:** start DPO on `dpo.jsonl`. Every few hundred steps, the model writes 2 new jokes per prompt, the judge picks the winner, and the new pair is added to the training data.
- **Output:** the `dpo-online-v1` checkpoint.
- **Done when:** it beats `sft-v1` on Supriya's leaderboard, without the jokes getting longer or all collapsing into one style.

**Step 8: Inference (best-of-n)**

- **Goal:** get the best joke the model can produce at test time.
- **Owner:** Tingyu build the logic; Supriya integrates it into the service.
- **Do:** generate several jokes per prompt (optionally one per persona), and let the judge pick the best one.
- **Done when:** best-of-n beats single-sample output on the leaderboard.

**Step 9: Evaluation**

- **Goal:** show which steps actually helped.
- **Owner:** Supriya builds the tools; Supriya builds the test prompts; you interpret the results.
- **Do:**
    - A writes an out-of-distribution test set: facts, scenarios, quotes, not only headlines.
    - B runs a leaderboard using a **different** judge model than the one used for training.
    - B builds a small blind human-voting web page and collects about 50+ pairs.
    - Compare the base model, `sft-v1`, `dpo-online-v1`, and `dpo-online-v1` + best-of-n.
- **Done when:** there is a leaderboard table and a human-agreement number.

