# ai201-project3-takemeter

# TakeMeter — r/movies Discourse Quality Classifier

A fine-tuned text classifier that labels Reddit comments from r/movies Official Discussion threads into one of four categories: `analysis`, `hot_take`, `reaction`, or `noise`.

---

## Community

**r/movies** — I chose this subreddit because its Official Discussion threads for new releases produce every type of comment in one place. Someone will write three paragraphs about the editing, someone else will say "best movie of the year," someone will post about a specific scene that wrecked them, and someone will ask if it's on Netflix yet. That range is what makes the labels meaningful.

There's also an informal norm in r/movies where people are more receptive to comments that back up their opinions with specific observations. So the distinction between "this was great" and "this worked because of X scene" actually matters to people who post there.

---

## Label Taxonomy

I used a fixed decision procedure applied top-to-bottom, stopping at the first match:

1. Does the comment cite a specific, checkable detail — a named scene, craft choice, or structural observation? → **`analysis`**
2. Does it state an overall verdict on the film without any specific evidence? → **`hot_take`**
3. Does it express a feeling about one specific moment, with no overall verdict? → **`reaction`**
4. Otherwise → **`noise`**

| Label | Definition | Example |
|---|---|---|
| `analysis` | A claim backed by a specific, verifiable film detail. Could someone check this by rewatching the named scene? | "The third act felt rushed because they cut the negotiation subplot the first half spent 20 minutes setting up." |
| `hot_take` | An overall quality verdict with no supporting evidence. How strong or mild the wording is doesn't matter — only whether evidence is present. | "Jenna Ortega carried this entire movie on her back. The rest of the cast felt like wallpaper." |
| `reaction` | An emotional response to one specific moment, with no overall verdict anywhere in the comment. If a verdict appears, `hot_take` fires first. | "When the mask appeared in the reflection I genuinely screamed in the theater." |
| `noise` | No opinion or emotional response about the film at all — logistical questions, off-topic remarks, memes with no evaluative content. | "Is this streaming anywhere yet or still theaters only?" |

The most important design decision was making the boundary a structural test rather than a judgment call about tone or intensity. An earlier draft split `hot_take` from a "hyperbolic/bait" category based on how exaggerated the language felt — that fell apart because two people would never apply it the same way. Switching to a binary test (evidence present or not; moment vs. film-wide scope) made annotation feel consistent.

---

## Data Collection

**Source:** Manual collection from r/movies Official Discussion threads. I originally planned to use PRAW to scrape via Reddit's API, but Reddit's Responsible Builder Policy (Nov 2025) now requires manual review for all API access, and the old `.json` URL workaround was closed in May 2026. So I browsed Reddit directly and copied comments by hand.

**Films sampled:** 6 films chosen for varied reception and genre, so no single label would dominate:

| Film | Rows | Reception profile |
|---|---|---|
| Scream 7 | 29 | Franchise sequel, mixed reception |
| THE BRIDE! | 30 | Sharply divisive |
| The Drama | 36 | Topical, polarizing subject matter |
| Project Hail Mary | 23 | Broadly acclaimed |
| Hoppers | 47 | Well-reviewed with some pushback |
| Michael | 47 | Mixed critical reception (40% RT, 96% audience) |

**Process:** Sorted each thread by Top, copied comments row-by-row, skipped AutoModerator posts, deleted/removed comments, and emoji-only or single-word entries that couldn't be labeled without reply context.

**Label distribution:**

| Label | Total | Train | Val | Test |
|---|---|---|---|---|
| `analysis` | 124 | 87 | 18 | 19 |
| `hot_take` | 50 | 35 | 8 | 7 |
| `reaction` | 31 | 22 | 5 | 4 |
| `noise` | 7 | 5 | 1 | 1 |
| **Total** | **212** | **148** | **32** | **31** |

I used stratified splits (70/15/15) so all four labels appear in each split.

**What the distribution actually reflects:** Going in, I expected `hot_take` to dominate — casual movie discussion usually skews toward unsupported verdicts. The actual dataset flipped that assumption. `analysis` is 58% of the total, and `noise` nearly disappeared (only 7 rows). Both are real findings about where the data came from, not labeling errors.

Official Discussion threads self-select for substantive engagement. People who sort by Top and post early on a film discussion are usually there to explain a reaction, not just state it. `hot_take` comments ("loved it," "best Pixar ever") exist but get fewer upvotes and end up lower in the sort — they're harder to find when you're copying from the top of the thread. `noise` is even rarer because r/movies moderates aggressively and logistical comments get removed or buried.

Thread type also matters within the dataset. The Scream 7 and THE BRIDE! early rows came from review aggregator threads before the films were widely seen — those produced more pre-screening `noise` and verdict-style `hot_take` comments. The Official Discussion threads for the same films after release produced almost entirely `analysis` and `reaction`. If I were collecting again, I'd pull exclusively from Official Discussion threads and add more films to increase `hot_take` and `noise` coverage.

The Michael biopic is the most analysis-heavy film in the dataset (36 of 47 rows, or 77%). Biopic discourse naturally involves comparing specific scenes to historical record, counting what was omitted, and debating what the film should have covered — all of which label as `analysis` under the decision procedure. The film's extreme critic/audience score gap (40% RT vs. 96% audience) also generated unusually detailed comments from people trying to explain the gap, which skews analytical.

---

## Difficult Annotation Examples

**1. "Couldn't sleep after. Still can't stop thinking about it."**
This felt like a `reaction` at first since it's expressing feeling. But there's no specific moment being referenced — "can't stop thinking about it" is about the film as a whole. I labeled it **`hot_take`**. The test I used: *what* is she still thinking about? If the comment doesn't say, it's a verdict, not a moment reaction.

**2. "Is the runtime really 2h20? Feels long for this kind of film."**
The second sentence sounds like a quality judgment. But the commenter hasn't seen the film yet — they're not evaluating what's in it, they're comparing the runtime to genre expectations. Labeled **`noise`**. I added a rule after this: if the commenter hasn't seen the film, they can't be making a `hot_take` or `reaction`.

**3. "The meta-commentary about franchise fatigue was smart until it kept repeating itself every fifteen minutes. Once is self-aware. Four times is the exact thing it's criticizing."**
The word "smart" reads like a verdict, but there's a specific structural claim here: the film uses a device a countable number of times in a way that contradicts its own theme. You could verify that by rewatching. Labeled **`analysis`**. This is the hardest `analysis`/`hot_take` boundary case in the whole dataset.

---

## Model

**Base model:** `distilbert-base-uncased` (66M parameters). With only 143 training examples, a larger model would be more likely to overfit without significant extra work.

**Key hyperparameter decisions:**

| Hyperparameter | Value | Why |
|---|---|---|
| `num_train_epochs` | 3 | Standard for small datasets. More epochs risked overfitting. |
| `learning_rate` | 2e-5 | Standard starting point for BERT fine-tuning. |
| `per_device_train_batch_size` | 16 | Fits T4 GPU without issues. |
| `warmup_steps` | 50 | Prevents large gradient updates while the classification head initializes. |
| `weight_decay` | 0.01 | Light regularization against overfitting. |

One thing I didn't do but should have: class weights. With `analysis` at 58% of training examples and `noise` at 3%, the model never had enough signal to learn the minority classes properly.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy | Test set size |
|---|---|---|
| Zero-shot baseline (Llama 3.3 70B via Groq) | **78.1%** | 32 |
| Fine-tuned DistilBERT | **62.5%** | 32 |

Fine-tuning **regressed** by 15.6 points. That's the main result and I address it directly in the reflection below.

---

### Per-Class Metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.63 | 1.00 | 0.77 | 19 |
| `hot_take` | 0.50 | 0.13 | 0.20 | 8 |
| `reaction` | 0.00 | 0.00 | 0.00 | 4 |
| `noise` | 0.00 | 0.00 | 0.00 | 1 |
| **weighted avg** | **0.51** | **0.63** | **0.52** | **32** |

**Zero-shot baseline — Llama 3.3 70B:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 1.00 | 0.68 | 0.81 | 19 |
| `hot_take` | 0.64 | 0.88 | 0.74 | 8 |
| `reaction` | 0.67 | 1.00 | 0.80 | 4 |
| `noise` | 0.50 | 1.00 | 0.67 | 1 |
| **weighted avg** | **0.85** | **0.78** | **0.79** | **32** |

---

### Confusion Matrix — Fine-Tuned Model (Test Set)

|  | Predicted: `analysis` | Predicted: `hot_take` | Predicted: `reaction` | Predicted: `noise` |
|---|---|---|---|---|
| **True: `analysis`** | **19** | 0 | 0 | 0 |
| **True: `hot_take`** | 7 | **1** | 0 | 0 |
| **True: `reaction`** | 3 | 1 | **0** | 0 |
| **True: `noise`** | 1 | 0 | 0 | **0** |

The model never predicted `reaction` or `noise` for any example. It got every `analysis` correct but almost nothing else — essentially functioning as an `analysis` vs. everything-else detector, and then defaulting `everything-else` to `analysis` as well.

---

### Error Analysis

There were 12 wrong predictions out of 32. Before writing this section, I pasted all 12 into Claude and asked it to find common patterns. It flagged three: all 12 had uniformly low confidence (0.30–0.39), pointing to genuine uncertainty rather than confident errors; 7 of 12 were `hot_take → analysis`, making that the single dominant failure mode; and several of the `hot_take` errors contained comparative or relational language ("Felt like a Joe Jackson movie," "Reminded me of Goofy Movie") that may have pattern-matched to `analysis` because analysis comments also frequently reference other films. Claude also suggested errors might cluster around comment length — shorter comments being misclassified more — but that didn't hold up on inspection since several multi-sentence `hot_take` comments were also wrong. The directional bias and the comparative language observation both held and shaped the analysis below.

**Error type 1: `hot_take` → `analysis` (7 errors — the dominant failure)**

Examples:
- "Hope they put the franchise on ice after this and sell off the rights to a better studio." → predicted `analysis` (0.33)
- "Felt like a Joe Jackson movie more than a Michael movie" → predicted `analysis` (0.32)
- "Watching certain scream subs get defensive told me this was probably a stinker. That's a shame scream was wildly consistent too" → predicted `analysis` (0.35)
- "BRO Rachel was a villain I am so glad I saw this. The actress did a great job at making me hate her." → predicted `analysis` (0.39)
- "Justice for the classroom turtle! Absolutely loved this movie. Reminds me of Goofy Movie. So funny, yet so heartwarming." → predicted `analysis` (0.37)

**Which labels are confused?** `hot_take → analysis` accounts for 7 of 12 errors and is entirely directional. The model predicted `analysis` 30 out of 32 times — it learned `analysis` as a near-universal default.

**Why is the boundary hard?** The training set is 58% `analysis`, so the model learned `analysis` vocabulary as the baseline signal for anything that isn't clearly neutral. Several of these misclassified `hot_take` comments contain comparative language ("Felt like a Joe Jackson movie," "Reminded me of Goofy Movie") — which is also common in `analysis` comments that compare a film to a director's earlier work or to genre peers. The model can't distinguish "this film reminded me of X" as a comparison-as-evidence from "this film reminded me of X" as a casual verdict.

**Is this a labeling or data problem?** The labeling is consistent — all seven belong in `hot_take` under the decision procedure (none cite a checkable film detail). The problem is training data imbalance: with 86 `analysis` examples in training vs. 35 `hot_take`, the model tipped heavily toward the majority class.

**What would fix it?** More `hot_take` examples and class weights during training. Class weights would penalize `analysis` predictions more when the true label is `hot_take`, preventing the majority class from dominating the learned boundary.

**Error type 2: `reaction` → `analysis` (3 errors)**

Examples:
- "It made me feel sad when Emma day dreamed about Charlie laughing with her and comforting her the morning after. Then the snap to reality of the 2 on completely different chairs." → predicted `analysis` (0.35)
- "The abrupt cut to the photographer repeatedly saying who they needed to shoot had my whole theater going nuts." → predicted `analysis` (0.35)
- "Did not expect this movie to have a body count" → predicted `analysis` (0.30)

These reaction comments all describe specific scenes in enough detail that they pattern-match to `analysis`. "The abrupt cut to the photographer" names a specific editing technique; "Emma day dreamed about Charlie" names a specific scene. The model appears to key on scene-specificity as an `analysis` signal regardless of whether the comment is making a claim about it or just reacting to it. This is a surface-feature failure: it detected the right textual marker (scene reference) but not the right function (reaction vs. argument).

**Error type 3: `noise` → `analysis` (1 error), `reaction` → `hot_take` (1 error)**

"Time to use my free trial of MGM+ I guess and promptly cancel after watching." → predicted `analysis` (0.33). This is a pre-screening logistics comment — the commenter hasn't seen the film and is commenting on streaming access. It contains no film evaluation whatsoever. The model's `analysis` prediction here is difficult to explain by surface features alone; it may be that "cancel after watching" read as a negative verdict pattern.

"All I want now, is a rocky friend 😢😢😭" → predicted `hot_take` (0.30). The emoji carry emotional weight but no verdict; the comment is a feeling about a character, not an overall quality judgment. This is the `reaction`/`hot_take` boundary failure from the previous run re-appearing with a single example.

---

### Sample Classifications

| Text (truncated) | True label | Predicted | Confidence |
|---|---|---|---|
| "What was so frustrating to me was the way the movie would present some great feminist ideas and then shrink away..." | `analysis` | `analysis` | 0.68 |
| "Truly a bizarre mix of tones." | `hot_take` | `hot_take` | 0.52 |
| "Felt like a Joe Jackson movie more than a Michael movie" | `hot_take` | `analysis` | 0.32 |
| "The abrupt cut to the photographer repeatedly saying who they needed to shoot had my whole theater going nuts." | `reaction` | `analysis` | 0.35 |
| "Time to use my free trial of MGM+ I guess and promptly cancel after watching." | `noise` | `analysis` | 0.33 |

The `analysis` prediction on the Bride! feminist critique comment makes sense — it names specific plot points, character behavior across acts, and structural problems, which is exactly the `analysis` pattern. The 0.68 confidence is higher than any wrong prediction, consistent with the model being more certain when the label is `analysis`. The correct `hot_take` ("Truly a bizarre mix of tones") is notable because it's short, unambiguous, and contains no film-specific details — the model's one reliable `hot_take` prediction. Everything else collapses into `analysis` at low confidence.

---

### Reflection: What the Model Captured vs. What Was Intended

The intended boundary was structural: does this comment give evidence, and if not, is it about a moment or the whole film? That's a question about what the comment is *pointing at*, not what words it uses.

What the model learned was simpler: is this comment structured like the majority of what I was trained on? With 58% of training data being `analysis`, the model learned to treat `analysis` as the default response for any comment that isn't immediately obvious. It got every `analysis` example right (19/19, 100% recall) but only 1/8 `hot_take` and 0/4 `reaction` — not because those classes are harder, but because it stopped trying to distinguish them.

This is the same problem as the previous run, just flipped. In the earlier version of the dataset, `hot_take` was 44% of training data and the model defaulted to `hot_take`. In this dataset, `analysis` is 58% and the model defaults to `analysis`. In both cases, the model learned the majority class distribution rather than the actual label boundaries. The boundary that matters — whether a comment gives evidence, and whether its scope is a moment or the whole film — is a semantic and referential distinction that a 66M parameter model fine-tuned on ~150 examples cannot learn from word co-occurrence alone.

The baseline at 78.1% does better on every non-`analysis` class because a 70B instruction-tuned model can actually follow the definitions in the prompt. The labels are learnable — the baseline proves it — but not by a small model from this data volume and this class imbalance.

---

## Spec Reflection

**Where the spec helped:** The instruction to write labels so "two people would agree on most examples" directly pushed me away from the first taxonomy draft, which separated `hot_take` from a "hyperbolic/bait" category based on tone intensity. That's a judgment call two people would never apply the same way. Switching to a structural test eliminated that.

**Where I diverged:** The spec assumes API scraping is the data collection path — the PRAW setup instructions are in the project guide. Reddit killed that option before this project was built (API approval required as of Nov 2025, `.json` endpoints closed May 2026). I ended up collecting manually, which was slower but also forced me to read each comment carefully before labeling it, which probably helped annotation consistency.

---

## AI Usage

**Instance 1 — Taxonomy design**
I asked Claude to help design a 4-label taxonomy for r/movies discourse quality. It proposed `Analytical`, `Reactive opinion`, `Hyperbolic/bait`, and `Noise`. The `Reactive opinion` / `Hyperbolic/bait` split relied on intensity judgment, which I knew would be inconsistent. I replaced those two with `hot_take` (any unsupported verdict, mild or extreme) and `reaction` (moment-specific feeling only), making the boundary structural rather than tonal.

**Instance 2 — Error pattern analysis**
After getting the 12 wrong predictions, I pasted them into Claude and asked it to find common patterns. It correctly identified the uniform low confidence scores, the `analysis` default bias, and the comparative language appearing in several `hot_take` errors. It also suggested errors might cluster around comment length — I checked and that didn't hold, since several multi-sentence `hot_take` comments were also wrong. I used the patterns that held up and discarded the one that didn't.

**Instance 3 — Edge case rules**
I asked Claude to help draft annotation rules for ambiguous cases. It produced 6 draft rules covering sarcasm, embedded jokes, and questions framed as opinions. I modified two: the sarcasm rule now specifies using intended meaning inferred from context (Claude's draft said "literal words"); the opinion-question rule defaults to `hot_take` rather than `reaction` to match the decision procedure ordering. I added two rules independently that Claude didn't surface: one for comments from people who haven't seen the film yet, and one for emoji-only comments (exclude rather than force into `noise`).

All 212 labels were applied manually. No AI assistance was used during annotation.

---

## Files

| File | Description |
|---|---|
| `comments_labeled.csv` | Annotated dataset (212 examples) |
| `planning.md` | Label definitions, edge case rules, data collection plan, annotation notes |
| `confusion_matrix.png` | Confusion matrix for the fine-tuned model on the test set |
| `evaluation_results.json` | Accuracy for both models (baseline: 0.781, fine-tuned: 0.625) |
