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

**Films sampled:** 6 films chosen for varied reception, so no single label would dominate:

| Film | Reception profile |
|---|---|
| Scream 7 | Franchise sequel, mixed reception |
| THE BRIDE! | Sharply divisive |
| The Drama | Topical, polarizing subject matter |
| Project Hail Mary | Broadly acclaimed |
| Hoppers | Well-reviewed with some pushback |
| Ella McCay | Mixed critical reception |

If I'd only pulled from a beloved film like Project Hail Mary, the dataset would skew heavily toward `analysis` because people want to explain why it worked. Mixing in divisive films brought in more `hot_take` and `reaction` examples.

**Process:** Sorted each thread by Top, copied comments row-by-row, skipped AutoModerator posts, deleted/removed comments, and emoji-only or single-word entries.

**Label distribution:**

| Label | Total | Train | Val | Test |
|---|---|---|---|---|
| `hot_take` | 91 | 63 | 14 | 14 |
| `analysis` | 57 | 40 | 8 | 9 |
| `reaction` | 34 | 24 | 5 | 5 |
| `noise` | 23 | 16 | 4 | 3 |
| **Total** | **205** | **143** | **31** | **31** |

I used stratified splits (70/15/15) so minority classes like `noise` and `reaction` would actually appear in the test set.

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

One thing I didn't do but should have: class weights. With `hot_take` at 44% of training examples and `noise` at 11%, the model never had enough signal to learn the minority classes properly.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy | Test set size |
|---|---|---|
| Zero-shot baseline (Llama 3.3 70B via Groq) | **90.3%** | 31 |
| Fine-tuned DistilBERT | **71.0%** | 31 |

Fine-tuning **regressed** by 19.4 points. That's the main result and I address it directly in the reflection below.

---

### Per-Class Metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.89 | 0.89 | 0.89 | 9 |
| `hot_take` | 0.64 | 1.00 | 0.78 | 14 |
| `reaction` | 0.00 | 0.00 | 0.00 | 5 |
| `noise` | 0.00 | 0.00 | 0.00 | 3 |
| **weighted avg** | **0.57** | **0.71** | **0.61** | **31** |

**Zero-shot baseline — Llama 3.3 70B:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 0.89 | 0.89 | 0.89 | 9 |
| `hot_take` | 0.92 | 0.86 | 0.89 | 14 |
| `reaction` | 0.83 | 1.00 | 0.91 | 5 |
| `noise` | 1.00 | 1.00 | 1.00 | 3 |
| **weighted avg** | **0.91** | **0.90** | **0.90** | **31** |

---

### Confusion Matrix — Fine-Tuned Model (Test Set)

|  | Predicted: `analysis` | Predicted: `hot_take` | Predicted: `reaction` | Predicted: `noise` |
|---|---|---|---|---|
| **True: `analysis`** | **8** | 1 | 0 | 0 |
| **True: `hot_take`** | 0 | **14** | 0 | 0 |
| **True: `reaction`** | 0 | 5 | **0** | 0 |
| **True: `noise`** | 1 | 2 | 0 | **0** |

The model never predicted `reaction` or `noise` for any example in the test set. Everything collapsed into `hot_take`.

---

### Error Analysis

There were 9 wrong predictions out of 31. Before writing this section, I pasted all 9 into Claude and asked it to find common patterns. It flagged three things: all 9 had nearly identical low confidence scores (0.30–0.32); 8 of 9 predictions were `hot_take` regardless of the true label; and the `noise` errors both had evaluative-sounding language in them. Claude also suggested errors might cluster around short post length — I checked and that didn't hold up, since error #5 (the `analysis` case) is one of the longer comments in the test set. The confidence and `hot_take` default observations were accurate and shaped the analysis below.

**Error type 1: `reaction` → `hot_take` (5 errors — the main failure)**

Examples:
- "The moment the letter arrived and she just put it down and walked away without opening it. I understood her completely in that instant." → predicted `hot_take` (0.31)
- "The moment he realized what Rocky was sacrificing and just said his name. I couldn't breathe." → predicted `hot_take` (0.31)
- "When the solution clicked into place I actually stood up. In the theater. By myself." → predicted `hot_take` (0.32)
- "The final shot absolutely destroyed me." → predicted `hot_take` (0.31)

**Which labels are confused?** All five `reaction` examples in the test set were predicted as `hot_take`. The model learned `hot_take` as a default and never predicted `reaction` at all.

**Why is the boundary hard?** The emotional language in `reaction` comments ("destroyed me," "I couldn't breathe") looks a lot like the emotional language in strong `hot_take` comments ("carried this entire movie," "best of the year"). The actual boundary — whether the feeling is about one moment or the film overall — lives in the noun phrase, not the sentiment. "The final shot destroyed me" vs. "This movie destroyed me" differ by one word.

**Is this a labeling or data problem?** The labeling is consistent — all five belong in `reaction` under the decision procedure. The problem is that 24 training examples isn't enough for the model to learn a boundary that depends on semantic reference rather than vocabulary.

**What would fix it?** More `reaction` examples, specifically ones where the only difference from a `hot_take` is the scope of what's being referenced. Class weights during training would also help.

**Error type 2: `noise` → `hot_take` (2 errors)**

Examples:
- "What award circuit is this targeting? Oscars or more of a festival film?" → predicted `hot_take` (0.30)
- "Is the runtime really 2h20? Feels long for this kind of film." → predicted `hot_take` (0.31)

Both contain evaluation-adjacent language — asking about awards implies quality; "feels long" sounds like a complaint. The model isn't entirely wrong to detect that, and these are two of the harder `noise` cases I documented during annotation. The distinction is communicative function: are they evaluating the film or asking a question? The model can't figure that out without pragmatic inference.

**Error type 3: `analysis` → `hot_take` (1 error)**

Example: "The meta-commentary about franchise fatigue was smart until it kept repeating itself every fifteen minutes. Once is self-aware. Four times is the exact thing it's criticizing." → predicted `hot_take` (0.30)

The word "smart" in the opening clause patterns like a `hot_take` verdict. The specific structural evidence comes after it. The model seems to classify based on the first sentence rather than reading the whole comment — a shortcut that works most of the time but breaks when the evidence comes later.

---

### Sample Classifications

| Text (truncated) | True label | Predicted | Confidence |
|---|---|---|---|
| "The kills in this one actually felt choreographed for once — the kitchen scene used the geography of the house..." | `analysis` | `analysis` | 0.71 |
| "Best Scream since 2." | `hot_take` | `hot_take` | 0.68 |
| "When the twist landed I grabbed my boyfriend so hard he yelped." | `reaction` | `hot_take` | 0.31 |
| "Is this connected to the other franchise?" | `noise` | `analysis` | 0.31 |
| "Devastating. Cannot recommend it enough." | `hot_take` | `hot_take` | 0.65 |

The `analysis` prediction makes sense — "the kitchen scene used the geography of the house" is a named scene with a specific filmmaking observation, which is exactly what the label requires. The model's 0.71 confidence is noticeably higher than any of the wrong predictions (all 0.30–0.32), which were basically at chance for a four-class problem. The model isn't confidently wrong — it's uncertain on everything it misclassifies.

---

### Reflection: What the Model Captured vs. What Was Intended

I designed the labels around a structural distinction: does this comment give evidence, and if not, is it about a moment or the whole film? That's a question about what the comment is *pointing at*, not what vocabulary it uses.

What the model learned was simpler: does this comment contain evaluative language? That actually works for separating `analysis` from `hot_take` — analysis comments tend to use descriptive language about specific scenes, while `hot_take` comments use sentiment language about the film overall. But it completely breaks for `reaction` and `noise`, where the evaluative language is present but the function is different.

The gap between 71% and 90% accuracy isn't really a hyperparameter problem. A 70B instruction-tuned model can follow a prompt that says "reaction means a feeling about one specific moment" and apply it correctly. A 66M model fine-tuned on 143 examples learns word co-occurrence, not semantic reference. The labels are learnable — the baseline proves that — but not by this model from this much data.

---

## Spec Reflection

**Where the spec helped:** The instruction to write labels so "two people would agree on most examples" directly pushed me away from the first taxonomy draft, which separated `hot_take` from a "hyperbolic/bait" category based on tone intensity. That's a judgment call two people would never apply the same way. Switching to a structural test eliminated that.

**Where I diverged:** The spec assumes API scraping is the data collection path — the PRAW setup instructions are in the project guide. Reddit killed that option before this project was built (API approval required as of Nov 2025, `.json` endpoints closed May 2026). I ended up collecting manually, which was slower but actually forced me to read each comment carefully before annotating it, which probably helped with label consistency.

---

## AI Usage

**Instance 1 — Taxonomy design**
I asked Claude to help design a 4-label taxonomy for r/movies discourse quality. It proposed `Analytical`, `Reactive opinion`, `Hyperbolic/bait`, and `Noise`. The `Reactive opinion` / `Hyperbolic/bait` split relied on intensity judgment, which I knew would be inconsistent. I replaced those two with `hot_take` (any unsupported verdict, mild or extreme) and `reaction` (moment-specific feeling only), making the boundary structural rather than tonal.

**Instance 2 — Error pattern analysis**
After getting the 9 wrong predictions, I pasted them into Claude and asked it to find common patterns across the failures. It correctly identified the uniform low confidence scores, the `hot_take` default bias, and the evaluative-adjacent language in `noise` errors. It also suggested errors might cluster around short post length — I checked and that didn't hold, since the `analysis` error is one of the longer comments. I used the patterns that held up and discarded the one that didn't.

**Instance 3 — Edge case rules**
I asked Claude to help draft annotation rules for ambiguous cases. It produced 6 draft rules covering sarcasm, embedded jokes, and questions framed as opinions. I modified two: the sarcasm rule now specifies using intended meaning inferred from context (Claude's draft said "literal words"); the opinion-question rule defaults to `hot_take` rather than `reaction` to match the decision procedure ordering. I added two rules independently that Claude didn't surface: one for comments from people who haven't seen the film yet, and one for emoji-only comments (exclude rather than force into `noise`).

All 205 labels were applied manually. No AI assistance was used during annotation.

---

## Files

| File | Description |
|---|---|
| `comments_labeled.csv` | Annotated dataset (205 examples) |
| `planning.md` | Label definitions, edge case rules, data collection plan, annotation notes |
| `confusion_matrix.png` | Confusion matrix for the fine-tuned model on the test set |
| `evaluation_results.json` | Accuracy for both models (baseline: 0.903, fine-tuned: 0.710) |
