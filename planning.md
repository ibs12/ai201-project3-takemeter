# TakeMeter — Planning: r/movies Discourse Quality Classifier

## Community

**r/movies** — chosen because it has daily/official discussion threads for every wide release, which means a single thread contains the full range of discourse quality (careful critique, flat opinions, ragebait, jokes) rather than requiring scraping across many smaller subs.

## Label Taxonomy

Four mutually exclusive labels. Apply them with a fixed decision procedure, in order — this removes the ambiguity that comes from judging "how bold" or "how exaggerated" a comment feels, which two annotators will never agree on consistently.

**Decision procedure (apply top to bottom, stop at first match):**
1. Does the comment cite a specific, checkable detail from the film (a named scene, a technical/craft choice, a structural comparison) to support its claim? → **analysis**
2. Does it state an overall verdict on the film's quality — explicitly or via a rating/recommendation — without citing a specific detail? → **hot_take**
3. Does it express a feeling tied to one specific moment or event, without stating an overall verdict? → **reaction**
4. Otherwise → **noise**

### 1. `analysis`
The claim is backed by a specific, verifiable piece of evidence about the film itself. The boundary test: could someone check this claim by rewatching the specific scene/element named?

- ✅ "The third act felt rushed because they cut the negotiation subplot the first half spent 20 minutes setting up."
- ✅ "The lead's restraint in the dinner scene is what made the later breakdown actually land."
- ❌ "The acting was great" — no scene or element named, so this is `hot_take`
- ❌ "5/10, mid" — a rating is a verdict, not evidence, so this is `hot_take`

### 2. `hot_take`
A confident, evaluative claim about the film's overall quality, stated with no supporting evidence. Mild or extreme phrasing makes no difference here — the only test is whether evidence is present, not how strongly the opinion is worded.

- ✅ "Loved it, easily my favorite this year." (no evidence — confident verdict)
- ✅ "7/10, solid but forgettable."
- ✅ "This single-handedly killed the franchise." (still just an unsupported verdict — extremity isn't the test)
- ✅ "Worst movie I've seen in years."

### 3. `reaction`
An emotional response to one specific moment or event in the film, with no stated verdict on the film as a whole. The boundary test: is this about a moment, or about the movie overall? If it names an overall verdict anywhere in the comment, rule 2 fires first and it's `hot_take` instead.

- ✅ "bro the ending 💀" (feeling about one moment, no overall verdict stated)
- ✅ "I was NOT ready for that twist."
- ❌ "The ending wrecked me, best of the year" — contains an overall verdict ("best of the year"), so this is `hot_take`

### 4. `noise`
Contains no opinion, verdict, or feeling about the film at all.

- ✅ "Wait, is this connected to the other franchise?" (factual question, no opinion)
- ✅ Off-topic tangents, unrelated meme references
- ✅ Pure box-office/scheduling comments with no quality judgment attached

## Edge Case Rules

These are decision rules to apply consistently once you're labeling real data. Add to this list any time a new ambiguous case shows up during annotation — don't resolve it ad hoc and move on, or you'll end up with inconsistent labels across your dataset.

1. **Joke with an embedded specific critique** ("the script feels like it was written by a particularly motivated golden retriever — no idea why Tom would leave his job, the motivation just isn't there") → Strip the joke. If a concrete, checkable detail remains, rule 1 fires → **`analysis`**. If nothing concrete remains, run the rest of the decision procedure on what's left (verdict → `hot_take`, moment-feeling → `reaction`, neither → `noise`).

2. **Comment that opens with one specific point, then devolves into ranting** → Test: if you removed the unsupported venting, would the specific point still stand as a complete, coherent claim on its own? If yes, the evidence is load-bearing → **`analysis`**. If the specific detail is just decoration and the rant would read the same without it → **`hot_take`**.

3. **Question framed as an opinion** ("did anyone else think the pacing dragged in the back half?") → No checkable detail, but implies an overall verdict (pacing = negative judgment) → **`hot_take`**. If reasoning is attached ("...because the action set pieces basically disappear for 40 minutes") → **`analysis`**.

4. **Unsupported comparison to another film** ("this is just [Film B] but worse") → No specific detail named → **`hot_take`**. If a specific shared or diverging element is named ("...but the villain has no motivation this time") → **`analysis`**.

5. **Sarcasm/irony** ("Oh yeah, real masterpiece" following a list of plot holes) → Run the decision procedure on the *intended* meaning, inferred from surrounding context, not the literal words. Specific reasoning present → **`analysis`**. Just exaggeration with nothing concrete → **`hot_take`**.

6. **Box office/financial speculation with no quality judgment** ("this is going to bomb, nobody wants a 4th installment") → No opinion about the film's quality at all → **`noise`**, unless the prediction is tied to a craft-based reason ("...the trailer's pacing problems are obvious") → **`analysis`**.

7. **Moment-reaction that also contains an overall verdict** ("the ending wrecked me, best of the year") → Rule 2 of the decision procedure fires before rule 3 — the overall verdict takes precedence → **`hot_take`**, not `reaction`. `reaction` only applies when no overall verdict is present anywhere in the comment.

8. **Emoji-only or single-interjection comments with no text** (a string of 🔥 emoji, a GIF reply, "lol") → Their meaning depends entirely on what they're replying to, which you may not reliably capture out of thread context. Exclude these from the dataset rather than force a label — don't stretch `noise` into a catch-all for things you can't actually judge.

## Data Collection Plan

**Method: manual collection, not API scraping.** Reddit's Responsible Builder Policy (updated Nov 2025) now requires manual approval for all API access, including hobby projects, and the old `.json`-URL workaround was closed in May 2026. Neither is viable on this timeline, so data collection happens by browsing r/movies normally and copying comment text into a spreadsheet — this also doubles as the "read 30-40 real posts before finalizing labels" step the project requires, so it's not pure overhead.

To get good label variance, mix films by reception rather than reading only one type of thread — an all-positive or all-negative thread will skew the dataset toward 1-2 labels. Recent 2026 releases with genuinely split reception, useful for this:

- **Scream 7** — franchise sequel, reviews ranged from "perfunctory" to "more engaging than any 7th film has any right to be"
- **THE BRIDE!** — sharply divisive (one critic called it one of the worst they've seen; another praised its "punk rock energy")
- **The Drama** — topical/controversial subject matter, explicitly noted by critics as polarizing
- **Project Hail Mary** — broadly acclaimed blockbuster (good source of detailed, positive `analysis` comments)
- **Hoppers** — well-reviewed but not universal (good contrast: mostly positive with some `hot_take` pushback)
- **Ella McCay** — mixed critical reception

Process: search r/movies for "Official Discussion" + the film title, sort comments by Top, and paste comment text into the spreadsheet template — skip AutoModerator/bot comments, deleted/removed comments, and emoji-only/single-word junk (edge case 8). Target ~35-40 comments per film × 6 films ≈ 210-240, giving buffer above the 200 minimum.

Split: stratify by label so train/val/test each have a proportional mix of all four labels, not just a random split (with 4 unevenly-distributed labels, a naive random split risks a test set missing one label almost entirely). Roughly 70/15/15.

## Annotation Process

1. Self-annotate using the rules above as the canonical reference.
2. After the first ~50 labeled examples, take a 24-hour break, then re-label a random 10% sample blind (without looking at your original labels) and compare. If disagreement with yourself is above ~15%, the taxonomy or rules need tightening before you label the rest — that's a signal the categories aren't crisp enough yet, not a personal failure.
3. Any new ambiguous case → write a rule for it in this doc *before* continuing, then spot-check a few earlier labels against the new rule.

## For the README (fill in after annotation)

- [ ] Final label distribution (count per label, train/val/test)
- [ ] At least 3 genuinely difficult real examples + what you decided + why
- [ ] Any rules added during annotation that aren't listed above yet
