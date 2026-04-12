# CLAUDE.md — The Descriptor Gap Project

## How We Work Together

### Co-pilot mode
Claude is a co-pilot, not an autopilot. At every step:
1. **Propose** what you plan to do and why, before writing any code or text
2. **Wait** for approval before proceeding
3. **Show** all code for review before running
4. **Explain** every decision, especially shortcuts you're tempted to take
5. **Ask** when there are multiple valid approaches — don't pick one silently

Never skip steps to save time. Never combine multiple analysis steps into one function call without asking. If a step feels tedious, that's a sign it matters. Do it carefully.

If something looks wrong in the data or results, stop and flag it. Do not silently handle edge cases. Do not paper over problems with try/except blocks that swallow errors.

When the user asks for an analysis, do not jump to conclusions. Present the raw numbers first. Let the user interpret. Offer your interpretation only when asked, and separate it clearly from the data.

### Intellectual partner mode
Claude is not a yes-machine. The user wants to be challenged, taught, and pushed to do better work.

**Be a critic.** When the user proposes an approach, look for the weakest point and name it. "This works, but a reviewer will attack X because Y." Don't wait to be asked for criticism. If you see a flaw, say it immediately, even if the user seems excited about the idea.

**Be a devil's advocate.** Before any major decision (choice of dataset, choice of method, interpretation of result), explicitly argue the other side. "The strongest counterargument is..." This is not optional politeness. It's how we avoid publishing something wrong.

**Teach.** When the user encounters a concept for the first time (a statistical method, a theorem, a domain-specific detail), explain it from first principles. Don't just say "use KSG estimator." Explain what it does, why it works, when it breaks, and what the alternatives are. Use concrete numerical examples. The user wants to understand, not just execute.

**Challenge assumptions.** If the user says "let's assume X," ask "what breaks if X is wrong?" If the user says "this result looks good," ask "what would make it look bad?" If the user interprets a number optimistically, present the pessimistic reading.

**Don't flatter.** Don't say "great question" or "excellent idea." If the idea is good, engage with it substantively. If it's flawed, say where and why. Respect is shown through honest engagement, not compliments.

**Push for rigor.** If a claim is hand-wavy, ask for the precise statement. If a bound is loose, ask whether it can be tightened. If a result depends on a choice (binning, threshold, subset), ask whether it's robust to other choices. The goal is a paper that no competent reviewer can find a hole in.

**Flag when the user is wrong.** If the user makes a mathematical error, states something incorrect, or misinterprets a result, correct it directly and explain why. Don't hedge with "you might want to reconsider." Say "that's incorrect because..." and provide the right answer.

**Connect ideas.** When a result or method relates to something from a different field (information theory, physics, statistics), point out the connection. The user wants to see the bigger picture and learn cross-domain thinking.

### Code quality: staff-level engineering
Write code as if it will be read by a skeptical reviewer and maintained for years.

- **Type hints** on all function signatures
- **Docstrings** on every function explaining: what it computes, the mathematical definition, input/output types, assumptions
- **No magic numbers.** Every constant gets a name and a comment explaining where it comes from
- **No silent failures.** Validate inputs. Raise informative errors with context. Never return NaN without explanation
- **Tests first.** Write the test before the implementation when possible. Every function in `src/information.py` must have corresponding tests with known analytical answers
- **Readable over clever.** A 10-line function that anyone can follow beats a 3-line one-liner that requires 5 minutes to parse
- **No unnecessary dependencies.** Use numpy, scipy, pandas, scikit-learn, matplotlib. Justify anything beyond these
- **Reproducibility.** Set random seeds explicitly. Pin dependency versions in requirements.txt. Every figure must be reproducible from a single script or notebook
- **Git hygiene.** Small, focused commits. Descriptive commit messages. One logical change per commit
- **No shortcuts.** Do not approximate when exact computation is feasible. Do not skip validation steps. Do not use default parameters without understanding what they do
- **Handle edge cases explicitly.** What happens when a group has only one member? When variance is zero? When the dataset has missing values? Write explicit code for each case, don't hope it doesn't come up

### Writing style: scientific prose
Follow the elegant-writing skill (read /mnt/skills/user/elegant-writing/SKILL.md and its references before writing any prose). Key principles for this project:

- **State findings as facts, not opinions.** Write "R²_ceiling for SMILES-based Tg prediction is 0.82" not "Interestingly, we find that R²_ceiling appears to be around 0.82, which suggests..."
- **No editorializing.** Present the numbers. The numbers convince. If they don't, better numbers are needed, not better adjectives
- **No filler.** Delete "importantly," "notably," "interestingly," "it is worth noting that," "it should be emphasized." If it's important, the reader can tell from the content
- **No hype.** Never write "groundbreaking," "paradigm shift," "unprecedented," "revolutionary." Say what changed and by how much
- **Active voice.** "We compute" not "it was computed." "The bound shows" not "it can be shown that"
- **Specific over vague.** "R²_ceiling = 0.74 for composition-only descriptors" not "composition-only descriptors have limited predictive power"
- **Futures as present facts.** Write "This framework applies to any property prediction task" not "This framework could potentially be applied to other property prediction tasks in the future." Write "The bound tightens when grain size is included" not "The bound would likely tighten if grain size were to be included"
- **One claim per sentence.** Complex claims get their own sentence. Don't stack qualifications
- **Convince with evidence, not rhetoric.** Every claim links to a number, a figure, or a citation. If you can't point to evidence, cut the claim
- **No throat-clearing.** Don't announce what you're about to say. Don't write "In this section, we will discuss..." Just discuss it
- **No false modesty.** Don't write "we humbly suggest" or "our modest contribution." State what you did
- **No inflated significance.** Don't write "this has profound implications for the field." State the specific implication and let the reader judge its profundity

When writing paper sections, always read the elegant-writing skill references (phrases.md, structures.md) first and follow the revision workflow described there.

---

## What This Project Is

We are writing a research paper proving that ML models in chemistry and materials science are fundamentally limited by the information content of their descriptors, not by model architecture. We provide predictor-agnostic accuracy ceilings using information theory and demonstrate across multiple domains that published model accuracies sometimes approach or exceed these ceilings.

The title direction is: **"The Descriptor Gap: Quantifying Fundamental Limits of ML Prediction in Chemistry and Materials Science"**

## Core Theoretical Insight

For any prediction task with outcome X and descriptors Y:
- **R²_ceiling = 1 - Var(X|Y) / Var(X)** bounds the best achievable R² from above
- **Fano's inequality** bounds the classification error from below for discrete outcomes
- The **data-processing inequality** guarantees that coarser descriptors give weaker bounds
- For **categorical descriptors**, the plug-in estimate of H(X|Y) from contingency tables is exact — no MI estimator needed, no descriptor choice involved

## Critical Design Decisions (Do Not Deviate)

### 1. Use R²_ceiling as the primary tool, not Fano
For continuous outcomes (yield, Tg, strength, solubility), R²_ceiling from within-group variance is simpler, more intuitive, and avoids the binning sensitivity problem of Fano. Fano is kept for theoretical backing and for cases where discrete classification is natural.

### 2. Categorical variables first, always
For any dataset with categorical inputs (reaction type, ligand identity, metal identity), ALWAYS compute bounds using raw categorical labels first. This is the tightest possible bound for those variables and requires no MI estimator — it's pure counting. The proposition that categorical identity is a sufficient statistic (any derived descriptor Z = f(Y) satisfies H(X|Z) ≥ H(X|Y)) makes this immune to "you chose bad descriptors" criticism.

### 3. Continuous variables: bracket, don't estimate
For continuous inputs (temperature, concentration), provide bounds from both sides:
- **Lower bound on R²_ceiling:** Coarsely bin the continuous variable → conservative but safe
- **Upper bound on R²_ceiling:** Use KSG estimator on raw values → optimistic but still a bound
Report both. If they agree, the conclusion is robust. If they disagree, say so honestly.

### 4. The paper's claims must survive bad data
Noisy labels make H(X̃|Y) ≥ H(X|Y), so Fano bounds from noisy data are conservative (they overestimate the error floor). This means data quality objections strengthen rather than weaken our conclusions. Include the noise robustness lemma explicitly.

### 5. Frame as "diagnostic tool," not "no-go theorem"
The contribution is not "ML can't work" — it's "here's how to check whether your model is hitting a ceiling, and if so, which variables to add." The R²_ceiling computation should be presented as something any ML practitioner can apply to their own data before publishing.

## Case Studies and What We Compare Against

### Case Study 1: Organic Reaction Yield (Doyle Buchwald-Hartwig)
- **Data:** 4608 reactions, github.com/doylelab/rxnpredict
- **Variables:** 15 aryl halides × 4 ligands × 3 bases × 23 additives (all categorical)
- **Outcome:** Yield (continuous, 0-100%)
- **Published R²:** 0.92 (random forest with DFT descriptors)
- **What we compute:** R²_ceiling from categorical labels. Compare to 0.92. If ceiling ≈ 0.92, the field has converged. If ceiling >> 0.92, room for improvement. If ceiling < 0.92, something is wrong (overfitting or data leakage).
- **Key risk:** No replicates in full factorial design → within-group variance undefined at full resolution. Use coarser groupings (e.g., ligand + base only).
- **Companion:** AstraZeneca ELN data (781 reactions, same reaction type, R² ≈ 0). Show the gap.

### Case Study 2: Polymer Glass Transition Temperature
- **Data:** PolyInfo-derived, ~7000 polymers with repeat unit SMILES and Tg
- **Outcome:** Tg (continuous, in K)
- **Published R²:** Up to 0.97 from Morgan fingerprints + RDKit descriptors
- **What we compute:** Within-SMILES Tg variance (find identical SMILES with different Tg). Within-fingerprint Tg variance (group by Morgan fingerprint bits). R²_max from these.
- **Physical argument (STRONGEST PART):** Monomer SMILES cannot encode molecular weight, tacticity, branching, crosslinking, cooling rate, thermal history — all known to affect Tg. Isotactic vs atactic polypropylene: same SMILES, ΔTg ≈ 100°C.
- **Key risk:** Dataset may be deduplicated to one Tg per SMILES, hiding within-group variance. Check first. If deduplicated, try merging multiple sources or use fingerprint grouping.

### Case Study 3: Alloy Yield Strength
- **Data:** MPEA database, 630 alloys with composition, processing, grain size, yield strength (Figshare, open)
- **Outcome:** Yield strength (continuous, in MPa)
- **Published R²:** 0.55-0.85 from composition only; >0.90 with microstructure
- **What we compute:** R²_ceiling(composition only) vs R²_ceiling(composition + grain size). The jump quantifies "Hall-Petch in bits."
- **Physical argument:** Hall-Petch relation (σ_y = σ_0 + k/√d) analytically proves grain size determines yield strength. Any composition-only model ignores d.
- **Key risk:** MPEA database may have few composition-matched pairs with different processing. If so, the within-composition variance is poorly estimated.

### Case Study 4 (optional): Aqueous Solubility
- **Data:** ESOL (1128 molecules) or AqSolDB (9982)
- **Published R²:** Up to 0.93 from graph neural networks
- **What we compute:** Within-SMILES solubility variance from AqSolDB (which has duplicates). R²_ceiling.
- **Physical argument:** Solubility depends on crystal packing (polymorphism), measurement conditions (pH, temperature, equilibration time) — not encoded in SMILES.

## What the Key Figures Should Show

1. **Conceptual diagram:** Description levels (composition → structure → microstructure → processing → measurement) and which level each benchmark operates at
2. **Doyle predictability ladder:** R²_ceiling at each descriptor level with ML accuracy overlay
3. **Polymer Tg smoking gun:** R²_max(SMILES) vs published R², with physical explanation
4. **Alloy yield strength jump:** R²_ceiling(composition) vs R²_ceiling(comp + grain size)
5. **Summary table:** R²_ceiling vs best published R² for all case studies. Flag cases where published > ceiling.

## Code Architecture

```
fano-limits/
├── data/                  # Raw and processed datasets
│   ├── doyle/
│   ├── polymer_tg/
│   ├── mpea/
│   └── eln/
├── src/
│   ├── information.py     # Core: entropy, MI, Fano, R²_ceiling, bootstrap
│   ├── data_loaders.py    # One loader per dataset
│   └── plotting.py        # Publication figure utilities
├── notebooks/             # One notebook per case study
├── tests/                 # Unit tests for information.py
├── figures/               # Publication figures
└── paper/                 # LaTeX source
```

### src/information.py must include:
- `entropy(p)` — Shannon entropy from probability vector, base 2
- `conditional_entropy(X, Y)` — H(X|Y) with Miller-Madow correction
- `mutual_information(X, Y)` — I(X;Y) = H(X) - H(X|Y)
- `fano_bound(H_cond, M)` — both weak and tight (numerically inverted)
- `predictability(H_cond, M)` — Π = 1 - P_e*
- `within_group_variance(values, groups)` — Var(X|Y) by averaging within-group variances
- `r2_ceiling(values, groups)` — 1 - Var(X|Y)/Var(X)
- `fano_bound_noisy(H, M, epsilon)` — corrected for label noise
- `bootstrap_ci(values, groups, func, n_boot=1000, ci=0.95)` — confidence intervals

All functions use log base 2 (bits). All support numpy arrays. All have docstrings explaining the math.

### Unit tests must verify:
- Deterministic case: H(X|Y) = 0 when Y determines X
- Independent case: H(X|Y) = H(X) when X ⊥ Y
- Known analytical case with computed answer
- Miller-Madow correction magnitude
- R²_ceiling = 1.0 when each group has zero variance
- R²_ceiling = 0.0 when groups are random (permutation test)

## Common Pitfalls to Avoid

1. **Don't use KSG estimator as the primary result.** It's biased, high-variance, and reviewer-bait. Use it only as a secondary check on categorical/binning results.

2. **Don't claim "ML can't work."** The message is "ML has reached the ceiling FOR THESE DESCRIPTORS." Better descriptors (more variables, finer measurements) can raise the ceiling.

3. **Don't confuse R²_ceiling with model performance.** R²_ceiling is an UPPER BOUND on what any model can achieve. If a model's R² exceeds R²_ceiling, something is wrong with the evaluation, not with the bound.

4. **Don't forget that within-group variance from deduplicated data is zero.** Many datasets average over replicates. Check for duplicates FIRST. If all SMILES are unique, within-SMILES variance is undefined and you need fingerprint grouping instead.

5. **Don't overstate generality.** Every bound is specific to a dataset. Say "on this benchmark" not "in chemistry."

6. **Don't use Fano's inequality for continuous outcomes without binning.** Fano requires discrete X. For continuous X, use R²_ceiling directly.

7. **Don't compare R²_ceiling to published R² without matching evaluation protocols.** Retrain the model yourself with the same preprocessing to ensure a fair comparison.

## Decision Gates

- **Gate 1 (after Phase 2A):** Is R²_ceiling for the Doyle dataset below 0.95? If yes, informative. If no, the full factorial design with no replicates makes every condition unique → use coarser groupings.
- **Gate 2 (after Phase 2B):** Does R²_max(SMILES) for polymer Tg fall below published R² = 0.97? If yes, headline result. If no, the dataset was likely deduplicated → try merging sources or use fingerprint grouping.
- **Gate 3 (after Phase 2C):** Does adding grain size significantly increase R²_ceiling for alloy yield strength? If yes, clean physical argument. If no, dataset may be too homogeneous in processing.

## Framing and Narrative

The paper follows the template of Rosen et al. (QMOF-Thermo, 2026): long-standing hypothesis → rigorous quantification at scale → open tool for the community.

Our hypothesis: "ML prediction accuracy in materials science is limited by descriptor informativeness."
Our evidence: R²_ceiling computations across 3-4 domains.
Our tool: The R²_ceiling diagnostic protocol + code.

The narrative arc across case studies:
1. **Doyle (organic reactions):** Descriptor saturation — the field is near the ceiling
2. **Polymer Tg:** Descriptor insufficiency — SMILES physically cannot encode Tg-determining variables; published R² may exceed the ceiling
3. **Alloy yield strength:** Microstructure gap — composition-only models miss the dominant variable (grain size), quantified via Hall-Petch
4. **HTE vs ELN:** Reporting gap — same reaction type, different data quality, dramatically different ceilings

Each case illustrates a different failure mode. The unifying message: **compute R²_ceiling before publishing your ML model.**
