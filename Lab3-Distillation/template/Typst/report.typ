#import "@preview/kunskap:0.1.0": *

#set math.mat(delim: "[")
#set math.vec(delim: "[")

#set page(footer: context align(center, text(size: 8pt, fill: rgb("#888888"), counter(page).display())))

#show: kunskap.with(
  title: "Lab3: Score Distillation Sampling Implementation",
  author: "Your Name (Student ID)",
  header: "NYCU - Image and Video Generation, Lab3",
  paper-size: "a4",
)

// ---------------------------------------------------------------------------
// Figures: copy outputs/{sds,vsd,sdi}/ (images + eval.json) into figures/{sds,vsd,sdi}/.
// draft = true draws dashed placeholder boxes and "–" for CLIP scores, so the file
// compiles before any result exists. Set it to false once figures/ is filled.
// ---------------------------------------------------------------------------
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set par(justify: false)
#show table: set text(hyphenate: false)

// Tighter spacing than kunskap's defaults (gap above headings: 6% / 4% of the page; paragraphs: 1.7em)
#set par(spacing: 1.2em)
#show heading.where(level: 1): it => v(-20pt) + it
#show heading.where(level: 2): it => v(-14pt) + it

#let draft = true

#let img(path) = if draft {
  layout(s => rect(width: s.width, height: s.width, stroke: (paint: gray, dash: "dashed"),
    align(center + horizon, text(s.width / 12, fill: gray, path.replace("/", "/\u{200B}").replace("_", "_\u{200B}")))))
} else { image(path, width: 100%) }

#let todo(body) = text(fill: gray, body)  // gray hint: what to write here

#let prompts = (
  "A red bus driving on a desert road",
  "a boat in a river",
  "A cabin surrounded by forests",
  "A church beside a lake",
  "A villa close to the pool",
  "A castle next to a river",
  "A burger on the table",
  "A dog sitting on grass",
  "a cat sitting on a table",
  "A car on the road",
)
#let key(p) = p.replace(" ", "_")
#let short = ("Bus", "Boat", "Cabin", "Church", "Villa", "Castle", "Burger", "Dog", "Cat", "Car")  // CLIP table column labels
#let methods = (("sds", "SDS"), ("vsd", "VSD"), ("sdi", "SDI"))

// CLIP score read from figures/<dir>/eval.json (written by eval.py); p = none -> average.
#let clip(dir, p: none) = if draft { [–] } else {
  let d = json("figures/" + dir + "/eval.json")
  let k = if p == none { "score" } else { d.keys().find(k => k.ends-with("/" + key(p) + ".png")) }
  let s = str(calc.round(d.at(k), digits: 4)) + "0000"  // pad to 4 decimals
  [#s.slice(0, s.position(".") + 5)]
}

// rows = methods, columns = the 10 prompts
#let result-grid = grid(
  columns: (auto,) + (1fr,) * prompts.len(),
  gutter: 2pt,
  align: center + horizon,
  ..methods.map(((dir, name)) =>
    (rotate(-90deg, reflow: true, text(7pt, name)),) + prompts.map(p => img("figures/" + dir + "/" + key(p) + ".png"))
  ).flatten(),
)

// Ablation grid: figures/ablation/<setting>/<value>/<prompt>.png, 3 prompts (rows) x 5 values (columns)
#let abl-cell = 11%  // ablation image size; the diff-loss figure uses 1.5x so both grids are equally tall
#let abl-prompts = (("A burger on the table", "Burger"), ("a cat sitting on a table", "Cat"), ("A villa close to the pool", "Villa"))  // (prompt, row label)
#let abl-figure(dir, values, name) = figure(
  grid(
    columns: (auto,) + (abl-cell,) * values.len(),
    gutter: 2pt,
    align: center + horizon,
    [], ..values.map(v => text(6pt, v)),
    ..abl-prompts.map(((p, s)) => (rotate(-90deg, reflow: true, text(7pt, s)),) + values.map(v => img("figures/ablation/" + dir + "/" + v + "/" + key(p) + ".png"))).flatten(),
  ),
  kind: image,
  caption: [*Ablation on #name.*],
)

#let booktab(columns, align, header, ..rows) = table(
  columns: columns, align: align, stroke: none, inset: (x: 5pt, y: 3pt),
  table.hline(), table.header(..header.map(h => [*#h*])), table.hline(stroke: 0.5pt),
  ..rows, table.hline(),
)

#v(-3em)  // kunskap leaves 5em below the title block
#figure(
  image("figures/teaser.png", width: 100%),
  placement: none,
  caption: [*Overview of score distillation.* #todo[Describe it briefly; cite its source if not yours.]],
) <fig-teaser>

// ===========================================================================
= *Introduction*
// [2 pts] Briefly describe and summarize the goal of this assignment.

#todo[What problem does score distillation address, and how is it different from ordinary diffusion sampling? Keep it to a few sentences; you may refer to @fig-teaser.]

#todo[State the goal of this assignment: which methods you implemented, what you generated, and how you evaluated the results.]

#todo[(Optional) Briefly outline the rest of the report.]

// ===========================================================================
= *Methodology*
// [6 pts] Explain SDS, VSD, and SDI; both the concept and your implementation.

#todo[Introduce the background and notation shared by the three methods (e.g. the pretrained diffusion model [6], the forward noising process [4], classifier-free guidance [9]), so the following subsections can refer to them.]

== *1. Score Distillation Sampling (SDS)*

*Concept.* #todo[In your own words: what is being optimized in SDS [1], where does the gradient come from, and why is it a reasonable objective? Point out the approximation that makes SDS practical.]

*Implementation.* #todo[Paste your `get_sds_loss()` into the code block below, then walk through it: the main steps, the design choices you made and why, and anything that was tricky to get right.]

```python
def get_sds_loss(
    self, latents, text_embeddings,
    guidance_scale=7.5,
):
    # Paste your implementation here
    ...
```

== *2. Variational Score Distillation (VSD)*

*Concept.* #todo[Which weakness of SDS does VSD [2] try to fix? What extra component does it introduce (cf. LoRA [7]), and what is that component trained to do?]

*Implementation.* #todo[Paste your `get_vsd_loss()` into the code block below, then walk through it: how the extra component is set up and updated, how its loss and the distillation loss interact, and which parts receive gradients.]

```python
def get_vsd_loss(
    self, latents, text_embeddings,
    guidance_scale=7.5, lora_loss_weight=1.0,
):
    # Paste your implementation here
    ...
```

== *3. Score Distillation via Inversion (SDI)*

*Concept.* #todo[Which weakness of SDS does SDI [3] try to fix? How does it obtain the noisy sample differently (cf. DDIM [5]), and how does its timestep schedule differ from SDS?]

*Implementation.* #todo[Paste your `invert_noise()` (DDIM inversion) and `get_sdi_loss()` into the two code blocks below, then walk through them: how your inversion loop works, how the timestep changes over the iterations, and how and when the target is updated.]

```python
@torch.no_grad()
def invert_noise(
    self, latents, target_t, text_embeddings,
    guidance_scale=-7.5, n_steps=10, eta=0.3,
):
    # Paste your DDIM inversion here
    ...
```

```python
def get_sdi_loss(
    self, latents, text_embeddings,
    guidance_scale=7.5,
    current_iter=0, total_iters=500,
    inversion_guidance_scale=-7.5,
    inversion_n_steps=10, inversion_eta=0.3,
    update_interval=25,
):
    # Paste your implementation here
    ...
```

// ===========================================================================
= *Experiments & Results*
// [6 pts] Quantitative results (CLIP score) + qualitative results (output images).
// [6 pts] Ablation: (1) SDS guidance_scale; (2) SDI inversion_n_steps; (3) SDI sdi_update_interval.
// [2 pts] All hyperparameters for each task.

== *1. Quantitative Results*

#todo[Which method has the best average CLIP score (@tab-clip)? Are there prompts where the ranking changes, and why might that be? Does the score agree with your visual impression?]

#figure(
  booktab((auto,) * 12, center, ([Method], ..short.map(s => [#s]), [Avg.]),
    ..methods.map(((dir, name)) => ([*#name*],) + prompts.map(p => clip(dir, p: p)) + (clip(dir),)).flatten(),
  ),
  caption: [*CLIP score* ($arrow.t$, ViT-B/32) per prompt.],
) <tab-clip>

== *2. Qualitative Results*

#todo[Compare the methods visually and point to concrete examples in @fig-qualitative. What are the typical strengths and failure cases of each method?]

#figure(
  result-grid,
  caption: [*Qualitative comparison* of SDS, VSD and SDI on the 10 evaluation prompts.],
) <fig-qualitative>

== *3. Ablation Study*

#todo[Briefly describe the setup: which prompts and values were tested; all other hyperparameters follow @tab-hparams. @tab-ablation lists the CLIP scores of all three ablations.]

#figure(
  table(
    columns: 6, align: (left,) + (center,) * 5, stroke: none, inset: (x: 5pt, y: 3pt), column-gutter: 0.8em,
    table.hline(),
    table.cell(colspan: 6)[_(a) SDS: `guidance_scale`_],
    [Value], [7.5], [15], [25], [50], [100],
    [CLIP], [0.0000], [0.0000], [0.0000], [0.0000], [0.0000],
    table.hline(stroke: 0.5pt),
    table.cell(colspan: 6)[_(b) SDI: `inversion_n_steps`_],
    [Value], [1], [3], [5], [10], [25],
    [CLIP], [0.0000], [0.0000], [0.0000], [0.0000], [0.0000],
    table.hline(stroke: 0.5pt),
    table.cell(colspan: 6)[_(c) SDI: `sdi_update_interval`_],
    [Value], [25], [75], [200], [500], [1000],
    [CLIP], [0.0000], [0.0000], [0.0000], [0.0000], [0.0000],
    table.hline(),
  ),
  caption: [*Ablation study.* Average CLIP score ($arrow.t$).],
) <tab-ablation>

=== *SDS: `guidance_scale`*

#todo[How do the results change as the guidance scale increases (@fig-abl-sds-gs, @tab-ablation (a)), and why? Which value works best, and how does it compare with the guidance scale typically used for ordinary sampling?]

#abl-figure("sds_guidance_scale", ("7.5", "15", "25", "50", "100"), [SDS `guidance_scale`]) <fig-abl-sds-gs>

=== *SDI: `inversion_n_steps`*

#todo[How do the results and the runtime change with the number of inversion steps (@fig-abl-sdi-steps, @tab-ablation (b)), and why? Is there a trade-off?]

#abl-figure("sdi_inversion_n_steps", ("1", "3", "5", "10", "25"), [SDI `inversion_n_steps`]) <fig-abl-sdi-steps>

=== *SDI: `sdi_update_interval`*

#todo[How do the results change with the update interval (@fig-abl-sdi-interval, @tab-ablation (c)), and why? What happens at the two extremes?]

#abl-figure("sdi_update_interval", ("25", "75", "200", "500", "1000"), [SDI `sdi_update_interval`]) <fig-abl-sdi-interval>

== *4. Hyperparameters*

#todo[Fill in @tab-hparams with the settings you actually used (defaults are pre-filled from `main.py` / README) and justify any value you changed.]

#figure(
  booktab((auto, auto, auto, auto), (left, center, center, center), ([Hyperparameter], [SDS], [VSD], [SDI]),
    [`guidance_scale`], [25], [7.5], [7.5],
    [`lr`], [0.01], [0.01], [0.005],
    [`steps`], [500], [500], [1000],
    table.hline(stroke: 0.5pt),
    [`lora_lr`], [–], [1e-4], [–],
    [`lora_rank`], [–], [4], [–],
    [`lora_loss_weight`], [–], [1.0], [–],
    table.hline(stroke: 0.5pt),
    [`inversion_n_steps`], [–], [–], [10],
    [`inversion_guidance_scale`], [–], [–], [-7.5],
    [`inversion_eta`], [–], [–], [0.3],
    [`sdi_update_interval`], [–], [–], [25],
  ),
  caption: [*Hyperparameters* for each task.],
) <tab-hparams>

// ===========================================================================
= *Discussion*

== *1. Why the Original Diffusion Loss Fails*
// [3 pts] Intuitively explain why L_Diff = ||eps_hat(x_t, y, t) - eps_t||^2 did not produce
// realistic samples (use < 1 math equation) & provide visual results of applying L_Diff.

#let diff-prompts = ("A dog sitting on grass", "A castle next to a river", "A car on the road", "A church beside a lake")
#todo[Explain how you produced the result in @fig-diff-loss. Then explain intuitively (at most one equation) why directly optimizing the diffusion loss does not give realistic images, and connect your explanation to what is visible in the figure.]

#figure(
  grid(
    columns: (auto,) + (1.5 * abl-cell + 1pt,) * diff-prompts.len(),
    gutter: 2pt,
    align: center + horizon,
    [], ..diff-prompts.map(p => text(6pt, p)),
    rotate(-90deg, reflow: true, text(7pt, "Diff. loss")), ..diff-prompts.map(p => img("figures/diff_loss/" + key(p) + ".png")),
    rotate(-90deg, reflow: true, text(7pt, "SDS")), ..diff-prompts.map(p => img("figures/sds/" + key(p) + ".png")),
  ),
  caption: [*Original diffusion loss* (top) vs. SDS (bottom).],
) <fig-diff-loss>

== *2. Why SDI and VSD Outperform SDS*
// [3 pts] Intuitively analyze why SDI and VSD perform better than SDS (without heavy math).

*SDS.* #todo[What are the typical artifacts of SDS in your results, and what in the SDS update causes them?]

*VSD.* #todo[What does VSD change compared with SDS, and why does that lead to better results? Explain intuitively, without heavy math, and support it with your own figures or numbers.]

*SDI.* #todo[What does SDI change compared with SDS, and why does that lead to better results? Explain intuitively, without heavy math, and support it with your own figures or numbers.]

// ===========================================================================
= *Conclusion*
// [2 pts] Wrap up all the context in your report and give a short summary.

#todo[Summarize what you implemented, the main quantitative and qualitative findings, the key lessons from the ablations, and your answer to why VSD and SDI outperform SDS.]

// ===========================================================================
= *Reference*

#text(size: 8pt)[
  #set enum(numbering: "[1]", spacing: 0.9em)
  + B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. DreamFusion: Text-to-3D using 2D Diffusion. In _Int. Conf. on Learning Representations (ICLR)_, 2023.
  + Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2023.
  + A. Lukoianov, H. Sáez de Ocáriz Borde, K. Greenewald, V. C. Guizilini, T. Bagautdinov, V. Sitzmann, and J. Solomon. Score Distillation via Reparametrized DDIM. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2024.
  + J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2020.
  + J. Song, C. Meng, and S. Ermon. Denoising Diffusion Implicit Models. In _Int. Conf. on Learning Representations (ICLR)_, 2021.
  + R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. In _Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)_, 2022.
  + E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. In _Int. Conf. on Learning Representations (ICLR)_, 2022.
  + A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning Transferable Visual Models From Natural Language Supervision. In _Int. Conf. on Machine Learning (ICML)_, 2021.
  + J. Ho and T. Salimans. Classifier-Free Diffusion Guidance. _arXiv preprint arXiv:2207.12598_, 2022.
]
