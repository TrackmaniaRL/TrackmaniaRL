# TrackmaniaRL article

The English article is maintained on `codex/library-article`, created from
`origin/main` at `556adeefdb401dae5d211e836f44ea734a48aa05`.

The public site is https://trackmaniarl.github.io/TrackmaniaRL/.
GitHub Pages serves the root of the `gh-pages` branch. That branch contains only
the static contents of `website/`. Publication does not require merging the
article into the library's main branch.

## Preview

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory website
```

The site uses relative paths, local assets, native MP4 controls and `.nojekyll`.
There is no backend, analytics or remote font dependency.

## Updating the published site

After reviewing and committing changes on the article branch:

```powershell
git push origin codex/library-article
git subtree split --prefix website -b codex/article-pages
git push origin codex/article-pages:gh-pages
```

Use a fresh local split branch name if `codex/article-pages` already exists.
Keep updates as normal fast-forward pushes. GitHub Pages rebuilds after the
`gh-pages` branch changes.

## Evidence and graphics

`evidence.html` documents chart selection, media provenance and paper credits.
The site exports only the numeric evaluation and benchmark data needed for its
charts. Raw experiment inventories, account links and local research notes are
excluded. The reward section distinguishes online training, offline re-scoring
and implemented options that were inactive in the confirmed controller.

Original IQN, TQC and SimbaV2 figures retain their author, source and CC BY 4.0
credits. Architecture diagrams and charts are SVG. `scripts/article_figures.py`
regenerates the development chart and illustrative finish-reward chart using
Matplotlib. The original-resolution gameplay files use native MP4 playback.

## Mathematical notation and exact inputs

Reward equations and inline symbols are authored in `scripts/article-math.json`.
The article includes pre-rendered HTML and accessible MathML, generated with
KaTeX 0.19.0. Its stylesheet and WOFF2 fonts are bundled under `assets/katex/`,
with the original MIT license. No client-side math renderer or CDN is required.
To re-render after editing the LaTeX sources, install that KaTeX version in a
local build directory and run:

```powershell
node scripts/render_article_math.cjs PATH_TO_KATEX/dist/katex.js
```

The article describes the four input groups in prose, including road sampling
every 2.5 metres. `assets/model-inputs.json` preserves every feature in tensor
order with its scaling and clipping. The data notes link that specification and
keep the complete benchmark rows and reward implementation details.
`assets/model-architecture.svg` is editable vector source for the diagram.

## Reading themes

Dark mode is the default on the article and data notes. The header button switches
between dark and light, with the choice saved locally in the browser. Storage
restrictions do not prevent switching during the current visit. Print styles use
the light palette and original light diagrams. Paper figures retain their source
colors in both themes.

After regenerating any original article SVG, rebuild its dark counterpart:

```powershell
node scripts/article_themes.cjs
```

The theme generator changes paint values only. Chart measurements, text and
geometry stay the same. Both image variants use CSS visibility so the saved theme
applies without a diagram flash or reliance on JavaScript to display the figures.

## Illustrated learning loop

`assets/rl-learning-dark.png` and `assets/rl-learning-light.png` replace the
original box diagram in the article. Both are 1536 by 1024 pixels, generated
with the built-in GPT Image tool. The exact prompts are preserved in
`scripts/rl-learning-prompts.json`. The original SVG stays available as source
material. The six connections were checked in both variants before publication.
The illustrated game scenes and telemetry are not experimental evidence.

## October QR-DQN update

The October section links the matched training episode (policy version 1841885), original QR configuration and all 30 final-checkpoint evaluation times. The filmed policy weights were not retained; distributed-update-02045500.pt is the later final checkpoint and must not be presented as the filmed weights. Recording provenance documents the 1080p source upscaled to a 1440p/60 fps container.
