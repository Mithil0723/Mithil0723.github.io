# Website Makeover Brief — mithil0723.github.io

Implementation brief for the redesign agent (Google Antigravity). Repo: `Mithil0723/Mithil0723.github.io`.
Stack stays vanilla HTML/CSS/JS — no framework migration. All work happens in `index.html`, `assets/css/style.css`, `assets/css/modal.css`, `assets/js/script.js`, `assets/js/projectDetails.js`.

---

## 1. Design direction

**Essence: warm business × tech hybrid.** Think "editorial consulting firm that ships software" — warm paper tones and confident typography (business), paired with monospace details, precise grids, and a restrained accent system (tech). Reference points: Anthropic's parchment-and-terracotta editorial style, Stripe's typographic discipline, Brittany Chiang's v4 portfolio structure.

**What this replaces:** the current dark glassmorphism theme (mesh blobs, noise overlay, blue/violet/teal gradients, blur layers). Remove `.mesh-bg`, `.mesh-blob-*`, `.noise-overlay`, and all `--glass-*` tokens.

---

## 2. Color system (eye-comfort first)

Principles: no pure white, no pure black, low blue-light bias, all text pairs meet WCAG AA (ratios verified below).

### Light theme (default)

| Token | Hex | Role | Contrast on bg |
|---|---|---|---|
| `--bg` | `#FAF8F5` | Warm paper background (not #FFF) | — |
| `--bg-raised` | `#F3EFE9` | Cards, section alternation | — |
| `--border` | `#E8E2DA` | Hairline borders, dividers | decorative |
| `--ink` | `#2A2521` | Headings/body (warm near-black) | 14.3:1 |
| `--ink-muted` | `#59524A` | Secondary text, captions | 7.3:1 |
| `--accent` | `#B4552D` | Terracotta — buttons, highlights | 4.6:1 (AA) |
| `--accent-strong` | `#9C4526` | Links, small accent text | 6.0:1 |
| `--tech` | `#1F4E5F` | Deep slate-teal — tags, code, mono details | 8.6:1 |

### Dark theme (optional toggle, `prefers-color-scheme` aware)

| Token | Hex | Role | Contrast on bg |
|---|---|---|---|
| `--bg` | `#201B17` | Warm charcoal (not blue-black) | — |
| `--text` | `#D9CFC4` | Warm off-white body | 11.1:1 |
| `--accent` | `#E07A4F` | Soft terracotta | 5.7:1 |
| `--tech` | `#7FB5C4` | Muted teal | 7.6:1 |

Rules: accent used sparingly (one accent element per viewport-height of content); large flat areas stay in `--bg`/`--bg-raised`; never place saturated accent text on dark backgrounds at small sizes.

## 3. Typography

- **Headings:** `Fraunces` (Google Fonts, optical-size axis) — warm, editorial, business-serif. Weight 600, tight tracking.
- **Body:** `Inter` — replaces Open Sans/Poppins. 1rem/1.65 line-height, `--ink`.
- **Mono details:** `JetBrains Mono` for section numbers ("01."), tags, tech-stack chips — this is the "tech" texture.
- Keep the existing `clamp()` fluid sizes; they're fine.

## 4. Mouse feature removal (required)

Remove the magnetic cursor entirely:

- `index.html` — delete `<div class="magnetic-cursor">` and `<div class="magnetic-cursor-follower">` (lines ~483–484).
- `assets/js/script.js` — delete the whole cursor block (~lines 46–90: element lookups, `mousemove` listener, RAF loop, hover class toggles).
- `assets/css/style.css` — delete `.magnetic-cursor*` rules (~lines 1422–1467), including the `cursor: none` override so the native cursor returns.

Also recommended (same category of gimmick): the 3D card-tilt `mousemove` handler at `script.js` ~line 150. Replace with a simple `translateY(-4px)` + shadow on `:hover`. Wrap any remaining animation in `@media (prefers-reduced-motion: reduce)` guards.

## 5. Creative recommendations (my additions)

1. **Kill the background video.** `project images/Background_video.mp4` (~12 MB) is the single biggest performance cost on a GitHub Pages site. The warm paper background needs no video. Biggest single win of the makeover.
2. **Hero rewrite.** Replace the centered name-over-gradient with an asymmetric editorial hero: left-aligned intro ("Data scientist building agentic AI systems"), a one-line value proposition drawn from `About_me.md`, and two buttons (View projects / Resume). Small mono label above the name (`// Chicago · UIC Data Science`).
3. **Projects as case-study rows, not modal pop-ups.** The current modal (`modal.css`, `projectDetails.js`) hides your best content behind a click. Convert to alternating image/text rows — screenshot left, description right — with outcome-first copy ("Recommendation system trained on N ratings" beats "Skills covered: ..."). Keep GitHub/Tableau links visible without opening anything. The content already exists in `project contents/`.
4. **Feature the RAG chatbot as a live demo.** You have a working backend (`backend/`, Coolify deployment). A "Chat with my portfolio" section is a stronger proof of the agentic-AI positioning than any paragraph. Give it its own section with a mono terminal-style frame — that's the tech half of the hybrid doing real work.
5. **Skills section: fewer, grouped, honest.** Three columns (ML & Data / Agentic AI & RAG / Engineering) using mono chips, instead of a long icon wall.
6. **Section numbering** (`01 — About`, `02 — Projects`…) in mono + terracotta. Cheap, distinctive, ties business typography to tech texture.
7. **Accessibility pass:** visible `:focus-visible` outlines in `--accent`, skip-to-content link, `alt` text on all project images, semantic `<section>` landmarks — quick wins Antigravity can do in one pass.

## 6. Live mockup (primary reference)

**`design-mockup.html` in this folder is the rendered target.** Open it in a browser. It demonstrates the token system, Fraunces/Inter/JetBrains Mono pairing, asymmetric hero, numbered section headings, project-row pattern, and the terminal-framed chat section. Match its look and tokens; the sections below are supporting context.

## 7. Reference repos to study (structure and taste, not to copy)

- `bchiang7/v4` — https://github.com/bchiang7/v4 — the widely praised standard for section rhythm, numbered headings, and project rows. MIT-style reuse with attribution.
- `emmabostian/developer-portfolios` — https://github.com/emmabostian/developer-portfolios — large curated list; scan for data-science portfolios specifically.
- `Evavic44/portfolio-ideas` — https://github.com/Evavic44/portfolio-ideas — curated design ideas with screenshots.
- Vercel/Stripe landing patterns — https://www.saasframe.io/examples/vercel-landing-page — for spacing scale and restrained accent use.

## 8. Out of scope

- No framework migration (stay vanilla).
- No backend changes beyond embedding the existing chatbot.
- Do not touch `wiki/` or anything outside this repo folder.

## 9. Acceptance checklist

- [ ] Magnetic cursor fully removed; native cursor restored
- [ ] No mesh blobs, noise overlay, or glass tokens in CSS
- [ ] All text passes WCAG AA on the new palette
- [ ] Background video removed; Lighthouse performance ≥ 90
- [ ] Projects readable without opening a modal
- [ ] `prefers-reduced-motion` respected
- [ ] Site deploys cleanly via existing `.github/workflows/static.yml`
