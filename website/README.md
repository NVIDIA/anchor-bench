# Anchor Bench website

The public site for Anchor Bench, built with [Hugo](https://gohugo.io/) and
deployed to GitHub Pages at **https://nvidia-dev.github.io/anchor-bench/**.

## Structure

```text
website/
  hugo.toml              Site config, theme cascade, and content mounts
  content/
    blog/                Blog posts (one markdown file per post)
    documentation/       Documentation landing page (_index.md)
  data/
    documentation.yaml   Titles/descriptions/order for the docs table of contents
  layouts/               HTML templates (home, blog, documentation, partials)
  static/
    css/main.css         Theme styles (Light / Dark / Novel)
    js/theme.js          Theme switcher logic
```

The actual documentation pages are **not** stored under `website/`. The repo's
top-level [`docs/`](../docs) folder is mounted into the site at build time (see
`[[module.mounts]]` in `hugo.toml`), so every `docs/*.md` file is published at
`/documentation/<name>/`. This keeps a single source of truth: the same
markdown is read on GitHub and rendered on the site.

## Prerequisites

Hugo (extended not required). Install the pinned version with Go:

```bash
go install github.com/gohugoio/hugo@v0.147.9
```

or use Homebrew: `brew install hugo`.

## Local preview

```bash
cd website
hugo server
```

Then open the URL Hugo prints, which includes the project base path:

**http://localhost:1313/anchor-bench/**

> Note: the site is served under `/anchor-bench/` (matching the GitHub Pages
> project URL). Visiting `http://localhost:1313/` without that path will 404 and
> make links look broken — always use the `/anchor-bench/` URL.

## Add a blog post

1. Create `website/content/blog/my-post-title.md`.
2. Add front matter with a title and date:

   ```markdown
   ---
   title: My post title
   date: 2026-07-13
   ---

   Your content here. The first paragraph is used as the preview/summary
   on the home page and blog index.
   ```

3. Save. The post automatically appears as the "Latest update" on the home page,
   as the featured post on `/blog/`, and in the "All posts" list (sorted newest
   first).

Links inside posts that point to other site pages should start with `/`, for
example `[CLI docs](/documentation/cli/)`.

## Add a documentation page

Documentation lives in the repo's top-level `docs/` folder and is surfaced on the
site through the table of contents.

1. Add your markdown file to [`../docs`](../docs), e.g. `docs/my-guide.md`.
   Start it with a single H1 heading (`# My guide`) — no front matter needed.
2. Add an entry to `website/data/documentation.yaml` so it shows on the
   `/documentation/` page:

   ```yaml
   - title: My guide
     path: documentation/my-guide/
     description: One-line summary shown on the documentation card.
   ```

   The `path` is `documentation/<filename-without-extension>/`. Entries render on
   the documentation page in the order listed here.
3. Save and preview. The page is published at `/documentation/my-guide/` and the
   card links to it.

> The `data/documentation.yaml` file controls the titles, descriptions, and order
> of the cards. A doc file added to `docs/` without a matching entry is still
> published but won't appear in the table of contents.

## Deploy to GitHub Pages

Deployment is automated by [`.github/workflows/github-pages.yml`](../.github/workflows/github-pages.yml),
which builds the site with Hugo and publishes it.

One-time repository setup:

1. Go to **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.

After that, every push to `main` (or the `ggallagher/github-pages` branch) builds
and deploys automatically. You can also trigger it manually from the
**Actions** tab (the workflow supports `workflow_dispatch`). The published URL
is shown in the workflow's `deploy` step and at
https://nvidia-dev.github.io/anchor-bench/.

### Notes on the base URL

`baseURL` in `hugo.toml` is set to the project Pages URL
(`https://nvidia-dev.github.io/anchor-bench/`). If the repository is ever renamed
or moved to a user/org root or a custom domain, update `baseURL` accordingly so
asset and page links resolve correctly.

## Themes

The header has a Light / Dark / Novel switcher. The selection is stored in
`localStorage` and applied before first paint. Colors are defined as CSS
variables in `static/css/main.css` under `:root`/`[data-theme="light"]`,
`[data-theme="dark"]`, and `[data-theme="novel"]`. You can also force a theme via
the `?theme=dark` (or `light`/`novel`) query parameter.
