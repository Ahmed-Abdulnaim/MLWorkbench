# Go live — the five things only the repo owner can click

Everything in the code is done and merged. These steps need owner-level permissions
(the automation token used to open the PR is not allowed to change repo settings), and each
one takes under a minute.

## 1. Turn on GitHub Pages  ← do this first

**Settings → Pages → Build and deployment**

- Source: **Deploy from a branch**
- Branch: **`main`**, folder: **`/site`**
- Save

About a minute later the page is live at:

```
https://ahmed-abdulnaim.github.io/MLWorkbench/
```

The site is a plain static folder — no build step, no Jekyll config needed, images included.
If it 404s for a while that is normal propagation; check the **Actions → pages build and
deployment** run for status.

## 2. Point the repo at the site

**Settings → General → Description / Website**, or the gear icon next to *About* on the repo
home page:

- **Website:** `https://ahmed-abdulnaim.github.io/MLWorkbench/`
- **Description:** already good, optionally add *"Free · offline · MIT"* at the end.
- **Topics:** already set (`machine-learning`, `cheminformatics`, `qsar`, `no-code`, …).
- Tick **Releases** and **Packages** in the About panel so the download is one click from the
  repo page.

## 3. Swap the contact address

The three pricing buttons on the landing page currently open a `mailto:` to the GitHub
noreply address (`site/index.html`). It works, but a real inbox converts better:

```bash
grep -n "mailto:" site/index.html
```

Replace `78787786+Ahmed-Abdulnaim@users.noreply.github.com` with a business address you are
happy to publish, then commit to `main`. If you would rather not publish an address, create a
free form (Tally, Google Forms) and point the buttons at that instead.

## 4. Apply for GitHub Sponsors

**<https://github.com/sponsors/accounts>** → *Join the waitlist* / *Set up Sponsors*.

- Requires: 2FA enabled, a supported payout region (Egypt **is** supported), and a bank
  account whose country matches your residence.
- Review usually takes a few days and is not automatic.
- `.github/FUNDING.yml` already points at your account, so the **Sponsor** button appears the
  moment you are approved — nothing further to change.
- Until approval the button 404s harmlessly. Delete `FUNDING.yml` if you would rather not show
  a dead link in the meantime.

## 5. Rebuild the exe through the new workflow (optional but recommended)

The shipped `v1.0.0` binary was cross-built under Wine on Linux, which forced `numpy < 2`.
The new workflow builds on a real Windows runner with no such constraint and **self-tests the
binary before publishing**:

**Actions → Build Windows exe → Run workflow**

Then either download the artifact, or cut a real release:

```bash
git tag v1.0.1 && git push origin v1.0.1
```

That attaches the fresh exe, its SHA-256, and the user guide to a new Release automatically.
You will need to update the hash printed in `README.md` (it is the v1.0.0 hash).

---

## After that: the only thing left is distribution

The bottleneck stops being the product and becomes attention. `docs/MONETIZATION.md` has the
copy and the ordering — the short version:

1. Post the Show HN draft on a weekday morning, US time, and stay available to reply.
2. Answer five real questions where this tool genuinely applies (search terms are in the doc).
3. One LinkedIn post aimed at a specific industry — process engineering, QC labs, formulation.
4. Put the pricing in front of people who already downloaded it.

Track one number for the first month: **downloads per release**, then **enquiries per
download**. Everything else is noise until those move.
