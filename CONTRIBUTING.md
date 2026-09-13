# Contributing

Thanks for taking a look! These are the house rules for this repository (and the sibling
ShadowRoom plugin repositories).

## README style

- `README.md` is **English** and written for users: what it is, features, requirements, install,
  configuration, tools, usage, licensing/safety, troubleshooting, license. Nothing else.
- **No implementation details in the README** — adapter contracts, job-file fields, code walkthroughs
  and "how we figured it out" stories go to `docs/internals.md`.
- Keep `README.zh-CN.md` as a Chinese translation with the same structure and headings.
- Prefer tables for options/tools and keep every code block copy-pasteable.

## Commits

- Prefix with a scope, e.g. `feat(generator): …`, `fix(jobs): …`, `docs(repo): …`,
  `chore(repo): …`.
- One logical change per commit.

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Licensing

This project is released under AGPL-3.0. By contributing you agree that your work is released
under the same license. Upstream model weights keep their own licenses.
