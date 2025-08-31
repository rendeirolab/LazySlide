# Release cycle of LazySlide

1. Decide on the release version, lazyslide uses [semantic versioning](https://semver.org/).
2. Make a release for [lazyslide-tutorial](https://github.com/rendeirolab/lazyslide-tutorials)
3. Make a release for lazyslide

## How to make a release

Notice that a release can only be made by the repository owner.

1. Got to [releases](https://github.com/rendeirolab/lazyslide/releases) and click on "Draft a new release".
2. Make a new tag with the version number, for example `v0.8.2`.
3. Add the release title, usually the same as the tag.
3. Fill the release notes.
4. Click on "Publish release".

## Checklist before the release

- Any updates from the upstream library should we include? Especially wsidata, huggingface and pytorch.
- Are all the tests passing?
- Is the documentation up to date?
- Is the tutorial up to date and tagged to the release version?
