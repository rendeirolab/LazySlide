from cyclopts import App
from huggingface_hub import HfApi, create_tag, delete_tag, get_dataset_tags

app = App()


@app.command
def list(dataset: str):
    """List git tags on a dataset repo.

    Example:
        python -m scripts.tag_datasets list rendeirolab/lazyslide-data
    """
    api = HfApi()
    refs = api.list_repo_refs(dataset, repo_type="dataset")
    tags = [t.name for t in refs.tags]
    print(tags)


@app.command
def create(dataset: str, tag: str, revision: str = "main"):
    """Create a git tag on a dataset repo pointing to a revision (branch name or commit id)."""
    create_tag(repo_id=dataset, tag=tag, revision=revision, repo_type="dataset")
    print(f"Created tag '{tag}' -> {revision} on dataset '{dataset}'")


@app.command
def delete(dataset: str, tag: str):
    """Delete a git tag from a dataset repo."""
    delete_tag(repo_id=dataset, tag=tag, repo_type="dataset")
    print(f"Deleted tag '{tag}' from dataset '{dataset}'")


if __name__ == "__main__":
    app()
