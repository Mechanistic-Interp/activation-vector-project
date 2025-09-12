# Cursor Notes

## Chat & AI Features
- **Cmd+l** - Opens Chat (Agent mode, Ask Mode, More Modes)
- **Cmd+k** - Inline ask (quick AI assistance)
- **Cmd+n** - New chat
- **@** - Reference files, code, or docs in chat
- **Auto mode** - Cost efficient (Claude costs more)
- **Cmd+t** - New tab

## Navigation & Productivity
- **Tab** - Auto completion suggestions
- **Indexing & Docs** - Super useful for learning new libraries

## Configuration
- **.cursorrules** - Put rules for Cursor to always follow


# Modal Notes

## Getting Started
- python -m venv env 
- source env/bin/activate

- Install: `pip install modal`
- Sign up at modal.com and get API token
- Run `modal token new` to authenticate
- Basic structure: `modal.App("app-name")` creates your app

## Core Concepts
- **Functions**: `@app.function()` - runs code on Modal's cloud
- **Classes**: `@app.cls()` - persistent stateful containers
- **Images**: `modal.Image.debian_slim()` - define your environment
- **Volumes**: `modal.Volume.from_name()` - persistent storage
- **Secrets**: `modal.Secret.from_name()` - store API keys securely

## Essential Decorators
- `@modal.enter()` - runs once when container starts (setup code)
- `@modal.method()` - callable methods on classes
- `@modal.fastapi_endpoint()` - create web APIs
- `@app.local_entrypoint()` - run locally for testing

## Common Patterns
- **GPU access**: Add `gpu="A100-80GB"` to function/class
- **Memory**: Set `memory=65536` (in MB)
- **Timeout**: `timeout=900` (in seconds)
- **Caching**: Use volumes for model weights, datasets
- **Snapshots**: `enable_memory_snapshot=True` for faster cold starts

## Key Commands
- `modal run -m package.module` - run a module
- `modal deploy -m package.module` - deploy to cloud
- `modal app list` - see your apps
- `modal app logs <app-name>` - view logs
- `modal volume list` - see volumes

## Best Practices
- Use volumes for large files (models, datasets)
- Enable snapshots for expensive setup (model loading)
- Set appropriate timeouts and memory limits
- Use secrets for API keys, not hardcoded values
- Test locally before deploying

## Common Gotchas
- Import statements go inside functions, not at module level
- Use `.remote()` to call Modal functions from local code
- GPU memory is expensive - be mindful of usage
- Cold starts can be slow without snapshots
- Always handle exceptions in Modal functions

# Pythia Notes/Transformer Lens Notes
