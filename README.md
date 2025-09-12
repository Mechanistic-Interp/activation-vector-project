# Activation Vector Project

End-to-end system for extracting, centering, and analyzing Pythia‑12B activation vectors using Modal. Includes utilities to compute a corpus mean, generate analysis matrices, and plot diagnostics.

This README gives newcomers a quick map of the repo, what each part does, and how to run the common workflows.

## Project Layout

```
activation-vector-project/
├── README.md
├── memories/                     # Research notes and plans
│   ├── activation_extraction_documentation.md
│   ├── activation_mean_computation_todo.md
│   ├── corpus_mean_implementation_plan.md
│   ├── gpu_snapshot_optimization.md
│   ├── implementation_plan_extract_vector_fix.md
│   ├── notes.md
│   └── pythia_12b_setup_todo.md
├── outputs/                      # Artifacts written by jobs/scripts
│   ├── activation_vector_*.safetensors
│   ├── activation_vector_*_metadata.json
│   ├── diagnostics/             # Plots and CSVs from diagnostics tools
│   │   ├── similarity_long_all_*.png
│   │   └── compare_report.csv
│   └── matrix_csv/              # Column-oriented vector CSVs (rows=dims)
│       ├── vectors_long_mode_raw.csv
│       ├── vectors_long_mode_centered.csv
│       ├── vectors_short_mode_raw.csv
│       └── vectors_short_mode_centered.csv
├── src/
│   ├── __init__.py
│   ├── corpus_mean.py           # Parallel extract + checkpointed aggregate on Modal
│   ├── cosine_similarity.py     # Compare two texts via vector cosine
│   ├── deploy.py                # Helper for deploying extractor
│   ├── extract_vector.py        # Deployed extractor (Modal class)
│   ├── generate_vector_csv.py   # Build analysis CSVs from bundled samples
│   ├── pythia_12b_modal_snapshot.py # GPU snapshot support for faster cold starts
│   ├── diagnostics/
│   │   ├── compare_vectors_local.py     # Local comparison helpers
│   │   ├── compare_vectors_with_mean.py # Compare raw/centered vs pooled mean
│   │   ├── plot_similarity_matrix.py    # Heatmaps of cosine similarity
│   │   └── visualize_vectors.py         # Rich vector distribution visualizations
│   ├── examples/
│   │   └── get_activation_vector.py     # Minimal example using modal run
│   ├── training_data/
│   │   ├── get_training_data.py         # Fetch/cache dataset samples to volume
│   │   ├── inspect_data.ipynb
│   │   ├── inspect_data.py
│   │   ├── text-samples/                # 10 long + 10 short texts used in CSVs
│   │   │   └── ...
│   │   └── training_data.pkl
│   └── utils/
│       ├── centering.py         # Mean subtraction + loader utilities
│       ├── diagnostics.py       # Core math for expected-diff checks
│       ├── io.py                # I/O helpers
│       ├── pooling.py           # Short/long pooling implementations
│       └── volume_utils.py      # Resolve latest corpus-mean path
├── visualize_vectors.py         # Standalone CSV visualizer (local)
└── visualize_vectors_simple.py  # Minimal plotting utility (local)
```

## Concepts

- Vector modes
  - `short` → 5120 dims. Mean across layers, then exp-weight across tokens.
  - `long` → 20480 dims. Concatenation of 4×5120: [last, 2.3%, 6.7%, 15.9% exp-weight].
- Centering
  - Subtracts a token‑positioned corpus mean matrix before pooling.
  - For long mode, the identity holds: `(raw_long - centered_long) ≈ pooled_mean_long`.

## Quickstart

1) Deploy the extractor (one‑time):
- `modal deploy src/extract_vector.py`

2) Fetch training data to a Modal volume (optional but recommended):
- `modal run src/training_data/get_training_data.py --num_samples 1000 --save_local`

3) Compute the corpus mean (two‑stage, detached):
- Stage 1 extract: `modal run src/corpus_mean.py extract --max_docs 1000`
- Stage 2 aggregate: `modal run src/corpus_mean.py aggregate`
- Check progress: `modal run src/corpus_mean.py check_progress`

4) Generate analysis CSVs for the bundled 20 samples:
- Raw: `modal run src/generate_vector_csv.py`
- Centered: `modal run src/generate_vector_csv.py --center`

5) Plot cosine‑similarity heatmaps from CSVs (saves under `outputs/diagnostics/`):
- Long, all chunks combined: `python -m src.diagnostics.plot_similarity_matrix --csv outputs/matrix_csv/vectors_long_mode_centered.csv --mode long`
- Long, per‑chunk: `python -m src.diagnostics.plot_similarity_matrix --csv outputs/matrix_csv/vectors_long_mode_centered.csv --mode long --per-chunk true`
- Short: `python -m src.diagnostics.plot_similarity_matrix --csv outputs/matrix_csv/vectors_short_mode_centered.csv --mode short`

6) Compare raw vs centered against pooled corpus mean (long mode):
- `modal run src/diagnostics/compare_vectors_with_mean.py --mean_path <path-to-corpus_mean.safetensors> --raw_csv outputs/matrix_csv/vectors_long_mode_raw.csv --centered_csv outputs/matrix_csv/vectors_long_mode_centered.csv --use_text_samples`
- Report is written to `outputs/diagnostics/compare_report.csv`.

7) Explore vectors locally from CSVs:
- Single file: `python visualize_vectors.py --file outputs/matrix_csv/vectors_long_mode_centered.csv`
- Compare two: `python visualize_vectors.py --compare outputs/matrix_csv/vectors_long_mode_centered.csv outputs/matrix_csv/vectors_long_mode_raw.csv`

Example: fetch a single vector directly via Modal:
- `modal run src/examples/get_activation_vector.py --text "Hello world" --mode short`
- `modal run src/examples/get_activation_vector.py --file src/training_data/text-samples/01_bonded_cats_apartment.txt --mode long --center`

## Key Files

- `src/extract_vector.py`
  - Modal class `Pythia12BActivationExtractor` with endpoints: activation matrix, pooled vectors, metadata.
  - Supports `center=True` with `centering_vector` path resolved via volume utilities.

- `src/utils/pooling.py`
  - Implements the `short` and `long` pooling strategies in float32.

- `src/corpus_mean.py`
  - Stage 1: parallel per‑document extraction to volume.
  - Stage 2: checkpointed aggregation that resumes and can run detached.

- `src/diagnostics/plot_similarity_matrix.py`
  - Renders cosine‑similarity heatmaps from column‑oriented vector CSVs.
  - Supports `--mode short|long` and optional `--per-chunk true` (long only).
  - All plot text (titles, ticks, colorbar, annotations) is black for readability.

- `src/diagnostics/compare_vectors_with_mean.py`
  - Checks the long‑mode identity `(raw − centered) ≈ pooled_mean` per chunk.
  - Resolves token lengths via `--lengths_csv` or `--use_text_samples`.
  - See docs: `docs/compare_vectors_with_mean.md` for labels and interpretation.

## Programmatic Use (Modal)

- Create a client: `modal.Cls.from_name("activation-vector-project", "Pythia12BActivationExtractor")`
- Call remotely, e.g. `get_activation_vector.remote(text=..., pooling_strategy="short"|"long", center=True|False, centering_vector=<path>)`
- See `src/cosine_similarity.py` and `src/generate_vector_csv.py` for patterns.

## Environment & Dependencies

- Modal CLI configured for your account
- Python 3.10 runtime on Modal
- PyTorch, Transformers, SafeTensors (installed in images defined in scripts)

## Outputs

- `outputs/matrix_csv/` → analysis‑friendly CSVs with columns as samples and rows as dimensions.
- `outputs/diagnostics/` → plots and comparison reports.
- `outputs/*.safetensors` → vectors and metadata from ad‑hoc runs.

## Tips v2

- Negative cosine values in similarity matrices are fine; they indicate anticorrelation (angles > 90°). When vectors are mean‑centered, small negatives are common.
- Long‑mode per‑chunk plots help diagnose which pooling window contributes most to similarities.
