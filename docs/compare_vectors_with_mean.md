# Comparing Raw vs Centered Long Vectors to the Corpus Mean

This note explains what `src/diagnostics/compare_vectors_with_mean.py` does, the expected inputs/outputs, and how to interpret the labels in both the input CSVs and the generated report.

## Goal

Verify the long‑mode centering identity on a per‑sample basis:

    diff_long = raw_long − centered_long ≈ pooled_mean_long

Where `pooled_mean_long` is computed by applying the same long‑mode pooling windows to the corpus‑mean matrix for the document’s token length.

## Inputs

- `--mean_path` (required): Path to a local `corpus_mean_*.safetensors` downloaded from your Modal volume.
- `--raw_csv` (required): Long‑mode vector CSV produced by `generate_vector_csv.py` (no centering).
- `--centered_csv` (required): Long‑mode vector CSV produced by `generate_vector_csv.py --center`.
- Token length resolution (choose one):
  - `--lengths_csv`: A CSV with one header row of labels matching the vector CSVs and one data row of integers giving effective token counts per sample (after reversal).
  - `--use_text_samples`: Recomputes lengths by re‑processing the 20 bundled text files via the deployed extractor.

## Output

Writes a CSV report to `outputs/diagnostics/compare_report.csv` with columns:

- `sample`: The sample label (see Labeling section).
- `doc_len`: Effective token length used to slice the corpus‑mean matrix.
- `chunk`: One of `last`, `exp_977`, `exp_933`, `exp_841` (see Chunk Labels section).
- `cosine`: Cosine similarity between `(raw − centered)` and the predicted pooled mean for that chunk.
- `l2_diff`: L2 norm of the difference between `(raw − centered)` and the prediction.
- `||diff||`: L2 norm of `(raw − centered)` for that chunk.
- `||pred||`: L2 norm of the predicted pooled mean for that chunk.

High cosine values and low `l2_diff` indicate good agreement between theory and measurements.

## Labeling

### Sample Labels (CSV headers)

The generated analysis CSVs use 20 columns (samples):

- `long_1` … `long_10`: Long documents from `src/training_data/text-samples/*_long.txt`
- `short_1` … `short_10`: Short documents from `src/training_data/text-samples/*_short.txt`

These labels are consistent across raw and centered CSVs and are used to align columns between files and with token length metadata.

### Chunk Labels (long‑mode vector segments)

Long‑mode vectors are 20480‑dimensional and are formed by concatenating four 5120‑dimensional chunks:

- `last`: A one‑hot window at the final token (after sequence reversal). Captures only the last token state.
- `exp_977`: Exponential window with a half‑life at 2.3% of document length. Heavily emphasizes the last ~2.3% of tokens.
- `exp_933`: Exponential window with a half‑life at 6.7% of document length. Broader than `exp_977`.
- `exp_841`: Exponential window with a half‑life at 15.9% of document length. Broadest of the three.

Notes:
- Internally, sequences are reversed so position 0 corresponds to the final token; windows decay moving backward through the document.
- The numeric suffixes (`977`, `933`, `841`) are mnemonic for the per‑step decay multipliers implied by those half‑lives; the important part is the relative breadth: `last` < `exp_977` < `exp_933` < `exp_841`.

## How It Works

For each sample/column:
1. Read the raw and centered long vectors and compute `diff = raw − centered`.
2. Resolve the document’s effective token length `L`.
3. Slice the corpus‑mean matrix to length `L` and apply the four long‑mode windows to obtain the predicted mean contribution per chunk.
4. Report cosine and L2 metrics comparing `diff` vs the prediction for each chunk.

## Usage Examples

Generate the CSVs for the bundled samples:

- Raw: `modal run src/generate_vector_csv.py`
- Centered: `modal run src/generate_vector_csv.py --center`

Run the comparison (recomputes lengths from the text files):

```
modal run src/diagnostics/compare_vectors_with_mean.py \
  --mean_path outputs/corpus_mean_XXXXXXXX.safetensors \
  --raw_csv outputs/matrix_csv/vectors_long_mode_raw.csv \
  --centered_csv outputs/matrix_csv/vectors_long_mode_centered.csv \
  --use_text_samples
```

Or provide explicit lengths:

```
modal run src/diagnostics/compare_vectors_with_mean.py \
  --mean_path outputs/corpus_mean_XXXXXXXX.safetensors \
  --raw_csv outputs/matrix_csv/vectors_long_mode_raw.csv \
  --centered_csv outputs/matrix_csv/vectors_long_mode_centered.csv \
  --lengths_csv path/to/lengths.csv
```

The report is saved to `outputs/diagnostics/compare_report.csv`.

## Interpreting Results

- Cosine near 1.0 and small `l2_diff` → strong agreement with the centering identity for that chunk.
- Lower cosine on a specific chunk may indicate that window is more sensitive to noise or mismatches in token length resolution.
- Negative cosines can occur and indicate anticorrelation; check token lengths and centering path if unexpected.

