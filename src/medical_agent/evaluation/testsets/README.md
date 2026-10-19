# Evaluation Test Sets

This directory contains CSV test sets for RAGAS evaluation of the medical RAG pipeline.

## CSV Format (RAGAS v0.4)

Test set CSVs must include the following columns:

| Column | Description |
|--------|-------------|
| `user_input` | The question to ask the pipeline |
| `reference` | Expected/reference answer for evaluation |

## Generating a Synthetic Test Set

Generation uses 80% single-hop (simple) and 20% multi-hop questions, with medical user POV instructions. Chunks are randomly sampled for diversity across papers.

```bash
# Generate 20 questions from up to 200 database chunks (random order for diversity)
python -m medical_agent.evaluation.generate_testset --size 20

# Reproducible generation (same chunk selection each run)
python -m medical_agent.evaluation.generate_testset --size 20 --seed 42

# Generate from a specific paper
python -m medical_agent.evaluation.generate_testset --size 10 --paper "vaginal microbiome"

# Limit chunks fetched from database
python -m medical_agent.evaluation.generate_testset --size 5 --limit 50
```

Generated files are saved as `testset_YYYYMMDD_HHMMSS.csv`.

## Creating a Manual Test Set

Create a CSV file with the required columns:

```csv
user_input,reference
"What is a normal vaginal pH range?","A normal vaginal pH is between 3.8 and 4.5."
"What causes bacterial vaginosis?","Bacterial vaginosis is caused by an imbalance in vaginal flora..."
```

## Running Evaluation

```bash
python -m medical_agent.evaluation.run_evaluation --testset testsets/testset_XXXX.csv
```

Results are saved as JSON in the `results/` directory.
