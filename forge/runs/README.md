# forge/runs/

Auto-generated Forge instance files land here (from `run_forge.py` experiment runner).

Each file is named `{family}_n{n}.frg` and contains:
- The base model from `../models/hypergraph.frg`
- An `inst` block with qubit/generator facts for the specific instance

## How to run

1. Install Forge: https://forge-fm.org/docs/install
2. Run a query:
   ```
   racket hypergraph.frg
   ```
3. Or run an instance file:
   ```
   racket bounded_degree_n8.frg
   ```

## Outputs

Forge outputs go here as `.xml` or `.json` witness files.
