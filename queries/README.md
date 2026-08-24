# queries/

Question–SPARQL pairs used to prompt, train, and evaluate the LLMBioSPARQL pipeline, plus reference use-cases and schema notes.

The two gold example files (`train_examples.json`, `test_examples.json`) share the same schema — one object per question:

```json
{
  "question": "Natural-language question",
  "sparql": "SELECT ...",
  "extractor": {"outputs": ["n0"], "nodes": {...}, "triples": [...]},
  "linker": {"input": [...], "output": [...]},
  "output_designation": {"input": [...], "output": {...}}
}
```

`extractor`, `linker`, and `output_designation` are the gold intermediate results for each pipeline stage (see the root `README.md` for what each stage produces) — used both as few-shot prompt material and to evaluate each stage independently, without upstream errors propagating in.

---

## Gold example files

### `train_examples.json`
14 questions used as the in-prompt few-shot examples for concept extraction, output designation, and entity linking. **Never used for evaluation.**

### `test_examples.json`
The held-out evaluation set — 47 question/gold-query pairs, restricted to six entity graphs (gene, protein, taxon, tad, GO, OMIM) and seven relation graphs (gene2phen, tfac2gene, prot2prot, prot2bp, prot2mf, prot2cc, reg2targ). Each triple additionally carries a `"correct_path"` field (the gold path-planner segments) used to evaluate path planning/resolution. This is the file `evaluation.ipynb`'s main sections score against.

---

## Raw source queries

`query_01_0.txt` … `query_08_0.txt` — hand-written SPARQL queries with an explanatory comment header, used as the source material.

---

## Use-cases (UC files)

Complex, multi-step SPARQL queries designed by **Juan Mulero** to illustrate realistic biomedical research workflows on BioGateway. They serve as the upper bound of difficulty for the pipeline and are not yet handled end-to-end. Each has an `-updated` variant with revised phrasing/output columns.

| File | Topic |
|---|---|
| `UC1.txt` / `UC1-updated.txt` | Impact of a SNP (rs4784227 / chr16:52565276) on gene-expression regulation: enhancers, target genes, proteins, biological processes, TFs |
| `UC2.txt` / `UC2-updated.txt` | Regulatory landscape of breast cancer: phenotype-linked genes, enhancers in breast tissue, biological processes, cooperating TFs |
| `UC3.txt` / `UC3-updated.txt` | Dataset generation linking OMIM phenotypes, genes, CRM enhancers (with coordinates and distances), proteins, GO terms, protein interactions, and TF regulation |
