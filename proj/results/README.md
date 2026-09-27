# Supplied experimental records

The original document-level corpora are not copied here.  `ChinaRe` is proprietary; the available process material contains aggregate curves and spreadsheet reports.  The records below preserve that material in a compact, dataset-first layout.

| Dataset directory | Paper corpus | Contents |
|---|---|---|
| `haodf/` | HaoDaiFu, 805 diseases | all/rare macro-F1 learning curves and supplied aggregate reports |
| `chinare/` | ChinaRe, 44 disease categories | all/rare macro-F1 learning curves and supplied aggregate reports |

Each `learning_curves/*.csv` has 10 rows for the 10%, 20%, …, 100% training rates. Columns follow the method order documented in `../README.md`. The `summaries/` files are unmodified copies selected from `过程数据/graph_attention_report`; their names retain the original experimental condition. `supplied_curve_alc.csv` is produced by `code/summarize_supplied_curves.py` using trapezoidal integration on the supplied curves.

Source provenance is recorded in `provenance.md`.
