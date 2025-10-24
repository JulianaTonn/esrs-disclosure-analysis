# ESRS Disclosure Analysis using Retrieval-Augmented Generation (RAG)

This repository contains the code, data, and documentation accompanying the master’s thesis **"Automating the Assessment of Corporate Sustainability Disclosures under the CSRD using Large Language Models"** by Juliana Tonn, submitted for the Master Examination in Data Science in Business and Economics of the Eberhard Karls Universität Tübingen.

The project develops an **open-source Retrieval-Augmented Generation (RAG) framework** to automatically extract and evaluate sustainability disclosures in accordance with the **European Sustainability Reporting Standards (ESRS)** — specifically focusing on **ESRS S1 Own Workforce** indicators. 


## RAG Pipeline

The RAG system retrieves relevant text sections from corporate sustainability reports and evaluates whether the disclosed content aligns with ESRS S1 requirements.

<p align="center">
  <img src="data/rag_pipeline.png" alt="RAG Pipeline" width="700"/>
</p>

## Directories
- **rag_system.py**: the script wit core implementation of the Retrieval-Augmented Generation framework
- **EsrsMetadata.xlsx**: metadata file containing ESRS S1 queries (yes/no question) and guidelines (additional context, ensuring that the model evaluates disclosures consistently with ESRS requirements)
- **main_inference.py**: the script to run the automated disclosure analysis

- performance_evaluation:
  - validation_dataset.xslx: manually coded validation dataset for performance assessment 
  - sample_reports/*: two sample reports forming the basis of the validation set
  - prompt_exploration.ipynb: exploration of different prompt templates, system prompts, queries, and guideline wording
  - rag_comparison.ipynb: compare performance of different rag configurations (model benchmarking, prompt modules, PDF text extraction, retrieval settings)

- data:
  - get_reports.ipynb: prepare dataset with reports and links that is used for inference
  - SRN-CSRD_report_archive - csrd.csv: downloaded esrs conform report dataset from SRN (Sustainability Reporting Navigator https://www.srnav.com/)
  - all_results.jsonl: inference results of the automated disclosure analysis
  - visualize_data.ipynb: prepares company data and inference results for visualization (plots, summary statistics)

## Acknowledgements
The author acknowledges support by the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant INST 35/1597-1 FUGG for providing computational resources.

--- 
## Contact
For questions, feedback, or collaborations please reach out juliana.tonn@student.uni-tuebingen.de.