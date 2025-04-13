## Usage

Steps to use our method and reproduce baselines. Please generate or download the QA dataset crafted using MIMIC.

### Steps:

- First step is to index MIMIC EHR/PDD to be used for attribution

```
python create_ehr_indexes.py --input <PATH TO DATA ROOT DIRECTORY> --output <PATH TO INDEXES OUTPUT DIRECTORY>
```


- Second step is to perform attribution, that is to fetch relevant content from PDD to the question answer pair. This scripts executes baselines methods as well as main method

```
python main.py --qa_dir <PATH TO QA-DATA ROOT DIRECTORY> --pdd_dir <PATH TO MIMIC DATASET'S ROOT DIRECTORY> --index_dir <PATH TO INDEX ROOT DIRECTORY> --output_dir <PATH TO OUTPUT DIRECTORY> --alpha <METHOD PARAMETER> --submodular_function <METHOD PARAMETER>
```


- Third step is to run the evaluations on the generated attribution results.

```
python evals.py --predictions_dir <PATH TO METHOD OUTPUTS> --results_dir <PATH TO SAVE EVALUATIONS>
```