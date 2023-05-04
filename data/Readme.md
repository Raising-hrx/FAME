# Data

```
Data
├── ControllerData                                                 # data for training the controller 
│   ├── controller.train.from_gold.v9.jsonl
│   ├── controller.train.from_gold.v9.task1.jsonl
│   ├── controller.train.from_imitation.v9.jsonl
│   └── Iterative_Training_Data
│
├── EntailmentBankQA                                               # We convert EntailmentBank to EntailmentBankQA by adding back the 4-way multiple options from the ARC dataset.
│   └── dataset                                                    # We also convert each question+oprion to a hypothesis
│       ├── task_1
│       ├── task_2
│       └── task_3
│
├── entailment_trees_emnlp2021_data_v3                             # EntailmentBank v3 (https://github.com/allenai/entailment_bank/)
│   ├── dataset
│   │   ├── task_1
│   │   ├── task_2
│   │   └── task_3
│   └── supporting_data
│       └── preprocessed_corpus.json                               # the EntailmentBank corpus
│
└── TeachableEntailmentWriterData                                  # data for training the verifier (https://allenai.org/data/entailer)
    └── processed_step_data
        ├── step.dev.v1.jsonl
        └── step.train.v1.jsonl
```