# synthetic-Chinese-instruction-tuning-dataset

## Description

Utilize 'evol instruction' and 'self-instruct' techniques to generate multi-turn synthetic Traditional Chinese instruction-tuning dataset

## setup

```bash
pip install -r requirements.txt
```

## Rewrite Synthetic Data

```bash
python rewrite.py \
		   --llm {Choose 'claude' or 'openai'} \
		   --model_name {Choose the model you want to use} \
		   --saved_path {Path where you want to save the output results} \
		   --dot-env {Your '.env' file path} \
		   --dataset {The dataset you want to rewrite. Dataset format should be 'sharegpt'} \
		   --split {Dataset split you want to rewrite}

python format.py \
		   --task evol \
		   --src_path {The 'saved_path' when running rewrite.py}
		   --output_path {Path where you want to save the output results. The output file would be a .jsonl file}
```

## Generate Synthetic Data

```bash
python gen_data.py \
		   --llm {Choose 'claude' or 'openai'} \
		   --model_name {Choose the model you want to use} \
		   --saved_path {Path where you want to save the output results} \
		   --begin_examples_path {Path to seed instruction file} \
		   --n_examples {How many in-context examples you want to use in your text prompt} \
		   --n_data {The number of synthetic data you want to generate} \
		   --turns {How many turns you want to generate for each conversation} \
		   --saved_path {Path where you want to save the output results} \
		   --dot-env {Your '.env' file path} \
		   --add_to_pool {Whether you want to add your generated data to seed task pool or not}

python format.py \
		   --task syn \
		   --src_path {The 'saved_path' when running gen_data.py}
		   --output_path {Path where you want to save the output results. The output file would be a .jsonl file}
```
