import argparse
import dotenv
import json
import os
import random
from datetime import datetime
from tqdm import tqdm
from data_generator import LLMDataGenerator

class ExampleSelector:
    @staticmethod
    def random_select(examples, n_examples):
        return random.sample(examples, n_examples)
    
    @staticmethod
    def balanced_random_select(examples, n_examples):
        examples_with_input = [e for e in examples if e["input"] != "None"]
        examples_without_input = [e for e in examples if e["input"] == "None"]

        selected_with_input = random.sample(examples_with_input, n_examples // 2)
        selected_without_input = random.sample(examples_without_input, n_examples - n_examples // 2)

        selected = selected_with_input + selected_without_input
        random.shuffle(selected)
        return selected
    
    @staticmethod
    def random_select_examples_without_input(examples, n_examples):
        examples_without_input = [e for e in examples if e["input"] == "None"]
        selected_without_input = random.sample(examples_without_input, n_examples)

        random.shuffle(selected_without_input)
        return selected_without_input
    
    @staticmethod
    def random_select_with_generated(real_examples, gen_examples, n_examples, real_ratio=0.8):
        n_real = round(n_examples * real_ratio)
        n_gen = n_examples - n_real

        selected_real = random.sample(real_examples, n_real)
        selected_gen = random.sample(gen_examples, n_gen)

        selected = selected_real + selected_gen
        random.shuffle(selected)

        return selected


def main(args):
    config = dotenv.dotenv_values(".env")
    if args.llm == "claude":
        api_key = config["ANTHROPIC_API_KEY"]
    else:
        api_key = config["OPENAI_API_KEY"]

    agent = LLMDataGenerator(
        llm=args.llm,
        model_name=args.model_name,
        api_key=api_key
    )

    with open(args.begin_examples_path, "r", encoding="utf8") as reader:
        examples = json.load(reader)
    
    with open(args.saved_path, "r", encoding="utf8") as reader:
        all_data = json.load(reader)
    
    target_num = args.n_data - len(all_data)
    gen_examples = [data["conversation"][0] for data in all_data]

    for i in tqdm(range(target_num)):
        if len(all_data) + i + 1 <= args.warmup or not args.add_to_pool:
            selected = ExampleSelector.random_select(examples, n_examples=args.n_examples)
        else:
            selected = ExampleSelector.random_select_with_generated(
                real_examples=examples,
                gen_examples=gen_examples,
                n_examples=args.n_examples,
            )

        print("= = = = = = Example = = = = = =")
        print("\n> > >\n".join([s["instruction"] for s in selected]))
        print("= = = = = = = = = = = = = = = =")

        try:
            data = agent.few_shot_generate(examples=selected, turns=3)
        except:
            print("[x] Failed")
            continue

        if not len(data):
            print("[x] Failed")
            continue

        all_data.append({ "conversation": data })
        gen_examples.append(data[0])

        with open(args.saved_path, "w", encoding="utf8") as writer:
            json.dump(all_data, writer, indent="\t", ensure_ascii=False)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["claude", "openai"], default="claude")
    parser.add_argument("--model_name", choices=["claude-3-opus-20240229", "gpt-4-0613", "gpt-4-0125-preview",
                                                 "gpt-3.5-turbo-0125", "claude-3-sonnet-20240229"],
                        default="claude-3-sonnet-20240229")
    parser.add_argument("--begin_examples_path", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--n_data", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--saved_path", type=str, required=True)
    parser.add_argument("--dot-env", type=str, default=".env")
    parser.add_argument("--add_to_pool", action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)

