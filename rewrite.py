import argparse
import dotenv
import json
import os
import random
from datasets import load_dataset
from datetime import datetime
from evol_instruction import *
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tqdm import tqdm

class InstructionRewriter:
    def __init__(self, llm, model_name, api_key):
        self._validation(llm)

        self.llm = llm
        self.model_name = model_name

        if self.llm == "claude":
            self.agent = ChatAnthropic(model=self.model_name, anthropic_api_key=api_key, default_request_timeout=60)
        elif self.llm == "openai":
            self.agent = ChatOpenAI(model=self.model_name, openai_api_key=api_key, timeout=60)

    def _validation(self, llm):
        if not llm in ["claude", "openai"]:
            raise ValueError(
                f"The LLM {llm} doesn't exist"
            )
    
    def rewrite(self, instruction, rewrite_func=createDeepenPrompt):
        prompt = rewrite_func(instruction)
        output = self.agent.invoke(prompt)
        return output.content
    
    def response_to_multi_turn_instruction(self, instructions):
        prompt, responses = [], []
        
        for instruction in instructions:
            prompt.append(HumanMessage(content=instruction))
            response = self.agent.invoke(prompt)
            responses.append(response.content)
            prompt.append(AIMessage(content=response.content))
        
        return responses
    
    def rewrite_log(self, log_data, rewrite_func=createDeepenPrompt):
        instructions = [msg["content"] for msg in log_data["messages"] if msg["from"] == "user"]
        if not len(instructions):
            return []
        instructions = [self.rewrite(instruction, rewrite_func) for instruction in instructions]
        return instructions

def rewrite_instructions(agent, args):
    dataset = load_dataset(args.dataset, split=args.split)
    n_data = len(dataset)

    with open(args.saved_path, "r", encoding="utf8") as reader:
        all_data = json.load(reader)
    
    indices = list(range(n_data))
    for data in all_data:
        indices.remove(data["id"])

    for idx in tqdm(indices):
        instructions = agent.rewrite_log(dataset[idx], rewrite_func=createDeepenPromptTraditionalChinese)
        all_data.append({
            "id": idx,
            "original_conversation": dataset[idx]["messages"],
            "instructions": instructions
        })

        all_data = sorted(all_data, key=lambda data: data["id"])

        with open(args.saved_path, "w", encoding="utf8") as writer:
            json.dump(all_data, writer, indent="\t", ensure_ascii=False)

    all_data = sorted(all_data, key=lambda data: data["id"])

    with open(args.saved_path, "w", encoding="utf8") as writer:
        json.dump(all_data, writer, indent="\t", ensure_ascii=False)


def main(args):
    config = dotenv.dotenv_values(args.dot-env)
    if args.llm == "claude":
        api_key = config["ANTHROPIC_API_KEY"]
    else:
        api_key = config["OPENAI_API_KEY"]

    agent = InstructionRewriter(
        llm=args.llm,
        model_name=args.model_name,
        api_key=api_key
    )

	rewrite_instructions(agent, args)

    with open(args.saved_path, "r", encoding="utf8") as reader:
        all_data = json.load(reader)
    
    n_data = len(all_data)
    
    for idx in tqdm(range(n_data)):
        if "responses" in all_data[idx]:
            continue
        
        try:
            instructions = all_data[idx]["instructions"]
            responses = agent.response_to_multi_turn_instruction(instructions)
            all_data[idx]["responses"] = responses
        except Exception as e:
            print(e)
            continue

        with open(args.saved_path, "w", encoding="utf8") as writer:
            json.dump(all_data, writer, indent="\t", ensure_ascii=False)
    
    with open(args.saved_path, "w", encoding="utf8") as writer:
        json.dump(all_data, writer, indent="\t", ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["claude", "openai"], default="claude")
    parser.add_argument("--model_name", choices=["claude-3-opus-20240229", "gpt-4-0613", "gpt-4-0125-preview",
                                                 "gpt-3.5-turbo-0125", "claude-3-sonnet-20240229"],
                        default="claude-3-opus-20240229")
    parser.add_argument("--saved_path", type=str, required=True)
    parser.add_argument("--dot-env", type=str, default=".env")
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()
    print(args)
    main(args)


