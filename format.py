import argparse
import json
from opencc import OpenCC
from tqdm import tqdm


def format_syn(file_path, output_path):
    with open(file_path, "r", encoding="utf8") as reader:
        all_data = json.load(reader)
    
    instructions = [[ { "from": "human", "value": conv["instruction"] } for conv in data["conversation"]] for data in all_data]
    outputs = [[ { "from": "gpt", "value": conv["output"] } for conv in data["conversation"]] for data in all_data]
    cc = OpenCC("s2twp")

    conversations = [
        { "conversations": [item for pair in zip(instructions[i], outputs[i]) for item in pair] }
        for i in range(len(instructions))
    ]
    cc_conversations = [
        [ { "from": conv["from"], "value": cc.convert(conv["value"]) } for conv in conversations[i]["conversations"] ]
        for i in tqdm(range(len(conversations)))
    ]
    for i in range(len(conversations)):
        conversations[i]["converted_conversations"] = cc_conversations[i]

    conv_len = {}
    print(len(conversations))
    for conv in conversations:
        val = len(conv["conversations"])
        if not val in conv_len:
            conv_len[val] = 0
        conv_len[val] += 1
    print(conv_len)

    with open(output_path, "w", encoding="utf8") as writer:
        for conv in conversations:
            data = json.dumps(conv, ensure_ascii=False) + "\n"
            writer.write(data)
    
    # with open(output_path, "w", encoding="utf8") as writer:
    #     json.dump(conversations, writer, indent="\t", ensure_ascii=False)
    

def format_evol(file_path, output_path):
    with open(file_path, "r", encoding="utf8") as reader:
        all_data = json.load(reader)
    
    cc = OpenCC("s2twp")
    indices = [idx for idx in range(len(all_data)) if "responses" in all_data[idx]]
    instructions = [
        [{ "from": "human", "value": instr } for instr in all_data[idx]["instructions"]]
        for idx in indices
    ]
    responses = [
        [{ "from": "gpt", "value": res } for res in all_data[idx]["responses"]]
        for idx in indices
    ]
    conversations = [
        { "conversations": [item for pair in zip(instructions[idx], responses[idx]) for item in pair] }
        for idx in range(len(responses))
    ]
    print(len(conversations))
    conversations = [conv for conv in conversations if len(conv["conversations"])]
    print(len(conversations))

    cc_conversations = [
        [ { "from": conv["from"], "value": cc.convert(conv["value"]) } for conv in conversations[i]["conversations"] ]
        for i in tqdm(range(len(conversations)))
    ]
    for i in range(len(conversations)):
        conversations[i]["converted_conversations"] = cc_conversations[i]


    with open(output_path, "w", encoding="utf8") as writer:
        for conv in conversations:
            data = json.dumps(conv, ensure_ascii=False) + "\n"
            writer.write(data)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", choices=["evol", "gen"])
	parser.add_argument("--src_path", type=str, required=True)
	parser.add_argument("--output_path", type=str, required=True)

	args= parser.parse_args()
	
	if args.task == "evol":
		format_evol(file_path=args.src_path, output_path=args.output_path)
	elif args.task == "gen":)
		format_syn(file_path=args.src_path, output_path=args.output_path)

