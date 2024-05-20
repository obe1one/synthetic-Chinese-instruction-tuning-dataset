from langchain.prompts.prompt import PromptTemplate

def create_first_instruction_prompt(parser, examples, input_variables=[]):
    # example_sentences = "\n".join([f"問題 {i + 1}: {e['instruction']}" for i, e in enumerate(examples)])
    example_sentences = "\n".join([f"問題: {e['instruction']}" for i, e in enumerate(examples)])

    prompt_template = PromptTemplate(
        template="以下是一系列的問題：\n\n" + example_sentences + "\n\n" + "請仿造這些指示範例，生成一個新的指示。\n" + "{format}",
        input_variables=input_variables,
        partial_variables={ "format": parser.get_format_instructions() }
    )

    return prompt_template

def create_first_instruction_prompt_v2(examples, input_variables=[]):
    # example_sentences = "\n".join([f"問題 {i + 1}: {e['instruction']}" for i, e in enumerate(examples)])
    example_sentences = "\n\n".join([f"指示範例: {e['instruction']}" for i, e in enumerate(examples)])

    prompt_template = PromptTemplate(
        template="我要你創造出一個能夠輸入給 AI Systems (e.g., GPT4 and Antropic) 的指示\n"
                 "以下為一些範例的指示：\n\n" + example_sentences + "\n\n"
                 "請仿造這些指示範例，生成一個全新的指令。這個指令必須要是繁體中文。\n\n"
                 "新的指示：",
        input_variables=input_variables,
    )

    return prompt_template

def create_first_instruction_prompt_v3(examples, input_variables=[]):
    example_sentences = "\n\n".join([f"指示範例: {e['instruction']}" for i, e in enumerate(examples)])

    prompt_template = PromptTemplate(
        template="我要你創造出一個能夠輸入給 AI Systems (e.g., GPT4 and Antropic) 的指示\n"
                 "以下為一些範例的指示：\n\n" + example_sentences + "\n\n"
                 "請仿造這些指示範例，生成一個全新的指令。\n"
                 "這個指示必須要是繁體中文。我會把你的回應直接貼給對方，請不要在回應中與我互動，也不要將 '範例指示' 和 '新的指示' 加入在你的回應中。\n\n"
                 "新的指示：\n",
        input_variables=input_variables,
    )

    return prompt_template

def create_multi_turn_prompt(dialogue):
    multi_turn = "\n\n".join([f"我:\n{data['instruction']}\n\n對方:\n{data['output']}" for data in dialogue])
        
    prompt_template = PromptTemplate(
        template="我現在在和另一個人進行對話，以下為我們的對話內容:\n\n" + multi_turn +
                    "\n\n請幫我構思接下來我下一句要怎麼回應，才能讓對話進行下去，並且主題必須要和我們現在討論的主題連貫。這個問題裡面最多只能有一個問題。\n"
                    "我會把你的回應直接貼給對方。",
        input_variables=[]
    )

    return prompt_template

def create_multi_turn_prompt_v2(dialogue, input_variables=[]):
    multi_turn = "\n\n".join([f"我:\n{data['instruction']}\n\nAI System:\n{data['output']}" for data in dialogue])

    prompt_template = PromptTemplate(
        template="我要你創造出一個能夠輸入給 AI Systems (e.g., GPT4 and Antropic) 的指示\n"
                 "以下是我和 AI System 的對話：\n\n" + multi_turn + "\n\n"
                 "請基於我們的對話內容，幫我構思下一輪的指示。這個指示必須和我們的對話主題相關，或者是針對 AI System 的回覆做進一步詢問。這個指示裡面最多只能有一個問題。\n"
                 "這個指示必須要是繁體中文。我會把你的回應直接貼給對方。\n\n"
                 "下一輪的指示：",
        input_variables=input_variables
    )

    return prompt_template

def create_multi_turn_prompt_v3(dialogue, input_variables=[]):
    multi_turn = "\n\n".join([f"我:\n{data['instruction']}\n\nAI System:\n{data['output']}" for data in dialogue])

    prompt_template = PromptTemplate(
        template="我要你創造出一個能夠輸入給 AI Systems (e.g., GPT4 and Antropic) 的指示\n"
                 "以下是我和 AI System 的對話：\n\n" + multi_turn + "\n\n"
                 "請基於我們的對話內容，幫我構思下一輪的指示。這個指示必須和我們的對話主題相關，或者是針對 AI System 的回覆做進一步詢問。這個指示裡面最多只能有一個問題。\n"
                 "這個指示必須要是繁體中文。而你的回應不需要包含針對對話內容的評論，也不要將 '下一輪的指示' 加入到回應中。我會把你的回應直接貼給對方。\n\n"
                 "下一輪的指示：",
        input_variables=input_variables
    )

    return prompt_template    