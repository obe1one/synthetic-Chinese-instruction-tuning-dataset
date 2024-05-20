base_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

base_instruction_traditional_chinese = "我想請你擔任提示重寫員。\n你的目標是將給定的提示重寫成更複雜的版本，讓那些著名的人工智慧系統（例如，chatgpt和GPT4）難以處理一些。\n\
										但是重寫的提示必須合理，並且必須被人類理解和回應。\n你的重寫不能省略非文本部分，例如表格和代碼在＃給定提示＃中。另外，請勿省略＃給定提示＃中的輸入。\n\
										你應該使用以下方法使給定的提示變得更複雜：\n{}\n\
										你應該盡量不讓＃重寫提示＃變得冗長，＃重寫提示＃只能在＃給定提示＃中增加10到20個詞。\n\
										＃給定提示＃主要以繁體中文為主，並混合其它語言。請在改寫時遵照著＃給定提示＃的語言，不要在相對應的地方進行翻譯。\
										'＃給定提示＃'、'＃重寫提示＃'不得出現在＃重寫提示＃中\n"


def createConstraintsPrompt(instruction):
	prompt = base_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt

def createDeepenPrompt(instruction):
	prompt = base_instruction.format("If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt

def createConcretizingPrompt(instruction):
	prompt = base_instruction.format("Please replace general concepts with more specific concepts.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt


def createReasoningPrompt(instruction):
	prompt = base_instruction.format("If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt

def createDeepenPromptTraditionalChinese(instruction):
	prompt = base_instruction.format("如果＃給定提示＃包含對特定問題的詢問，則可以增加詢問的深度和廣度。")
	prompt += "＃給定提示＃:\n{}\n".format(instruction)
	prompt += "＃重寫提示＃:\n"
	return prompt
