import json
import openai
import copy
from openai.error import RateLimitError
import backoff
from tqdm import tqdm

qna_path_yesno = "CPL-master/CPL/prompt/infilled_template/yesno.json"

with open(qna_path_yesno) as f:
    #data = [json.loads(line) for line in f]
    data=json.load(f)
    

            
openai.api_key = "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"

initial_prompt="Q:is this a creamy soup?/nA1:this is a creamy soup/nA2:this is not a creamy soup/n"
#print(initial_prompt)

prompts=[initial_prompt+"Q:"+dat["question"]+"/nA1:"+dat["prompt"]+"/nA2:" for idx,dat in enumerate(data)]

all_responses=[]

#prompts=prompts[10000:30000]


@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(curr_prompt):
    response = openai.Completion.create(
                #engine="text-davinci-003",
                engine="text-curie-001",
                prompt=curr_prompt,
                temperature=0,
                max_tokens = 85,
                n=1,
                stop="."
            ) 
    return response


for curr_prompt in tqdm(prompts):
    

    response = completions_with_backoff(curr_prompt)
    all_responses.append(response["choices"][0]["text"])

new_data=copy.deepcopy(data)

j=0
start = 0 
end=len(new_data)
for i in range(start,end):
    new_data[i]["prompt2"] = all_responses[j]
    j+=1

#with open("new_file_opt_10000_to_30000.json.json", 'w') as f:
with open("new_file_gpt_all.json.json", 'w') as f:
	json.dump(new_data, f)

