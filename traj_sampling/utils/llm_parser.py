import json
import requests
import time
import numpy as np
import gzip
MAX_TRY = 5

import os
API_KEY=os.environ['API_KEY']
ORGANIZATION=os.environ['ORGANIZATION']

class KeyComponentExtractor(object):
    def __init__(self, obj_labels, action_labels) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Organization": f"{ORGANIZATION}"
        }
        
        self.prompts = f"Please help to decompose an intruction into a object-and-action sequence in temporal order. You can only choose object labels from {obj_labels} and choose action labels from {action_labels}. The output format should follows a list of tuple. Each tuple must contains the type of the label (object or action) and the actual label you identified. For any object/action appears in the instruction but you cannot find a good match in the object/action label set, skip it. \n\nHere is an example: \n\nInstruction: Pass the bed then turn right. Walk toward the fireplace. when you get to the fireplace turn right. Turn left into the room and wait near the jacuzzi. \nYour answer: [('object', 'bed'), ('action', 'turn right'), ('object', 'fireplace'), ('action', 'turn right'), ('action', 'turn left'), ('object', 'bathtub')]\n\n"

        self.total_price = {"input_tokens":[], "output_tokens": [], "price": 0}
        # self.rexp = re.compile("\{.*?\}")
    
    def update_price(self, response):
        price = np.ceil(response['usage']['prompt_tokens']/1000) * 0.0005 + np.ceil(response['usage']['completion_tokens']/1000)*0.0015
        price_record = {
            "input_tokens": response['usage']['prompt_tokens'],
            "output_tokens": response['usage']['completion_tokens'],
            "price": price
        }
        self.total_price["input_tokens"].append(price_record["input_tokens"])
        self.total_price["output_tokens"].append(price_record["output_tokens"])
        self.total_price["price"] += price_record["price"]
        return price_record
    
    def _post_process_response(self, msg):
        msg = msg["choices"][0]['message']['content']
        try:
            msg = msg.replace('(', '[').replace(')', ']').replace('\'', '\"')
            msg_dict = json.loads(msg)
        except:
            import ipdb;ipdb.set_trace() # breakpoint 65
            print()
        return msg_dict

    def run(self, instr, log_root=None):
        for i in range(MAX_TRY):
            try:
                sys_msg = {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompts + f"Instruction: {instr} \nYour answer: "
                        }
                    ]
                }

                payload = {
                    # "model": "gpt-4o-mini-2024-07-18",
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        sys_msg
                    ],
                    "max_tokens": 500
                } 
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
                
                response = response.json()
                if log_root is not None:
                    log_name = f"{log_root}/attr_extractor_{int(time.time())}.json"
                    with open(log_name, 'w') as f:
                        json.dump(response, f, indent=4)
                msg_dict = self._post_process_response(msg=response)
                self.update_price(response)
                # return msg_dict, response, payload["messages"]
                return msg_dict, response
            except Exception as e:
                print('\033[1;31m [WARNING]: \033[0m ')
                print(e)
                time.sleep(30)
                continue
        return None, None