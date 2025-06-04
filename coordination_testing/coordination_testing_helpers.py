import datetime
from openai import OpenAI
import pandas as pd

def run_trials(models,prompt_dict,number_of_trials,temperature=1,filename=""):
    client = OpenAI()
    trials=[]
    for model in models:
        print("Starting run for model:",model)
        for prompt_name in prompt_dict:
            print("Starting run for prompt", prompt_name)
            for i in range(0,number_of_trials):
                prompt_text=prompt_dict[prompt_name]
                response = client.responses.create(
                    model=model,
                    input=prompt_text,
                    temperature=temperature
                )
                trial={"model":model,"prompt_name":prompt_name,"prompt_text":prompt_text,"temperature":temperature,"response_text":response.output_text}
                trials.append(trial)
        df=pd.DataFrame(trials)
        df['date']=datetime.datetime.now().date()
        if filename!="":
            df.to_csv(f'Outputs/{filename}.csv',index=False)
        return df