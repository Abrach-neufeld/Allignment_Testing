import datetime
from openai import OpenAI
import pandas as pd

def run_trials(models,prompt_dict,number_of_trials,temperature=1,filepath=""):
    """
    Runs trials for the specified models and prompts and returns dataframe with response texts and trial info. If filepath is set, saves csv file.
    :param models: List of openAI models to prompt
    :param prompt_dict: Dictionary with keys of prompt names and values of prompt text to prompt models with
    :param number_of_trials: Number of trials per prompt/model combination to run.
    :param temperature: Temperature to pass to model. Default: 1
    :param filename: Filepath to save data in csv form. If not set, data is not saved. Default:""
    :returns: Dataframe with trial results and info.
    """
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
        if filepath!="":
            df.to_csv(filepath,index=False)
        return df