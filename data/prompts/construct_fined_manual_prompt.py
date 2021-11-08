import csv
import ipdb


def read_definition(definition_path):
    definition_dict = {}

    with open(definition_path, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            event_type_role, definition = line
            
            event_type, role = event_type_role.split('_')
            if event_type not in definition_dict:
                definition_dict[event_type] = []
            definition_dict[event_type].append(role+"? "+definition)
            
    return definition_dict


def read_prompt(prompt_path):
    with open(prompt_path) as f:
        lines = f.readlines()
    prompts = dict()
    for line in lines:
        if not line:
            continue
        event_type, prompt = line.split(":")
        prompts[event_type] = prompt
    return prompts


def create_fined_manual_prompt(definition_dict, old_prompt_dict):
    prompt_dict = dict()
    for event_type, definition_list in sorted(definition_dict.items()):
        prompt = old_prompt_dict[event_type]
        # ipdb.set_trace()
        prompt = prompt.strip().lstrip("prompt start, ").rstrip(",end")
        for definition in definition_list:
            prompt += '. ' + definition
        prompt = "<prompt start> " + prompt + "<prompt end>"
        prompt_dict[event_type] = prompt

    return prompt_dict


definition_path = "../dset_meta/ontology_definition_ace.csv"
old_prompt_path = "./prompts_ace_full.csv"
new_prompt_path = "./prompts_ace_full_and_def.csv"

definition_dict = read_definition(definition_path)
old_prompt_dict = read_prompt(old_prompt_path)
new_prompt_dict = create_fined_manual_prompt(definition_dict, old_prompt_dict)

with open(new_prompt_path, 'w') as csvfile:
    writter = csv.writer(csvfile, delimiter=':', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for event_type, prompt in new_prompt_dict.items():
        writter.writerow([event_type, prompt])

