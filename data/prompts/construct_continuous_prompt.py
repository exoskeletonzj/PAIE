import csv


def read_roles(template_path):
    role_dict = {}

    with open(template_path, "r", encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            event_type_role, template = line
            
            event_type, role = event_type_role.split('_')
            if event_type not in role_dict:
                role_dict[event_type] = []
            role_dict[event_type].append(role)

    return role_dict


def create_continuous_prompt(role_dict, P=1):
    prompt_dict = dict()
    for event_type, role_list in sorted(role_dict.items()):
        prompt = []
        for role in role_list:
            prompt.extend(
                ["<{}_left_{}>".format(role, i) for i in range(P)] + [role] + ["<{}_right_{}>".format(role, i) for i in range(P)]
            )
        prompt = " ".join(prompt)
        prompt_dict[event_type] = prompt
    return prompt_dict


if __name__ == "__main__":
    # ACE2005
    template_path = "../dset_meta/description_ace.csv"
    new_prompt_path = "prompts_ace_continuous.csv"

    # # RAMS 1.0
    # template_path = "../dset_meta/description_rams.csv"
    # new_prompt_path = "prompts_rams_continuous.csv"

    # # WikiEvent
    # template_path = "../dset_meta/description_wikievent.csv"
    # new_prompt_path = "prompts_wikievent_continuous.csv"
    
    P = 1
    role_dict = read_roles(template_path)
    prompt_dict = create_continuous_prompt(role_dict, P)

    with open(new_prompt_path, 'w') as csvfile:
        writter = csv.writer(csvfile, delimiter=':', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for event_type, prompt in prompt_dict.items():
            writter.writerow([event_type, prompt])