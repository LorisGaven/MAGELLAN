'''

'''

def generate_prompt(o, g):
    prompt = f'Goal: {g}\n'
    prompt += o
    prompt += '\nAction: '
    return prompt