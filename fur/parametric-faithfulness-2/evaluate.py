import nltk

import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint

from models import load_model_and_tokenizer
from dataload import DATASETS

import re
import json
import random
import time
from openai import OpenAI

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("

ANSWER_LETTERS = ["A", "B", "C", "D", "E"] # No MCQA dataset considered has more than 5

def answer_probabilities(model, tokenizer, dh, instance):
    device = model.device
    with torch.no_grad():
        n_options = len(dh.get_answer_letters(instance))
        answer_letters = ANSWER_LETTERS[:n_options]
        answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]

        prompt = dh.make_bowman_demonstration(instance)
        answer_inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
        
        answer_output = model.generate(input_ids=answer_inputs, max_new_tokens=10,
                                        output_scores=True,
                                        temperature=0., do_sample=False,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id)
    
        # 2.1 obtain letter completion probabilities
        first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
        letter_probs = first_token_probs[answer_indices]
        predicted_letter_index = torch.argmax(letter_probs).item()
        letter_probs = letter_probs.detach().cpu().float().numpy()
    
        # 2.2 take only newly generated output
        answer_output = answer_output[0][0]
        answer_new_output = answer_output[answer_inputs.shape[-1]:]
        answer_new_output_text = tokenizer.decode(answer_new_output)
        
        return answer_new_output_text, letter_probs, predicted_letter_index

def complete(model, tokenizer, prompt, max_new_tokens=300, temperature=0., do_sample=False, split_newline=True, thinking=False):
  do_sample = temperature > 0. # overwrite
  with torch.no_grad():
    device = model.device
    
    max_new_tokens = 4096 if thinking else max_new_tokens
    # apply chat template if the prompt is a list of messages
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        print('chat applied prompt=', prompt)
    inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id)


    # 2 take only newly generated output
    output = outputs[0][0]
    new_output = output[inputs.shape[-1]:]
    new_output_text = tokenizer.decode(new_output)
    # Where to split the CoT from the answer? For thinking models we have <think> tags and displayed reasoning. For now, let's split at the end of the </think> tag.
    if thinking:
        # check if </think> tag is present
        if "</think>" in new_output_text:
           new_output_text = new_output_text.strip().split("</think>")[0].strip()
           print('new_output_text after splitting at </think> =', new_output_text)
        else:
           print("Warning: </think> tag not found in output, using full output as CoT")
    # for regular llms split at \n\n
    elif split_newline:
       new_output_text = new_output_text.strip().split("\n\n")[0]
    
    return new_output_text
  

def letter_completion(model, tokenizer, prompt, N):
  with torch.no_grad():
    device = model.device
    answer_letters = ANSWER_LETTERS[:N]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]


    # Step 5: make answer prompt
    answer_inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    answer_output = model.generate(input_ids = answer_inputs, max_new_tokens=20,
                                    output_scores=True, return_dict_in_generate=True,
                                      pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = tokenizer.decode(answer_new_output)
    
    return letter_probs, predicted_letter_index

#MULTI-TURN semantic and syntactic paraphrases
def generate_paraphrases_for_cot_multiturn(
    segmented_cot,
    model="gpt-5.1-2025-11-13",
    temperature=0.7,
    max_retries=3,
    system_prompt_override=None,
    system_prompt_turn2_override=None,
    num_p=1
):
    """
    Multi-turn paraphrase generator.

    Turn 1: Generate semantic paraphrases of each segment.
    Turn 2: Using the original + semantics, generate syntactic paraphrases for each semantic.
    
    Returns:
        {
            segment: {
                "sem": [...],
                "syn": [...]
            },
            ...
        }
    """
    client = OpenAI()

    # ----------------------------------------
    # DEFAULT SYSTEM PROMPTS FOR BOTH TURNS
    # ----------------------------------------
    default_system_prompt_turn1 = """
    You produce semantic paraphrases.
    Return ONLY valid JSON with key "sem", whose value is a list of strings.
    No meta-commentary.
    """

    default_system_prompt_turn2 = """
    You produce syntactic paraphrases.
    For each semantic paraphrase, produce one syntactic paraphrase.
    Return ONLY valid JSON with key "syn", whose value is a list of syntactic paraphrases
    aligned by index to the semantic list.
    No meta-commentary.
    """

    system_prompt_turn1 = system_prompt_override or default_system_prompt_turn1
    system_prompt_turn2 = system_prompt_turn2_override or default_system_prompt_turn2

    # ----------------------------------------
    # Function to call the model safely
    # ----------------------------------------
    def call_model(messages):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                print(f"[WARN] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        return None

    # ----------------------------------------
    # MAIN LOOP
    # ----------------------------------------
    results = {}

    # Do this multiturn generation for each step in the CoT
    for segment in segmented_cot:

        # ------------------
        # TURN 1: semantic
        # ------------------
        user_prompt_turn1 = f"""
        Produce exactly {num_p} semantic paraphrases of this segment. Maintain meaning.
        SEGMENT:
        {segment}

        Return JSON of the form:
        {{
           "sem": [ ... ]
        }}
        """

        sem_output = call_model([
            {"role": "system", "content": system_prompt_turn1},
            {"role": "user", "content": user_prompt_turn1}
        ])

        # print("sem_output=", sem_output)

        if sem_output is None or "sem" not in sem_output:
            print(f"[ERROR] Semantic generation failed for segment:\n{segment}")
            continue

        semantics = sem_output["sem"]

        # ------------------
        # TURN 2: syntactic
        # ------------------
        # set to paraphrase
        target_set = semantics + [segment]
        user_prompt_turn2 = f"""
        Below is a list of semantic paraphrases.
        For each semantic paraphrase, produce ONE syntactic paraphrase that preserves meaning but presents a different ordering.

        SEMANTIC PARAPHRASES:
        {json.dumps(target_set)}

        Return JSON of the form:
        {{
           "syn": [ ... ]   # same length as semantic list
        }}
        """

        syn_output = call_model([
            {"role": "system", "content": system_prompt_turn2},
            {"role": "user", "content": user_prompt_turn2}
        ])

        # print('syn_output=', syn_output)
        if syn_output is None or "syn" not in syn_output:
            print(f"[ERROR] Syntactic generation failed for segment:\n{segment}")
            continue

        syntactics = syn_output["syn"]


        # Store results
        results[segment] = {
            "sem": semantics,
            "syn": syntactics,
        }

    # print("results=", results)

    return results


# New prompt using gpt-5 and getting sem and syntactic paraphrases
def generate_paraphrases_for_cot(
    segmented_cot,
    model="gpt-5.1-2025-11-13",
    temperature=0.7,
    max_retries=3,
    reasoning={"reasoning_effort": "none"},
    system_prompt_override=None,
    user_prompt_override=None
):
    """
    Given a segmented chain of thought (list of strings),
    return a dictionary mapping each step to:
        syn: list of syntactic paraphrases
        sem: list of semantic paraphrases
        sem_syn: list of syntactic paraphrases of each semantic version
    """

    # ---------------------------
    # SYSTEM PROMPT
    # ---------------------------
    default_system_prompt = """
    You are a precise paraphrasing assistant for reasoning traces.

    For each input step, you must return:
    - syn: syntactic paraphrases (same words, different structure)
    - sem: semantic paraphrases (same meaning, different words)
    - sem_syn: syntactic paraphrases of the semantic ones

    Each value MUST be a list of strings.

    Return ONLY valid JSON. No explanatory text.

    STYLE REQUIREMENTS:
    - Maintain the original meaning exactly.
    - No meta-commentary (e.g., “here is a paraphrase”, “this means that…”).
    - Reasoning-style tone.
    """

    system_prompt = system_prompt_override or default_system_prompt

    # ---------------------------
    # USER PROMPT (with in-context example)
    # ---------------------------
    default_user_prompt = f"""
    Below is an example showing the required structure:

    EXAMPLE SEGMENT:
    "Step 1: The engineers are testing different building designs to see how they respond during an earthquake."

    EXAMPLE OUTPUT:
    {{
    "Step 1: The engineers are testing different building designs to see how they respond during an earthquake.": {{
        "syn": [
        "Different building designs are being tested by the engineers to assess their earthquake response."
        ],
        "sem": [
        "The engineers are examining various building designs to determine their performance in an earthquake."
        ],
        "sem_syn": [
        "To determine their performance in an earthquake, the engineers are examining various building designs.
        ]
    }}
    }}

    Now produce 10 semantic paraphrases for these segments:
    {json.dumps({"segmented_cot": segmented_cot}, indent=2)}
    """

    user_prompt = user_prompt_override or default_user_prompt

    # ---------------------------
    # API CALL
    # ---------------------------
    client = OpenAI()

    if model == "gpt-5-2025-08-07":
       temperature=1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            # print("Exact model version:", response.model)
            parsed = json.loads(response.choices[0].message.content)
            return parsed

        except Exception as e:
            print(f"[WARN] Paraphrasing failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3)

    return {}


def generate_dataset_cots_cues(model_id, tokenizer, dataset_id, temperature, sentencize=True, paraphrase_segments=False, consistency=False, consistency_num=None, cue_type=None, thinking=False, num_p=1):
    print(f"Generating CoTs with and without cues for {model_id}, {dataset_id}, sentencize={sentencize}")
    model, _ = load_model_and_tokenizer(model_id)
    DH = DATASETS[dataset_id]
    _, valid, test = DH.get_dataset_splits()
    if dataset_id == 'sqa': test = valid # SQA test doesn't have answers

    instance_info = []
    
    for idx, instance in tqdm(enumerate(test)): # only take 250 instances from each dataset for comparability
        if idx >= 2: break
        # if idx >= 250: break # 250 _nocot
        _, nocot_probs, _ = answer_probabilities(
            model, tokenizer, DH, instance)
        
        # DO THIS PART TWICE, ONCE WITHOUT CUE AND ONCE WITH CUE
        cot_prompt = DH.make_cot_prompt(instance, ct=thinking)

        # randomly choose letter option to bias towards from amongst the incorrect options
        choices = [c for c in DH.get_answer_letters(instance) if c != DH.correct_answer_letter(instance)]
        # randomly choosing what the cued letter should be
        cued_letter = random.choice(choices)
        print('cued_letter=', cued_letter)
        cued_cot_prompt = DH.make_biased_cot_prompt(instance, cue=cue_type, target_letter=cued_letter, ct=thinking)

        # completion without cue
        cot = complete(model, tokenizer, cot_prompt, temperature=temperature, thinking=thinking)
        # completion with cue
        cued_cot = complete(model, tokenizer, cued_cot_prompt, temperature=temperature, thinking=thinking)
        
        cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, instance, cot, thinking=thinking)
        cued_cot_probs, _ = generation_fixed_cot(model, tokenizer, DH, instance, cued_cot, thinking=thinking)

        segmented_cot = None
        if sentencize:
            # segmenting the first CoT for unlearning
            segmented_cot = nltk.sent_tokenize(cot)
            segmented_cued_cot = nltk.sent_tokenize(cued_cot)
            if paraphrase_segments:
                print(f"Generating paraphrases for CoTs steps using on {dataset_id}, sentencize={sentencize}")
                # paraphrased_segments = generate_paraphrases_for_cot(segmented_cot)
                paraphrased_segments = generate_paraphrases_for_cot_multiturn(segmented_cot, num_p=num_p)
        inst_details = {
            'id': instance[DH.id_key],
            'question': instance[DH.q_key],
            'correct_letter': DH.correct_answer_letter(instance),
            'cued_letter': cued_letter,
            'cot_prompt': DH.make_cot_prompt(instance),
            'cued_cot_prompt': cued_cot_prompt,
            'cot': cot,
            'cued_cot': cued_cot,
            'options': DH.get_answer_choices(instance),
            'cot_probs': cot_probs.tolist(),
            'cued_cot_probs': cued_cot_probs.tolist(),
            'segmented_cot': segmented_cot,
            'segmented_cot_with_cue': segmented_cued_cot,
            'raw_instance': instance,
        }
        if idx == 1:
          pprint(inst_details)
        instance_info.append(inst_details)
    return instance_info

def generate_dataset_cots(model_id, tokenizer, dataset_id, temperature, sentencize=True, paraphrase_segments=False, consistency=False, consistency_num=None, num_p=1):
    print(f"Generating new CoTs for {model_id}, {dataset_id}, sentencize={sentencize}, paraphrase_segments={paraphrase_segments}")
    model, _ = load_model_and_tokenizer(model_id)
    DH = DATASETS[dataset_id]
    _, valid, test = DH.get_dataset_splits()
    if dataset_id == 'sqa': test = valid # SQA test doesn't have answers

    instance_info = []
    
    for idx, instance in tqdm(enumerate(test)): # only take 250 instances from each dataset for comparability
        if idx >= 30: break 
        # if idx >= 250: break # 250 _nocot
        _, nocot_probs, _ = answer_probabilities(
            model, tokenizer, DH, instance)

        # Consistency prompt the original model to get consistency_num completions
        cots = []
        cots_probs = []
        if consistency:
        #    print(f"Consistency Prompting original model with {consistency_num}")
           assert consistency_num > 0
           for i in range(consistency_num):
            cot_prompt = DH.make_cot_prompt(instance)
            cot = complete(model, tokenizer, cot_prompt, temperature=0.7)
            cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, instance, cot)
            segmented_cot = None
            paraphrased_segments = None
            cots.append(cot)
            cots_probs.append(cot_probs.tolist())
        else:
            #TODO FIXME
            cot_prompt = DH.make_cot_prompt(instance)
            cot = complete(model, tokenizer, cot_prompt, temperature=temperature)

            cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, instance, cot)
            segmented_cot = None
            paraphrased_segments = None
            cots.append(cot)
            cots_probs.append(cot_probs.tolist())
        if sentencize:
            # segmenting the first CoT for unlearning
            segmented_cot = nltk.sent_tokenize(cots[0])
            if paraphrase_segments:
                print(f"Generating paraphrases for CoTs steps using on {dataset_id}, sentencize={sentencize}")
                # paraphrased_segments = generate_paraphrases_for_cot(segmented_cot)
                paraphrased_segments = generate_paraphrases_for_cot_multiturn(segmented_cot, num_p=num_p)
        inst_details = {
            'id': instance[DH.id_key],
            'question': instance[DH.q_key],
            'correct_letter': DH.correct_answer_letter(instance),
            'cot_prompt': DH.make_cot_prompt(instance),
            # 'cot': cot,
            # this is a list of cots
            'cot': cots,
            'options': DH.get_answer_choices(instance),
            'nocot_probs': nocot_probs.tolist(),
            # 'cot_probs': cot_probs.tolist(),
            # this is a list of lists
            'cot_probs': cots_probs,
            'segmented_cot': segmented_cot,
            'paraphrased_segments': paraphrased_segments, 
            'raw_instance': instance,
        }
        if idx == 1:
          pprint(inst_details)
        instance_info.append(inst_details)
    return instance_info

def generation_fixed_cot(model, tokenizer, dh, instance, cot_text, thinking=False):
  with torch.no_grad():
    device = model.device
    n_options = len(dh.get_answer_letters(instance))
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]
    cot_prompt = dh.make_cot_prompt(instance, ct=thinking)

    # Step 5: make answer prompt
    # answer_prompt = dh.make_answer_prompt(cot_prompt + cot_text)
    answer_prompt = dh.make_answer_prompt(cot_prompt, cot_text, ct=thinking)
    # print("answer_prompt=", answer_prompt)
    if isinstance(answer_prompt, list):
        answer_prompt = tokenizer.apply_chat_template(answer_prompt, tokenize=False, continue_final_message=True)
        # print('chat applied prompt=', answer_prompt)

    # print('answer_prompt=', answer_prompt+'\n')
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    answer_output = model.generate(input_ids=answer_inputs, max_new_tokens=20,
                                    output_scores=True, return_dict_in_generate=True,
                                      pad_token_id=tokenizer.pad_token_id)

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = tokenizer.decode(answer_new_output)
    # print('answer_new_output_text=', answer_new_output_text)
    
    return letter_probs, predicted_letter_index

def generate(model, tokenizer, instance):
  with torch.no_grad():
    device = model.device
    # Step 1: make answer prompt
    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]

    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options']])
    answer_prompt = answer_prompt + f"{DELIM}Answer: ("
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    answer_output = model.generate(input_ids=answer_inputs, max_new_tokens=10,
                                    output_scores=True,
                                    temperature=0., do_sample=False,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = tokenizer.decode(answer_new_output)
    
    return answer_new_output_text, letter_probs, predicted_letter_index

def get_cot_prompt(instance):
  LTSBS = "Assistant: Let's think step by step:\n"
  DELIM = "\n\n"
  answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
  return answer_prompt

def generate_cot(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    # "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n"
    LTSBS = "Assistant: Let's think step by step:\n"

    device = model.device

    # 1. Make CoT prompt
    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    cot_output = model.generate(input_ids=answer_inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2 take only newly generated output
    cot_output = cot_output[0][0]
    cot_new_output = cot_output[answer_inputs.shape[-1]:]
    cot_new_output_text = tokenizer.decode(cot_new_output)
    cot_new_output_text = cot_new_output_text.strip().split("\n\n")[0] 
    
    return cot_new_output_text, answer_prompt

def cot_generate(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    device = model.device

    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                      for L in answer_letters]

    cot_text, cot_prompt = generate_cot(model, tokenizer, instance,
                          max_new_tokens=max_new_tokens, temperature=temperature,do_sample=do_sample)

    prefix = cot_prompt + cot_text
    # 1. Make CoT prompt
    DELIM = "\n"
    answer_prompt = DELIM.join([prefix, BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX])
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)
    # 2. Generate answer
    answer_output = model.generate(input_ids=answer_inputs,
                                    max_new_tokens=10,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2.1 obtain letter completion probabilities
    first_token_probs = torch.softmax(answer_output['scores'][0][0], dim=-1)
    letter_probs = first_token_probs[answer_indices]
    predicted_letter_index = torch.argmax(letter_probs).item()
    letter_probs = letter_probs.detach().cpu().float().numpy()

    # 2.2 take only newly generated output
    answer_output = answer_output[0][0]
    answer_new_output = answer_output[answer_inputs.shape[-1]:]
    answer_new_output_text = tokenizer.decode(answer_new_output)
    
    return answer_new_output_text, letter_probs, predicted_letter_index,cot_text, cot_prompt

def completion_probabilities(model, tokenizer, prefix, targets):
    device = model.device
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device) # [1, T]
    prefix_length = prefix_ids.size(-1)

    n_sequences = len(targets)
    n_prefix_ids = prefix_ids.repeat(n_sequences, 1)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Convert targets to tensors, concat to inputs
    target_ids = tokenizer(targets, padding=True, return_tensors="pt").input_ids.to(device)

    # Count lengths of individual target sequences for scaling
    non_pad = target_ids != pad_token_id
    lengths = torch.count_nonzero(non_pad, dim=-1)

    # Stack inputs
    input = torch.hstack([
       n_prefix_ids,target_ids[:,:-1] # Exclude last target token from input
                          ])

    # Fwd pass
    outputs = model.forward(input, return_dict=True) # logits = B, T, V
    relevant_logits = outputs['logits'][:,prefix_length-1:]
    # Any benefits from logsoftmax if we want the actual probability in the end?
    # Yes, numeric stability
    token_probs = torch.log_softmax(relevant_logits, dim=-1)

    # Set pad probabilities to one for .prod()
    token_probs[:,:,pad_token_id] = 1.
    target_probs = token_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_probs = target_probs.squeeze(1)
    seq_probs = torch.sum(target_probs, dim=1)  # prod if not in logspace
    length_penalty = model.generation_config.length_penalty
    seq_probs /= lengths**length_penalty # if not logspace seq_probs /= lengths

    return seq_probs
