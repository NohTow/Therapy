import os
os.environ['TRANSFORMERS_CACHE'] = "./cache"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OPTForCausalLM, GPT2TokenizerFast, RepetitionPenaltyLogitsProcessor
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm

import csv
import argparse
import logging
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--c",
    default=None,
    type=float,
    required=True,
    help="The exploration constant"
)
parser.add_argument(
    "--alpha",
    default=1,
    type=float,
    help="The parameter that guide the exploration toward likelihood or value"
)
parser.add_argument(
    "--temperature",
    default=None,
    type=float,
    required=True,
    help="Temperature when calculating priors"
)

parser.add_argument(
    "--penalty",
    default=1.0,
    type=float,
    help="Penalty factor to apply to repetitions"
)

parser.add_argument(
    "--num_it",
    default=50,
    type=int,
    required=False,
    help="Number of MCTS iteration for one token"
)

parser.add_argument(
    "--batch_size",
    default=25,
    type=int,
    required=False,
    help="Number of prompts used for generation at once"
)

args = parser.parse_args()

logging.basicConfig(
    format="%(message)s",
    level=logging.WARNING,
    filename=("test_ag.log".format(args.c, args.temperature, args.penalty, args.num_it, args.alpha))
)
logger = logging.getLogger(__name__)

        


print("loading GPT model")
gpt =  OPTForCausalLM.from_pretrained("facebook/opt-125m")
gpt.eval()
gpt.to("cuda")
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("facebook/opt-125m")
tokenizer_gpt.padding_side = "left"
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token 
eos_token_id = gpt.config.eos_token_id
vocab_size = tokenizer_gpt.vocab_size
print("GPT model loaded")


from typing import Optional
if not os.path.exists("log"):
    os.makedirs("log")


black_box = pickle.load(open("reproduction/models/agnews_reglog", 'rb'))
vectorizer = pickle.load(open("reproduction/vectorizer/agnews_reglog", 'rb'))


def get_values(tokens_ids, labels):
        propositions = tokenizer_gpt.batch_decode(tokens_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tokenizer_res = vectorizer.transform(propositions)
        predictions = black_box.predict_proba(tokenizer_res)
        # print(propositions)
        # print(predictions[np.arange(len(labels)), labels])
        return predictions[np.arange(len(labels)), labels]
    

def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor


def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    out_tensor = sequences[0].new_full(out_dims, padding_value, device="cuda")
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor


def root_fun(original_input, prompt_masks, labels, temperature, repetition_penalty):
    # Forward pass of GPT-2 to get priors and states
    model_inputs = gpt.prepare_inputs_for_generation(original_input.input_ids, attention_mask=original_input.attention_mask, use_cache=True)
    with torch.no_grad():
        outputs = gpt(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        states = outputs.past_key_values
        prompt_masked_input_ids = torch.clone(original_input.input_ids)
        #penalizing an unused token
        prompt_masked_input_ids[prompt_masks]=14827
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        priors = F.softmax(priors, dim=-1).cpu().numpy()
        
    # Use of our discriminator to get values
    values = get_values(original_input.input_ids, labels)
   

    return priors, values, states

def rec_fun(states, token_ids, attention_masks, prompt_masks, labels, temperature, repetition_penalty):
    # Forward pass of GPT-2 to get priors and states
    model_inputs = gpt.prepare_inputs_for_generation(token_ids, attention_mask=attention_masks, use_cache=True, past=states)
    # model_inputs = gpt.prepare_inputs_for_generation(token_ids[:,[-1]], attention_mask=attention_masks[:,[-1]], use_cache=True, past=states)
    with torch.no_grad():
        outputs = gpt(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        next_states = outputs.past_key_values 
        prompt_masked_input_ids = torch.clone(token_ids)
        #penalizing an unused token
        prompt_masked_input_ids[prompt_masks]=14827
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        priors = F.softmax(priors, dim=-1).cpu().numpy()
       

    # Use of our discriminator to get values
    values = get_values(token_ids, labels)

    return priors, values, next_states



class BatchedMCTS():
    def __init__(self, root_fun, rec_fun, batch_size, num_simulations, num_actions, num_sparse_actions, pb_c_init, temperature, alpha, penalty):
        self._batch_size = batch_size
        self._num_simulations = num_simulations
        self._num_actions = num_actions
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        self._pb_c_init = pb_c_init
        self._temperature = temperature
        self.alpha = alpha

        self._root_fun = root_fun # a function called at the root
        self._rec_fun = rec_fun # a function called in the tree
        self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)
        # self._labels = torch.zeros((batch_size, 2), dtype=torch.uint8, device="cuda")
        self._labels = np.zeros((batch_size, ), dtype=np.uint8)
        self._prompt_lengths = torch.zeros(batch_size, dtype=torch.uint8, device="cuda")

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=np.bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._states = {}
        self._token_ids = {}
        self._attention_mask = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()
        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._states = {}
        self._token_ids = {} # Indexed by tuples (batch index, node index)
        self._attention_mask = {}
    
    def set_labels(self, labels): 
        self._labels = labels
    
    def set_prompt_lengths(self, lengths): 
        self._prompt_lengths = lengths    
    
    def search(self, original_input):
        self._reset_tree()

        prompt_masks = torch.arange(len(original_input.input_ids[0]), device="cuda")[None, :] < torch.unsqueeze(torch.sum(original_input.attention_mask ==0, dim=1).add(self._prompt_lengths) - 1, 1)
        # Evaluate the root.
        prior, values, states = self._root_fun(original_input, prompt_masks, self._labels, self._temperature, self._repetition_penalty)

       
        self._adaptive_min_values = values
        self._adaptive_max_values = values + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, values, states, original_input.input_ids, original_input.attention_mask, np.full(self._batch_size, False, dtype=np.bool))

       
        

        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        for sim in range(self._num_simulations):
            node_indices, actions = self.simulate()
            next_node_index = sim + 1 # root is 0, therefore we offset by 1.
            self.expand(node_indices, actions, next_node_index)
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

        # print(self.dense_visit_counts())
        return self.dense_visit_counts()
        # return self.dense_scores()
        # return self.dense_mean_scores()
    
    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        # print(root_visit_counts)
        return dense_visit_counts
    
    def dense_scores(self):
        root_index = 0
        root_scores = self._children_values[:, root_index, :]
        dense_root_scores = np.zeros((self._batch_size, self._num_actions))
        dense_root_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_scores
        root_visit_counts = self._children_visits[:, root_index, :]
        # print(root_visit_counts)
        return dense_root_scores

    def dense_mean_scores(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        root_scores = self._children_values[:, root_index, :]
        root_mean_scores = root_scores / root_visit_counts
        dense_mean_scores = np.zeros((self._batch_size, self._num_actions))
        dense_mean_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_mean_scores
        return dense_mean_scores

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)
    
    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :] # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)
        node_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1) 
        # (B, A)
        
        # Remap values between 0 and 1.
        node_value_score = node_children_values 
        # print(node_value_score[0])
        # node_value_score = (node_value_score != 0.) * node_value_score + (node_value_score == 0.) * self._adaptive_min_values[:, None]
        # node_value_score = (node_value_score - self._adaptive_min_values[:, None]) / (self._adaptive_max_values[:, None] - self._adaptive_min_values[:, None])
        
        node_uct_score = node_value_score + node_policy_score # (B, A)
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    def get_states_from_node(self, b, n, d): 
        # print(d)
        state_array = [None] * d
        # state = torch.clone(self._states[(b, n)])
        state_array[d-1] = self._states[(b, n)]
        while n!=0:
            n = self._parents[(b, n)]
            d -= 1
            state_array[d-1] = self._states[(b, n)]
        result = torch.cat(state_array, 3)
        return result
 
    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""

      
        # Retrieve token ids for nodes to be evaluated.
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, eos_token_id)
        attention_masks = pad_sequences_to_left([self._attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)
        children_priors = np.array([self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        depths = [self._depth[(b, n)]+1 for b, n in enumerate(node_indices)]
        previous_node_is_terminal = self._is_terminal[self._batch_range, node_indices[self._batch_range]] # (B)
        
        states_tensor = pad_sequences_to_left_states([self.get_states_from_node(b, n, depths[b].item()) for b, n in enumerate(node_indices)], 0, max_len=len(tokens_ids[0]))
        states = tuple(tuple(type_of_value for type_of_value in layer) for layer in states_tensor)
        
        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        # Add actions to list of tokens and extend attention mask by 1
        tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
        attention_masks = torch.cat((attention_masks, torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device="cuda"), 1)), dim = 1)
        prompt_masks = torch.arange(len(tokens_ids[0]), device="cuda")[None, :] < torch.unsqueeze(torch.sum(attention_masks==0, dim=1).add(self._prompt_lengths) - 1, 1)
        

        # Check if expanded nodes are terminal 
        expanded_node_is_terminal = dense_actions == eos_token_id 

        # Evaluate nodes.
        (prior, values, next_states) = self._rec_fun(states, tokens_ids, attention_masks, prompt_masks, self._labels, self._temperature, self._repetition_penalty)
       
        # Create the new nodes.
        self.create_node(next_node_index, prior, likelihoods*children_priors, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal)
        
        # Update the min and max values arrays
        # self._adaptive_min_values = np.minimum(self._adaptive_min_values, values**(self.alpha) * (likelihoods*children_priors)**(1-self.alpha))
        # self._adaptive_max_values = np.maximum(self._adaptive_max_values, values**(self.alpha) * (likelihoods*children_priors)**(1-self.alpha))
        self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)
        
        # Update tree topology.
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1
        
    def create_node(self, node_index, prior, likelihoods, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        prior = prior[self._batch_range[:, None], prior_topk_indices] # (B, A)
        
        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices
        
        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        self._likelihoods[:, node_index] = likelihoods

        # raw_values = values**(self.alpha) * likelihoods**(1-self.alpha)
        raw_values = values
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        key_value_tensor = torch.stack(list(torch.stack(list(next_states[i]), dim=0) for i in range(len(next_states))), dim=0)
        if(node_index == 0):
            for b in range(len(tokens_ids)):
                self._states[(b, node_index)] = torch.clone(key_value_tensor[:, :, b])
        else:
            for b in range(len(tokens_ids)):
                self._states[(b, node_index)] = torch.clone(key_value_tensor[:, :, b, :, -1:])

        # Updates tokens ids
        for b, token_ids in enumerate(tokens_ids):
            self._token_ids[(b, node_index)] = token_ids
        
        # Updates attention masks
        for b, attention_mask in enumerate(attention_masks):
            self._attention_mask[(b, node_index)] = attention_mask


    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                return
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = not_root_mask * (self._values[self._batch_range, parents] *
                self._visit_counts[self._batch_range, parents] + leaf_values) / (self._visit_counts[self._batch_range,
                parents] + 1.0) + root_mask * self._values[self._batch_range, parents]
            
            # self._values[self._batch_range, parents] = not_root_mask * (np.maximum(self._values[self._batch_range, parents],leaf_values)) + root_mask * self._values[self._batch_range, parents]

            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range,node_indices] + root_mask * self._children_values[self._batch_range, parents, actions]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            node_indices = parents

def main():
    generated_counter = 0
    samples_size = 3000
    batch_size = args.batch_size
    labels = np.zeros((batch_size, ), dtype=np.uint8)
    prompt_texts = [None] * batch_size
    MCTS = BatchedMCTS(root_fun, rec_fun, batch_size=batch_size, num_simulations=args.num_it, num_actions=vocab_size+1, num_sparse_actions=50, pb_c_init=args.c, temperature = args.temperature, alpha=args.alpha, penalty=args.penalty)
    samples_pbar = tqdm(total = 4 * samples_size, desc="Samples generated")
    fieldnames = ['text', 'label']
    with open("ag_news_mcts_{}_{}_{}_{}_{}.csv".format(args.c, args.temperature, args.penalty, args.num_it, args.alpha), "w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for i in range(4):
            generated_counter = 0
            labels.fill(i)
            prompt_texts = [""] * batch_size
            MCTS.set_labels(labels)
            while(generated_counter + batch_size <= samples_size): 
                original_input = tokenizer_gpt(prompt_texts, return_tensors="pt", padding="longest", add_special_tokens=True, truncation=True).to("cuda")
                # Randomly generating first few tokens to get different initial prompts
                with torch.no_grad():
                        output_sequences = gpt.generate(original_input.input_ids, attention_mask=original_input.attention_mask, min_length=7, max_length=7, do_sample=True, top_p=0.9, num_return_sequences=1)
                original_input = tokenizer_gpt(tokenizer_gpt.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True), return_tensors="pt", padding="longest", add_special_tokens=True).to("cuda")
                tokens_pbar = tqdm(total = 97, desc="Tokens generated")
                for j in range(0, 97):
                    res_search = MCTS.search(original_input)
                    original_input.input_ids = torch.cat((original_input.input_ids, torch.unsqueeze(torch.cuda.LongTensor(np.argmax(res_search,axis=1)),1)), dim = 1)
                    original_input.attention_mask = torch.cat((original_input.attention_mask, torch.unsqueeze(torch.ones(batch_size, dtype=torch.long, device="cuda"),1)), dim = 1)
                    # To print tokens at each generation step
                    # prompt_texts = [tokenizer_gpt.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True) for token_ids in original_input.input_ids]
                    # print(prompt_texts)
                    tokens_pbar.update(1)

                final_texts = tokenizer_gpt.batch_decode(original_input.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                res_row = {}
                res_row["label"] = str(i)
                for text in final_texts:
                    res_row["text"] = text.replace("\n", "").replace("<|endoftext|>", "") + "<|endoftext|>"
        
                generated_counter += batch_size
                samples_pbar.update(batch_size)
                    



if __name__ == "__main__":
    main()