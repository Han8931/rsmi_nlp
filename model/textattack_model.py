import torch
import transformers
import textattack

from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult, FailedAttackResult
from textattack.attack_results.attack_result import AttackResult
import pdb

import numpy as np

class AttackSummary:
    def __init__(self, max_seq_length=256):
        self._succeed = 0
        self._fail = 0
        self._skipped = 0

        self.pert_ratio = []
        self.pert_ratio_fail = []
        self.fail_q = []
        self.succ_q = []

        self.max_seq_length = max_seq_length

    def __str__(self):
        return ' || '.join(['{}: {:.2f}'.format(key, value) for (key, value) in self.get_metric().items()])

    def __call__(self, result):
        assert isinstance(result, AttackResult)
        if isinstance(result, SuccessfulAttackResult):
            self._succeed += 1

            n_query, n_pert, n_words = self.text_analysis(result)
            pert_word_ratio =(n_pert/n_words)*100
            self.pert_ratio.append(pert_word_ratio)

            self.succ_q.append(n_query)

        elif isinstance(result, FailedAttackResult):
            self._fail += 1
            n_query = result.num_queries
            self.fail_q.append(n_query)

            n_query, n_pert, n_words = self.text_analysis(result)
            pert_word_ratio =(n_pert/n_words)*100
            self.pert_ratio_fail.append(pert_word_ratio)

        elif isinstance(result, SkippedAttackResult):
            self._skipped += 1

    def text_analysis(self, result):
        words_pert = result.original_result.attacked_text.all_words_diff(result.perturbed_result.attacked_text)
        n_pert = len(words_pert)
        n_words = len(result.original_result.attacked_text.words)
        n_query = result.num_queries

        return n_query, n_pert, n_words

    def get_metric(self, reset: bool = False):
        all_numbers = self._succeed + self._fail + self._skipped
        correct_numbers = self._succeed + self._fail

        if correct_numbers == 0:
            success_rate = 0.0
        else:
            success_rate = self._succeed / correct_numbers
        
        if all_numbers == 0:
            clean_accuracy = 0.0
            robust_accuracy = 0.0
        else:
            clean_accuracy = correct_numbers / all_numbers
            robust_accuracy = self._fail / all_numbers

        try:
            pert_ratio = np.mean(self.pert_ratio)
        except:
            pert_ratio=0

        try:
            pert_ratio_fail = np.mean(self.pert_ratio_fail)
        except:
            pert_ratio_fail=0

        if len(self.succ_q)==0:
            avg_succ_q = 0
        else:
            avg_succ_q = np.array(self.succ_q).mean()
        if len(self.fail_q)==0:
            avg_fail_q = 0
        else:
            avg_fail_q = np.array(self.fail_q).mean()

        avg_query = np.mean(self.succ_q+self.fail_q)
        
        if reset:
            self.reset()

        out_stat = {"C-Acc" : clean_accuracy*100, "A-Acc" : robust_accuracy*100, "ASR": success_rate*100, 
                "Pert": pert_ratio, "Avg-Q": avg_query, "AvgS-Q":avg_succ_q, "AvgF-Q":avg_fail_q} 

        return out_stat

    def reset(self) -> None:
        self._succeed += 1
        self._fail += 1
        self._skipped += 1        


def print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped):
    n_trials = num_successes+num_failed+num_skipped

    valid_try =num_successes+num_failed

    if valid_try==0:
        asr = 0
    else:
        asr = num_successes/(num_successes+num_failed)

    print("="*100, flush=True)
    print(f"Model: {args.load_model} || FileName: {f_name} || Attack: {args.attack_method}", flush=True) 
    if args.multi_mask:
        print(f"Multi_Mask: {args.multi_mask}", flush=True) 
    print(f"Trials: {n_trials} || # Success: {num_successes} || # Failed: {num_failed} || # Skipped: {num_skipped}", flush=True) 
    print(f"ASR: {asr:.4f} || AAcc: {num_failed/(n_trials):.4f}", flush=True) 
    print("="*100, flush=True)


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args=None):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.num_classes = args.num_classes

        self.args = args
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        model_device = next(self.model.parameters()).device

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)
            logits = outputs['logits']

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class CustomWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.num_classes = args.num_classes

        self.ensemble = True if args.num_ensemble>1 else False
        self.num_ensemble = args.num_ensemble
        self.multi_mask = args.multi_mask
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx
        self.hf_model = args.hf_model
        self.max_seq_length = args.max_seq_length

        self.binom_ensemble = args.binom_ensemble
        self.two_step = args.two_step
        self.binom_n_eval = args.binom_n_eval

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device 
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']

        indices, _ = self.model.grad_mask(input_ids, attention_mask, pred=None, mask_filter=True)
        self.model.zero_grad()           
        logits = self.model.two_step_ensemble(input_ids, attention_mask, indices, self.num_ensemble, self.binom_ensemble)

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.eval()

        if self.hf_model:
            embedding_layer = self.model.get_word_embeddings()
        else:
            embedding_layer = self.model.enc.encoder.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']


        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

