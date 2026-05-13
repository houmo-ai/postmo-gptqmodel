import argparse
from types import SimpleNamespace

import torch

from scripts_qwen35moe.eval import evalscope_eval


def test_build_task_config_uses_work_dir_for_cache_resume():
    args = argparse.Namespace(
        work_dir="/tmp/evalscope-test",
        eval_batch_size=1,
        max_tokens=512,
        limit=None,
        debug=False,
        use_cache=True,
    )

    task_cfg = evalscope_eval._build_task_config(
        args=args,
        model_api=object(),
        model_id="qwen35moe-test",
        resolved_datasets=["gpqa_diamond"],
    )

    assert task_cfg.use_cache == args.work_dir
    assert task_cfg.no_timestamp is True


def test_build_task_config_disables_cache_resume_when_requested():
    args = argparse.Namespace(
        work_dir="/tmp/evalscope-test",
        eval_batch_size=1,
        max_tokens=512,
        limit=None,
        debug=False,
        use_cache=False,
    )

    task_cfg = evalscope_eval._build_task_config(
        args=args,
        model_api=object(),
        model_id="qwen35moe-test",
        resolved_datasets=["gpqa_diamond"],
    )

    assert task_cfg.use_cache is None


def test_disable_gptqmodel_torch_compile_restores_class_dequantize_weight():
    class FakeTorchQuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dequantize_weight = lambda num_itr=1: "compiled"

        def dequantize_weight(self, num_itr: int = 1):
            return f"orig:{num_itr}"

    FakeTorchQuantLinear.__module__ = "gptqmodel.nn_modules.qlinear.torch"

    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qlinear = FakeTorchQuantLinear()

    model = Container()
    assert model.qlinear.dequantize_weight() == "compiled"

    changed = evalscope_eval._disable_gptqmodel_torch_compile(model)

    assert changed == 1
    assert model.qlinear.dequantize_weight(3) == "orig:3"


def test_parse_dataset_subset_overrides_resolves_aliases():
    overrides = evalscope_eval._parse_dataset_subset_overrides(
        [
            "math500=Level 1",
            "math_500=Level 3",
            "gpqa:default",
        ]
    )

    assert overrides == {
        "math_500": ["Level 1", "Level 3"],
        "gpqa_diamond": ["default"],
    }


def test_parse_dataset_subset_overrides_supports_ceval():
    overrides = evalscope_eval._parse_dataset_subset_overrides(
        [
            "ceval=advanced_mathematics",
            "ceval=law",
        ]
    )

    assert overrides == {
        "ceval": ["advanced_mathematics", "law"],
    }


def test_build_task_config_preserves_dataset_subset_args():
    args = argparse.Namespace(
        work_dir="/tmp/evalscope-test",
        eval_batch_size=1,
        max_tokens=512,
        limit=None,
        debug=False,
        use_cache=True,
    )

    task_cfg = evalscope_eval._build_task_config(
        args=args,
        model_api=object(),
        model_id="qwen35moe-test",
        resolved_datasets=["math_500"],
        dataset_args={"math_500": {"subset_list": ["Level 1", "Level 3"]}},
    )

    assert task_cfg.dataset_args == {"math_500": {"subset_list": ["Level 1", "Level 3"]}}


def test_build_model_api_kwargs_disables_thinking_by_default():
    args = argparse.Namespace(
        model="/tmp/model",
        device_map="auto",
        dtype="bfloat16",
        max_tokens=512,
        enable_thinking=False,
    )

    kwargs = evalscope_eval._build_model_api_kwargs(args, "qwen35moe-test")

    assert kwargs["enable_thinking"] is False


def test_build_model_api_kwargs_respects_enable_thinking_flag():
    args = argparse.Namespace(
        model="/tmp/model",
        device_map="auto",
        dtype="bfloat16",
        max_tokens=512,
        enable_thinking=True,
    )

    kwargs = evalscope_eval._build_model_api_kwargs(args, "qwen35moe-test")

    assert kwargs["enable_thinking"] is True


def test_is_multiple_choice_answer_prompt_detects_gpqa_format():
    prompt = (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\n"
        "A) foo\nB) bar\nC) baz\nD) qux"
    )

    assert evalscope_eval._is_multiple_choice_answer_prompt(prompt) is True


def test_finalize_multiple_choice_answer_requests_followup_when_missing_answer():
    class StubAPI:
        model_name = "stub-model"

        def __init__(self):
            self.rendered = []

        def _render_chat(self, messages):
            self.rendered.append(messages)
            return "\n".join(f"{message.role}: {message.content}" for message in messages)

        def _generate_prompt_texts(self, prompts, config):
            prompt = prompts[0]
            if "provide only the final multiple-choice answer" in prompt:
                return [("ANSWER: C", 1, 3)]
            return [("Long reasoning without final answer", 10, 512)]

    api = StubAPI()
    messages = [SimpleNamespace(role="user", content="A) foo\nB) bar\nC) baz\nD) qux\nANSWER: [LETTER]")]

    finalized = evalscope_eval.Qwen35MoEEvalScopeAPI._finalize_multiple_choice_answer(
        api,
        messages=messages,
        prompt=messages[0].content,
        text="Long reasoning without final answer",
        config=evalscope_eval.GenerateConfig(max_tokens=512, do_sample=False, top_p=None, top_k=None),
    )

    assert finalized == "ANSWER: C"


def test_extract_multiple_choice_answer_accepts_prefilled_option_text():
    assert evalscope_eval._extract_multiple_choice_answer("\nB) 6.3x10^-7 M") == "B"


def test_extract_multiple_choice_answer_normalizes_lowercase_letters():
    assert evalscope_eval._extract_multiple_choice_answer("ANSWER: a") == "A"
    assert evalscope_eval._extract_multiple_choice_answer("\nc) choice text") == "C"


def test_finalize_multiple_choice_answer_uses_prefilled_recovery_when_followup_stays_meta():
    class StubAPI:
        model_name = "stub-model"

        def __init__(self):
            self.rendered = []
            self.prefill_calls = []

        def _render_chat(self, messages):
            self.rendered.append(messages)
            return "\n".join(f"{message.role}: {message.content}" for message in messages)

        def _generate_prompt_texts(self, prompts, config):
            prompt = prompts[0]
            if "provide only the final multiple-choice answer" in prompt:
                return [("The user wants me to answer a multiple-choice question", 1, 32)]
            return [("Long reasoning without final answer", 10, 512)]

        def _generate_prefilled_assistant_text(self, messages, assistant_prefix, config):
            self.prefill_calls.append((messages, assistant_prefix, config.max_tokens))
            return ("\nB) 6.3x10^-7 M", 1, 4)

    api = StubAPI()
    messages = [SimpleNamespace(role="user", content="A) foo\nB) bar\nC) baz\nD) qux\nANSWER: [LETTER]")]

    finalized = evalscope_eval.Qwen35MoEEvalScopeAPI._finalize_multiple_choice_answer(
        api,
        messages=messages,
        prompt=messages[0].content,
        text="Long reasoning without final answer",
        config=evalscope_eval.GenerateConfig(max_tokens=512, do_sample=False, top_p=None, top_k=None),
    )

    assert api.prefill_calls
    assert finalized == "ANSWER: B"


def test_finalize_multiple_choice_answer_uses_working_prefill_prompt_shape():
    class StubAPI:
        model_name = "stub-model"

        def _render_chat(self, messages):
            return "\n".join(f"{message.role}: {message.content}" for message in messages)

        def _generate_prompt_texts(self, prompts, config):
            prompt = prompts[0]
            if "provide only the final multiple-choice answer" in prompt:
                return [("The user wants me to answer a multiple-choice question", 1, 32)]
            return [("Long reasoning without final answer", 10, 512)]

        def _generate_prefilled_assistant_text(self, messages, assistant_prefix, config):
            prompt_text = messages[0].content
            if "Respond with only the final answer in the form ANSWER: X." in prompt_text:
                return ("\nB) 6.3x10^-7 M", 1, 4)
            return ("Question. The concentration of calcium ions", 1, 8)

    api = StubAPI()
    messages = [SimpleNamespace(role="user", content="A) foo\nB) bar\nC) baz\nD) qux\nANSWER: [LETTER]")]

    finalized = evalscope_eval.Qwen35MoEEvalScopeAPI._finalize_multiple_choice_answer(
        api,
        messages=messages,
        prompt=messages[0].content,
        text="Long reasoning without final answer",
        config=evalscope_eval.GenerateConfig(max_tokens=512, do_sample=False, top_p=None, top_k=None),
    )

    assert finalized == "ANSWER: B"


def test_is_boxed_answer_prompt_detects_math500_format():
    prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    assert evalscope_eval._is_boxed_answer_prompt(prompt) is True


def test_finalize_boxed_answer_requests_followup_when_missing_boxed_answer():
    class StubAPI:
        model_name = "stub-model"

        def __init__(self):
            self.rendered = []

        def _render_chat(self, messages):
            self.rendered.append(messages)
            return "\n".join(f"{message.role}: {message.content}" for message in messages)

        def _generate_prompt_texts(self, prompts, config):
            prompt = prompts[0]
            if (
                "provide only the final answer wrapped in \\boxed{}" in prompt
                and "Please reason step by step" not in prompt
                and "What is $DE$?" in prompt
            ):
                return [("Final answer: \\boxed{28}", 1, 5)]
            return [("Long reasoning without boxed answer", 10, 512)]

    api = StubAPI()
    messages = [
        SimpleNamespace(
            role="user",
            content="Suppose $\\sin D = 0.7$. What is $DE$?\nPlease reason step by step, and put your final answer within \\boxed{}.",
        )
    ]

    finalized = evalscope_eval.Qwen35MoEEvalScopeAPI._finalize_boxed_answer(
        api,
        messages=messages,
        prompt=messages[0].content,
        text="Long reasoning without boxed answer",
        config=evalscope_eval.GenerateConfig(max_tokens=512, do_sample=False, top_p=None, top_k=None),
    )

    assert finalized == "Final answer: \\boxed{28}"


def test_generate_prompt_texts_strips_left_padding_before_decoding_generated_tokens():
    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, prompts, return_tensors="pt", padding=True, **kwargs):
            assert prompts == ["short", "long"]
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": torch.tensor(
                    [
                        [0, 0, 11, 12],
                        [21, 22, 23, 24],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [0, 0, 1, 1],
                        [1, 1, 1, 1],
                    ]
                ),
            }

        def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return " ".join(str(int(token)) for token in token_ids.tolist())

    class DummyModel:
        def generate(self, **kwargs):
            return torch.tensor(
                [
                    [0, 0, 11, 12, 101, 102],
                    [21, 22, 23, 24, 201, 202],
                ]
            )

    api = SimpleNamespace(
        tokenizer=DummyTokenizer(),
        tokenizer_call_args={},
        input_device=torch.device("cpu"),
        model=DummyModel(),
    )

    outputs = evalscope_eval.Qwen35MoEEvalScopeAPI._generate_prompt_texts(
        api,
        prompts=["short", "long"],
        config=evalscope_eval.GenerateConfig(max_tokens=2, do_sample=False, top_p=None, top_k=None),
    )

    assert outputs == [
        ("101 102", 2, 2),
        ("201 202", 4, 2),
    ]
