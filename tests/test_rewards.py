from ear_grpo_reasoning.rewards import compute_math_reward, extract_answer


def test_extract_answer() -> None:
    assert extract_answer("The answer is 42. #### 42") == "42"
    assert extract_answer("So the total cost is $15.50") == "15.50"
    assert extract_answer("No numbers here") is None

    assert extract_answer(r"Therefore, \boxed{-3/4}.") == "-3/4"
    assert extract_answer("Final answer: 1,250") == "1250"


def test_compute_math_reward() -> None:
    completion_correct = "Step 1: 5 * 10 = 50. #### 50"
    completion_wrong = "Step 1: 5 * 10 = 40. #### 40"
    ground_truth = "50"

    assert compute_math_reward(completion_correct, ground_truth) == 1.0
    assert compute_math_reward(completion_wrong, ground_truth) == 0.0


def test_fraction_and_decimal_are_equivalent() -> None:
    assert compute_math_reward("#### 1/2", "#### 0.5") == 1.0


def test_malformed_and_infinite_answers_receive_zero() -> None:
    assert compute_math_reward("No numeric answer", "42") == 0.0
    assert compute_math_reward("#### 1/0", "42") == 0.0
