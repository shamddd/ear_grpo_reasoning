from ear_grpo_reasoning.rewards import compute_math_reward


def test_verifier_comprehensive_cases() -> None:
    # 30 Explicit test cases: 15 Valid, 15 Invalid/Tricky
    valid_cases = [
        ("The answer is 42.", "42", True),
        ("#### 42", "42", True),
        ("####  100 ", "100", True),
        (r"\boxed{42}", "42", True),
        ("So the final count is 3.55.", "3.55", True),
        ("Total price is -15 dollars.", "-15", True),
        ("First we add 10 + 20 = 30. #### 30", "30", True),
        ("The value is 1/2 or 0.5.", "0.5", True),
        ("Answer: 12345", "12345", True),
        ("#### -42.5", "-42.5", True),
        ("Result is 0.", "0", True),
        ("Final answer: 99", "99", True),
        ("The total cost is $50", "50", True),
        ("Thus x = 12.", "12", True),
        ("#### 0", "0", True),
    ]

    invalid_cases = [
        ("The answer is wrong.", "42", False),
        ("No numbers here!", "10", False),
        ("First step: 10 + 20 = 30 #### 99", "42", False),
        ("The answer is 10.", "42", False),
        ("#### 99", "42", False),
        ("", "42", False),
        ("#### 15", "42", False),
        ("#### -10", "10", False),
        ("Answer: 50", "100", False),
        ("Calculated 100 but ground truth was 200 #### 100", "200", False),
        ("#### 15", "16", False),
        ("Result is 0.5", "0.6", False),
        ("#### 1000", "100", False),
        ("#### 1.23", "1.24", False),
        ("#### -5", "5", False),
    ]

    for text, gt, expected in valid_cases + invalid_cases:
        is_correct = compute_math_reward(text, gt) > 0.0
        assert is_correct == expected, (
            f"Failed case: text={text!r}, gt={gt!r}, expected={expected}, got={is_correct}"
        )
