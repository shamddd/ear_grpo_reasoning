# EAR-GRPO Phase III Prompt & Chat Template Audit

## 1. Objective

To verify that instruction-tuned models (`Qwen/Qwen2.5-0.5B-Instruct`) receive properly formatted chat-template inputs with system, user, and assistant generation tokens, preventing prompt misalignment.

---

## 2. Formatted Prompt Structures

### A. System Message & User Query Format

```jinja2
<|im_start|>system
You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

---

## 3. Representative Formatted Prompt Example

### Question
> *Natalia sold clips to 48 of her friends in April, and then in May she sold half as many clips as in April. How many clips did Natalia sell altogether in April and May?*

### Fully Rendered Input Token Stream
```text
<|im_start|>system
You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then in May she sold half as many clips as in April. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
```

---

## 4. Special Tokens Verification

- `system_start`: `<|im_start|>system`
- `user_start`: `<|im_start|>user`
- `assistant_start`: `<|im_start|>assistant`
- `end_of_text`: `<|im_end|>`

### Verification Status
- Checked via `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.
- Verified 100% compliant with Qwen ChatML formatting.
