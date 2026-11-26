# Correctness judge

LLM-judge rubric: does the assistant's final answer correctly and completely
address the user's question, using only information returned by the tools?

Choices: `correct` (accurate, complete, no invented details) vs `incorrect`
(wrong, incomplete, hallucinated, or ignores the question).

The judge prompt lives in the fenced block below; `{input}`, `{output}` and
`{trajectory_text}` are filled from trajectory-row dataframe columns.

```
Decide whether the assistant's final answer correctly and completely
addresses the user's question, using only information returned by the
catalog, inventory, order, and policy tools.

- correct: the answer is accurate, addresses what was asked, and does not
  invent product details, stock levels, order statuses, or policy terms.
- incorrect: the answer is wrong, incomplete, hallucinated, or ignores
  what the user asked.

## User question

{input}

## Assistant answer

{output}

## Tool calls made

{trajectory_text}
```
