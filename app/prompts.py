"""
DocuMind AI - Prompt Templates

Contains all prompt templates for QA, Quiz generation, and other tasks.
"""

from langchain_core.prompts import PromptTemplate

# QA Chain Prompt
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.

Context from documents:
{context}

Chat history:
{chat_history}

Current question: {question}

Instructions:
1. Use the context above to answer the question accurately.
2. If the context doesn't contain enough information, say "I don't have enough information in the loaded documents to answer this question."
3. If you need to perform calculations, clearly show your work.
4. Be concise but thorough in your answer.
5. Mention which document(s) your answer is based on.

Answer:""",
    input_variables=["context", "chat_history", "question"],
)

# Condensed prompt for retrieval
CONDENSE_PROMPT = PromptTemplate(
    template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:""",
    input_variables=["chat_history", "question"],
)

# Quiz Generation Prompt
QUIZ_PROMPT = PromptTemplate(
    template="""You are an educational assistant that creates quizzes based on document content.

Document content:
{context}

Quiz type: {quiz_type}
Number of questions: {num_questions}

Instructions:
1. Create {num_questions} {quiz_type} questions based on the document content.
2. For Multiple Choice: Provide 4 options (A, B, C, D) with one correct answer.
3. For True/False: State a fact that can be clearly judged as true or false.
4. For Short Answer: Create open-ended questions that require brief descriptive answers.

Format your response as follows:

## QUIZ

**Question 1:** [Your question here]
A) Option A
B) Option B
C) Option C
D) Option D
Correct Answer: [Letter]

**Question 2:** [Your question here]
A) Option A
B) Option B
C) Option C
D) Option D
Correct Answer: [Letter]

[Continue for all questions]

## ANSWERS
1. [Correct answer letter]
2. [Correct answer letter]
[etc.]

Quiz:""",
    input_variables=["context", "quiz_type", "num_questions"],
)

# Calculator prompt
CALCULATOR_PROMPT = PromptTemplate(
    template="""The user wants to perform a calculation. Execute the calculation and provide the result.

Calculation requested: {question}

Result:""",
    input_variables=["question"],
)

# Summary prompt
SUMMARY_PROMPT = PromptTemplate(
    template="""Summarize the following document content briefly.

Content:
{context}

Summary (2-3 sentences):""",
    input_variables=["context"],
)
