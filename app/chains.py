"""
DocuMind AI - LangChain Chains

Contains QA chain, Quiz generation chain, and Calculator agent.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory

from app.models import get_chat_model
from app.prompts import QA_PROMPT, CONDENSE_PROMPT, QUIZ_PROMPT
from app.database import get_database


class QARetrievalChain:
    """Manages the Question Answering retrieval chain with chat history."""

    def __init__(self):
        self.chat_model = get_chat_model()
        self.db = get_database()
        self.message_history = ChatMessageHistory()
        self._chain: Optional[BaseConversationalRetrievalChain] = None

    def _create_chain(self) -> BaseConversationalRetrievalChain:
        """Create the conversational retrieval chain."""
        def get_chat_history(chat_history_messages: List) -> str:
            """Format chat history for the prompt."""
            history_str = ""
            for msg in chat_history_messages:
                if hasattr(msg, "type") and msg.type == "human":
                    history_str += f"Human: {msg.content}\n"
                elif hasattr(msg, "type") and msg.type == "ai":
                    history_str += f"AI: {msg.content}\n"
            return history_str

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self._get_retriever(),
            condense_question_prompt=CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            get_chat_history=get_chat_history,
            verbose=True,
        )

        return qa_chain

    def _get_retriever(self):
        """Create a retriever from the vector store."""
        from langchain_community.vectorstores import Chroma
        from app.models import get_embedding_model

        embedding_model = get_embedding_model()
        db_path = "data/chroma_db"

        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model,
            collection_name="document_chunks"
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

    def get_chain(self) -> BaseConversationalRetrievalChain:
        """Get or create the retrieval chain."""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain

    def ask_question(self, question: str) -> Tuple[str, List[str]]:
        """
        Ask a question and get an answer with sources.

        Returns:
            Tuple of (answer, sources)
        """
        if not question.strip():
            return "Please enter a question.", []

        # Check for calculator-related keywords
        if self._is_math_question(question):
            return self._handle_calculator(question)

        # Get relevant chunks for sources display
        chunks = self.db.get_relevant_chunks(question, top_k=5)
        sources = [chunk[0][:200] + "..." if len(chunk[0]) > 200 else chunk[0]
                   for chunk in chunks]

        try:
            # Run the chain
            chain = self.get_chain()
            result = chain.invoke({
                "question": question,
                "chat_history": self.message_history.messages
            })

            # Extract the answer
            if isinstance(result, dict):
                answer = result.get("answer", str(result))
            else:
                answer = str(result)

            # Add to message history
            self.message_history.add_user_message(question)
            self.message_history.add_ai_message(answer)

            return answer, sources

        except Exception as e:
            return f"Error processing question: {str(e)}", []

    def _is_math_question(self, question: str) -> bool:
        """Check if the question likely involves calculations."""
        math_keywords = [
            "calculate", "calculation", "math", "sum", "total",
            "average", "percentage", "percent", "multiply", "divide",
            "add", "subtract", "plus", "minus", "times", "product",
            "cost", "price", "budget", "expense", "revenue", "profit"
        ]
        question_lower = question.lower()
        # Also check for mathematical expressions
        has_math_expr = bool(re.search(r'[\d+\-*/=()]+', question))
        return any(keyword in question_lower for keyword in math_keywords) or has_math_expr

    def _handle_calculator(self, question: str) -> Tuple[str, List[str]]:
        """Handle calculator/math questions using the math agent."""
        try:
            # Extract the actual calculation from the question
            # Look for patterns like "5 + 3", "10 * 2", etc.
            calc_pattern = r'([\d\s+\-*/().]+)'
            matches = re.findall(calc_pattern, question)

            if matches:
                # Try to find a valid expression
                for match in matches:
                    expression = match.strip()
                    try:
                        # Validate it looks like a math expression
                        if re.match(r'^[\d\s+\-*/().]+$', expression):
                            result = eval(expression)
                            return (
                                f"Calculation: {expression} = {result}\n\n"
                                f"I detected a math expression in your question. "
                                f"If you need more context from your documents about this calculation, "
                                f"please let me know!",
                                []
                            )
                    except:
                        continue

            # If no simple expression found, use the chat model for complex math
            response = self.chat_model.invoke(
                f"Please calculate the answer to: {question}. "
                f"If this involves unit conversions or complex math steps, show your work."
            )

            answer = response.content if hasattr(response, 'content') else str(response)
            return answer, []

        except Exception as e:
            return f"I can help with calculations, but encountered an issue: {str(e)}", []

    def clear_history(self):
        """Clear the chat history."""
        self.message_history.clear()

    def get_conversation_context(self) -> str:
        """Get formatted conversation context."""
        messages = self.message_history.messages
        context = ""
        for msg in messages[-10:]:  # Last 10 messages
            if hasattr(msg, "type"):
                if msg.type == "human":
                    context += f"Human: {msg.content}\n"
                elif msg.type == "ai":
                    context += f"AI: {msg.content}\n"
        return context


class QuizChain:
    """Manages quiz generation and evaluation."""

    def __init__(self):
        self.chat_model = get_chat_model()
        self.db = get_database()
        self.current_quiz: Optional[Dict[str, Any]] = None
        self.current_answers: Optional[Dict[int, str]] = None

    def generate_quiz(
        self,
        doc_id: int,
        quiz_type: str = "Multiple Choice",
        num_questions: int = 5
    ) -> str:
        """
        Generate a quiz based on document content.

        Args:
            doc_id: Document ID in the database
            quiz_type: Type of quiz (Multiple Choice, True/False, Short Answer)
            num_questions: Number of questions to generate

        Returns:
            Formatted quiz string
        """
        # Get document chunks
        chunks = self.db.get_document_chunks(doc_id)

        if not chunks:
            return "No document content available to generate quiz."

        # Combine chunks for context (limit to avoid token overflow)
        context = "\n\n".join(chunks[:10])  # Use first 10 chunks

        # Truncate if too long
        if len(context) > 8000:
            context = context[:8000] + "\n\n[Content truncated for quiz generation]"

        try:
            # Generate quiz using Runnable
            from langchain_core.output_parsers import StrOutputParser
            chain = self.chat_model | QUIZ_PROMPT | StrOutputParser()
            result = chain.invoke({
                "context": context,
                "quiz_type": quiz_type,
                "num_questions": num_questions
            })

            quiz_text = result if isinstance(result, str) else str(result)

            # Parse and store the quiz
            self.current_quiz = self._parse_quiz(quiz_text, quiz_type)
            self.current_answers = {}

            return quiz_text

        except Exception as e:
            return f"Error generating quiz: {str(e)}"

    def _parse_quiz(self, quiz_text: str, quiz_type: str) -> Dict[str, Any]:
        """Parse the generated quiz text into a structured format."""
        # Basic parsing - extract questions and answers
        quiz_data = {
            "type": quiz_type,
            "questions": [],
            "correct_answers": {}
        }

        # Split by Question markers
        question_blocks = re.split(r'\*\*Question \d+:\*\*', quiz_text)

        if len(question_blocks) > 1:
            for i, block in enumerate(question_blocks[1:], 1):
                quiz_data["questions"].append(block.strip())

        # Try to extract answers section
        answers_match = re.search(r'## ANSWERS\s*\n(.*)', quiz_text, re.DOTALL)
        if answers_match:
            answers_text = answers_match.group(1)
            # Parse individual answers
            for match in re.finditer(r'(\d+)\.\s*\[?([A-D]|[TF])\]?', answers_text):
                q_num = int(match.group(1))
                answer = match.group(2)
                quiz_data["correct_answers"][q_num] = answer

        return quiz_data

    def check_answers(self, user_answers: Dict[int, str]) -> Tuple[float, str, Dict[int, str]]:
        """
        Check user answers and calculate score.

        Args:
            user_answers: Dict of question_number -> user_answer

        Returns:
            Tuple of (score_percentage, feedback_string, correct_answers_dict)
        """
        if not self.current_quiz or "correct_answers" not in self.current_quiz:
            return 0, "No quiz available to check.", {}

        correct = self.current_quiz["correct_answers"]
        total = len(correct)

        if total == 0:
            return 0, "No correct answers found in the quiz.", {}

        correct_count = 0
        feedback_parts = []

        for q_num, correct_answer in correct.items():
            user_answer = user_answers.get(q_num, "").upper().strip()
            is_correct = user_answer == correct_answer.upper().strip()

            if is_correct:
                correct_count += 1
                feedback_parts.append(f"Question {q_num}: Correct!")
            else:
                feedback_parts.append(
                    f"Question {q_num}: Incorrect (Your answer: {user_answer}, "
                    f"Correct: {correct_answer})"
                )

        score = (correct_count / total) * 100
        feedback = "\n".join(feedback_parts)
        feedback += f"\n\nFinal Score: {correct_count}/{total} ({score:.1f}%)"

        return score, feedback, correct

    def get_current_quiz(self) -> Optional[Dict[str, Any]]:
        """Get the current quiz data."""
        return self.current_quiz

    def clear_quiz(self):
        """Clear the current quiz."""
        self.current_quiz = None
        self.current_answers = None


# Global chain instances
_qa_chain: Optional[QARetrievalChain] = None
_quiz_chain: Optional[QuizChain] = None


def get_qa_chain() -> QARetrievalChain:
    """Get or create the global QA chain instance."""
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = QARetrievalChain()
    return _qa_chain


def get_quiz_chain() -> QuizChain:
    """Get or create the global Quiz chain instance."""
    global _quiz_chain
    if _quiz_chain is None:
        _quiz_chain = QuizChain()
    return _quiz_chain


def reset_chains():
    """Reset all chain instances (useful when database is cleared)."""
    global _qa_chain, _quiz_chain
    _qa_chain = None
    _quiz_chain = None
