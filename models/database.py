from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

db = SQLAlchemy()

# Define models
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

class Paper(db.Model):
    __tablename__ = 'papers'
    id = db.Column(db.Integer, primary_key=True)
    hash = db.Column(db.String(32), unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    model = db.Column(db.String(50), nullable=True)
    pdf_path = db.Column(db.String(255), nullable=False)
    evaluations = db.relationship('Evaluation', backref='paper', lazy=True)
    chats = db.relationship('Chat', backref='paper', lazy=True)
    applied_macros = db.relationship('AppliedMacro', backref='paper', lazy=True)

class Rubric(db.Model):
    __tablename__ = 'rubrics'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    criteria = db.relationship('RubricCriteria', backref='rubric', lazy=True)
    macros = db.relationship('FeedbackMacro', backref='rubric', lazy=True)

class RubricCriteria(db.Model):
    __tablename__ = 'rubric_criteria'
    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey('rubrics.id'), nullable=False)
    section_name = db.Column(db.String(255), nullable=False)
    criteria_text = db.Column(db.Text, nullable=False)
    weight = db.Column(db.Float, nullable=False, default=1.0)  # Weight for this criterion

class Evaluation(db.Model):
    __tablename__ = 'evaluations'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    criteria_id = db.Column(db.Integer, db.ForeignKey('rubric_criteria.id'))
    evaluation_text = db.Column(db.Text, nullable=False)

class Chat(db.Model):
    __tablename__ = 'chats'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())

class GradeDescriptors(db.Model):
    __tablename__ = 'grade_descriptors'
    id = db.Column(db.Integer, primary_key=True)
    range_start = db.Column(db.Integer, nullable=False)
    range_end = db.Column(db.Integer, nullable=False)
    descriptor_text = db.Column(db.Text, nullable=False)

class SavedFeedback(db.Model):
    __tablename__ = 'saved_feedback'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    consolidated_feedback = db.Column(db.Text)
    mark = db.Column(db.Float)
    updated_at = db.Column(db.DateTime, nullable=False, default=func.now())
    # Add relationship to criterion-specific feedback
    criterion_feedback = db.relationship('CriterionFeedback', backref='saved_feedback', lazy=True)

class CriterionFeedback(db.Model):
    """Stores core feedback for each criterion."""
    __tablename__ = 'criterion_feedback'
    id = db.Column(db.Integer, primary_key=True)
    saved_feedback_id = db.Column(db.Integer, db.ForeignKey('saved_feedback.id'), nullable=False)
    criteria_id = db.Column(db.Integer, db.ForeignKey('rubric_criteria.id'), nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)
    mark = db.Column(db.Float)  # Mark for this specific criterion
    updated_at = db.Column(db.DateTime, nullable=False, default=func.now())

class MacroCategory(db.Model):
    """Stores custom categories for feedback macros, specific to each rubric."""
    __tablename__ = 'macro_categories'
    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey('rubrics.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    # Add relationship to rubric
    rubric = db.relationship('Rubric', backref='macro_categories')
    # Add relationship to macros
    macros = db.relationship('FeedbackMacro', backref='category_ref', lazy=True)

class FeedbackMacro(db.Model):
    """Stores feedback macros that can be reused across papers with the same rubric."""
    __tablename__ = 'feedback_macros'
    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey('rubrics.id'), nullable=False)
    criteria_id = db.Column(db.Integer, db.ForeignKey('rubric_criteria.id'), nullable=True)  # Null means general macro
    name = db.Column(db.String(255), nullable=False)  # Short name for the macro (e.g., "poor thesis")
    text = db.Column(db.Text, nullable=False)  # The text to insert when the macro is used
    category_id = db.Column(db.Integer, db.ForeignKey('macro_categories.id'), nullable=True)  # Link to custom category
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())

class AppliedMacro(db.Model):
    """Tracks which macros have been applied to a specific paper."""
    __tablename__ = 'applied_macros'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    macro_id = db.Column(db.Integer, db.ForeignKey('feedback_macros.id'), nullable=False)
    applied_at = db.Column(db.DateTime, nullable=False, default=func.now())
    # Define relationship to FeedbackMacro
    macro = db.relationship('FeedbackMacro', backref='applications')

class ModerationSession(db.Model):
    """Stores moderation sessions for papers."""
    __tablename__ = 'moderation_sessions'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    original_feedback = db.Column(db.Text, nullable=False)  # The feedback before moderation
    moderated_feedback = db.Column(db.Text)  # The feedback after moderation
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, completed, rejected
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    completed_at = db.Column(db.DateTime)
    # Add relationship to paper
    paper = db.relationship('Paper', backref='moderation_sessions')
    # Add relationship to moderation results
    results = db.relationship('ModerationResult', backref='session', lazy=True)

class ModerationResult(db.Model):
    """Stores criterion-specific moderation results."""
    __tablename__ = 'moderation_results'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('moderation_sessions.id'), nullable=False)
    criteria_id = db.Column(db.Integer, db.ForeignKey('rubric_criteria.id'), nullable=False)
    result = db.Column(db.String(10), nullable=False)  # PASSES or FAILS
    reasoning = db.Column(db.Text)  # The reasoning behind the PASSES or FAILS decision
    moderated_feedback = db.Column(db.Text)  # The suggested feedback after moderation
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    # Add relationship to criteria
    criteria = db.relationship('RubricCriteria', backref='moderation_results')

class AIEvaluation(db.Model):
    """Stores AI evaluations for each criterion."""
    __tablename__ = 'ai_evaluations'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    criteria_id = db.Column(db.Integer, db.ForeignKey('rubric_criteria.id'), nullable=False)
    evaluation_text = db.Column(db.Text, nullable=False)  # The AI's evaluation
    mark = db.Column(db.Float)  # The AI's suggested mark for this criterion
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    # Add relationships
    paper = db.relationship('Paper', backref='ai_evaluations')
    criteria = db.relationship('RubricCriteria', backref='ai_evaluations')