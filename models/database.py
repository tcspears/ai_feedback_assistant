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
    model = db.Column(db.String(50), nullable=False)
    pdf_path = db.Column(db.String(255), nullable=False)
    evaluations = db.relationship('Evaluation', backref='paper', lazy=True)
    chats = db.relationship('Chat', backref='paper', lazy=True)

class Rubric(db.Model):
    __tablename__ = 'rubrics'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())
    criteria = db.relationship('RubricCriteria', backref='rubric', lazy=True)

class RubricCriteria(db.Model):
    __tablename__ = 'rubric_criteria'
    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey('rubrics.id'), nullable=False)
    section_name = db.Column(db.String(255), nullable=False)
    criteria_text = db.Column(db.Text, nullable=False)

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
    additional_feedback = db.Column(db.Text)
    consolidated_feedback = db.Column(db.Text)
    mark = db.Column(db.Float)
    updated_at = db.Column(db.DateTime, nullable=False, default=func.now())