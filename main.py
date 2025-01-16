# Flask backend (app.py)
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, send_file
from flask_dropzone import Dropzone
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
import PyPDF2
from openai import OpenAI
import yaml
from markupsafe import escape, Markup
import markdown
import sqlite3
import json
import uuid
from collections import OrderedDict
import hashlib
from datetime import datetime
from anthropic import Anthropic
import anthropic
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import re
import cProfile
import pstats
import io
from functools import wraps
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
import csv
from io import StringIO
from io import BytesIO
from services.llm_service import LLMService
from models.database import (
    db, User, Paper, Rubric, RubricCriteria, Evaluation, 
    Chat, GradeDescriptors, SavedFeedback
)
        self.sections[name] = content
        # Add to section_order if not already present
from utils.prompt_builder import StructuredPrompt

def adapt_datetime(ts):
    return ts.isoformat()

sqlite3.register_adapter(datetime, adapt_datetime)

def load_config(app, config_file='settings.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    app.config.update(config)

app = Flask(__name__)
app.secret_key = app.config.get('SECRET_KEY', 'fallback_secret_key')
load_config(app)
dropzone = Dropzone(app)

client_openai = OpenAI(api_key=app.config['OPENAI_API_KEY'])
client_anthropic = Anthropic(api_key=app.config['ANTHROPIC_API_KEY'])

# Set up LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)  # Initialize db with app


def create_admin_user(username="admin", password="admin"):
    with app.app_context():
        # Check if admin user already exists
        admin = User.query.filter_by(username=username).first()
        if not admin:
            admin = User(
                username=username,
                password=generate_password_hash(password),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user '{username}' created successfully")
        else:
            print(f"Admin user '{username}' already exists")


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 time-consuming calls
        print('Profile data:')
        print(s.getvalue())
        return result
    return wrapper

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a logger
    logger = logging.getLogger('api_logger')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create a file handler
    log_filename = f'logs/api_calls_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
api_logger = setup_logging()

# Add a test log message at app startup
api_logger.info("Logging system initialized")


@app.template_filter('markdown')
def markdown_filter(text):
    if text:
        return Markup(markdown.markdown(text, extensions=['extra']))
    return ''

# Example of refactored route using SQLAlchemy
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully.')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            print(f"Login failed for user: {username}")  # Add debugging
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

@app.route('/paper/<file_hash>')
@login_required
def paper(file_hash):
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    # Get the rubric ID for the current paper through its evaluations
    current_rubric_id = (db.session.query(RubricCriteria.rubric_id)
                        .join(Evaluation, Evaluation.criteria_id == RubricCriteria.id)
                        .filter(Evaluation.paper_id == paper.id)
                        .first())
    
    # Get all papers that use the same rubric (including current)
    if current_rubric_id:
        related_papers = (Paper.query
                         .join(Evaluation)
                         .join(RubricCriteria)
                         .filter(RubricCriteria.rubric_id == current_rubric_id[0])
                         .distinct()
                         .order_by(Paper.filename.collate('NOCASE'))
                         .all())
    else:
        related_papers = [paper]  # Just show current paper if no rubric found

    # Format papers for template, marking current paper
    formatted_related = [(p.hash, p.filename, p.hash == file_hash) 
                        for p in related_papers]
    
    # Simplified query to get evaluations
    evaluations = (Evaluation.query
                  .filter_by(paper_id=paper.id)
                  .order_by(Evaluation.criteria_id.nullsfirst())
                  .all())
    
    # Format evaluations for template
    md = markdown.Markdown(extensions=['extra'])
    formatted_evaluations = []
    
    # Process evaluations
    for eval in evaluations:
        if eval.criteria_id is None:
            # Summary evaluation - convert markdown to HTML
            formatted_evaluations.append({
                'id': 'summary',
                'section_name': '_summary',
                'evaluation_text': Markup(md.convert(eval.evaluation_text))
            })
        else:
            # Get criteria info - keep raw markdown
            criteria = RubricCriteria.query.get(eval.criteria_id)
            if criteria:
                formatted_evaluations.append({
                    'id': eval.id,
                    'section_name': criteria.section_name,
                    'evaluation_text': eval.evaluation_text
                })
    
    # Get chat history
    chats = [(chat.user_message, chat.ai_response) for chat in 
             Chat.query.filter_by(paper_id=paper.id).order_by(Chat.created_at).all()]
    
    # Get saved feedback (don't convert to HTML)
    saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
    
    # Get the rubric name
    rubric_name = None
    if evaluations:
        first_eval = evaluations[0]
        if first_eval.criteria_id:
            criteria = RubricCriteria.query.get(first_eval.criteria_id)
            if criteria and criteria.rubric:
                rubric_name = criteria.rubric.name
    
    # Get the upload time for this file
    upload_time = paper.created_at
    
    # Calculate average grading time for this rubric
    avg_time = (db.session.query(func.avg(
        func.strftime('%s', SavedFeedback.updated_at) - 
        func.strftime('%s', Paper.created_at)
    ))
    .select_from(Paper)  # Explicitly specify the starting point
    .join(SavedFeedback)  # Join to SavedFeedback
    .join(Evaluation, Evaluation.paper_id == Paper.id)  # Join to Evaluation with explicit condition
    .join(RubricCriteria, RubricCriteria.id == Evaluation.criteria_id)  # Join to RubricCriteria with explicit condition
    .filter(
        RubricCriteria.rubric_id == current_rubric_id[0],
        SavedFeedback.updated_at.isnot(None)
    ).scalar())
    
    # Format average time as HH:MM:SS
    if avg_time:
        avg_seconds = int(avg_time)  # Now avg_time is already in seconds
        average_time = f"{avg_seconds // 3600:02d}:{(avg_seconds % 3600) // 60:02d}:{avg_seconds % 60:02d}"
    else:
        average_time = None

    return render_template('paper.html',
                         filename=paper.filename,
                         full_text=paper.full_text,
                         model=paper.model,
                         evaluations=formatted_evaluations,
                         chats=chats,
                         file_hash=file_hash,
                         related_papers=formatted_related,
                         rubric_name=rubric_name,
                         saved_feedback=saved_feedback,
                         saved_additional_feedback=saved_feedback.additional_feedback if saved_feedback else None,
                         saved_consolidated_feedback=saved_feedback.consolidated_feedback if saved_feedback else None,
                         pdf_path=paper.pdf_path.replace('static/', ''),
                         upload_time=upload_time.isoformat(),
                         average_time=average_time)

# Initialize the service with API clients
llm_service = LLMService(
    openai_client=client_openai,
    anthropic_client=client_anthropic
)

# Example of refactored chat route
@app.route('/chat/<file_hash>', methods=['POST'])
@login_required
def chat(file_hash):
    data = request.json
    user_message = data['message']
    model = data['model']
    
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    chat_history = Chat.query.filter_by(paper_id=paper.id).order_by(Chat.created_at).all()
    
    # Get grade descriptors from database
    descriptors = GradeDescriptors.query.order_by(GradeDescriptors.range_start.desc()).all()
    descriptors_text = "\n".join([
        f"{d.range_start}-{d.range_end}%: {d.descriptor_text}"
        for d in descriptors
    ])
    
    # Create initial context prompt using StructuredPrompt
    prompt = StructuredPrompt()
    prompt.add_section("context", 
        "You are an AI assistant specialized in discussing academic essays. "
        "You have access to both the full essay text and the grading criteria.")
    
    prompt.add_section("essay_content", paper.full_text)
    
    prompt.add_section("grading_framework", 
        "Here is how essays are evaluated at different grade levels:\n" + 
        (descriptors_text if descriptors_text else "No grade descriptors available."))
    
    prompt.add_section("instructions", 
        "Please keep this essay and grading framework in mind during our conversation. "
        "When discussing the essay's quality or suggesting improvements, consider these "
        "grading criteria and reference specific parts of the text when relevant.")

    initial_context = prompt.build()
    system_msg = "You are an expert in academic writing assessment..."
    
    # Build messages array
    messages = [{"role": "user", "content": initial_context}]
    
    # Add chat history
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.user_message})
        messages.append({"role": "assistant", "content": chat.ai_response})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    try:
        ai_message = llm_service.generate_response(
            model=model,
            messages=messages,
            system_msg=system_msg
        )
        
        # Save chat message
        new_chat = Chat(
            paper_id=paper.id,
            user_message=user_message,
            ai_response=ai_message
        )
        db.session.add(new_chat)
        db.session.commit()
        
        return jsonify({"response": ai_message})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/clear_chat/<file_hash>', methods=['POST'])
@login_required
def clear_chat(file_hash):
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    Chat.query.filter_by(paper_id=paper.id).delete()
    db.session.commit()
    return jsonify({"success": True})


@app.route('/update_filename/<file_hash>', methods=['POST'])
@login_required
def update_filename(file_hash):
    data = request.json
    new_filename = data['filename']
    
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    paper.filename = new_filename
    db.session.commit()
    
    return jsonify({"success": True})


@app.route('/save_rubric', methods=['POST'])
@login_required
def save_rubric():
    data = request.json
    name = data['name']
    description = data['description']
    criteria = data['criteria']
    rubric_id = data.get('rubric_id')
    
    try:
        if rubric_id:
            # Update existing rubric
            rubric = Rubric.query.get_or_404(rubric_id)
            rubric.name = name
            rubric.description = description
            
            # Delete existing criteria
            RubricCriteria.query.filter_by(rubric_id=rubric_id).delete()
        else:
            # Create new rubric
            rubric = Rubric(name=name, description=description)
            db.session.add(rubric)
            db.session.flush()  # Get the rubric_id before committing
        
        # Add new criteria
        for criterion in criteria:
            new_criterion = RubricCriteria(
                rubric_id=rubric.id,
                section_name=criterion['name'],
                criteria_text=criterion['description']
            )
            db.session.add(new_criterion)
        
        db.session.commit()
        return jsonify({"success": True})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})


def generate_evaluation(text, section_name, criteria_text, model):
    """Generate an evaluation for a section of text using the specified AI model."""
    
    # Create prompt using StructuredPrompt class
    prompt = StructuredPrompt()
    
    # Add initial task summary
    prompt.add_section("initial_task_summary", 
        "You are evaluating a section of an academic essay according to specific criteria.")
    
    # Add essay text
    prompt.add_section("essay_to_evaluate", text)
    
    # Add detailed instructions with subsections
    prompt.add_section("detailed_instructions", "", subsections={
        "section_focus": section_name,
        "evaluation_criteria": criteria_text,
        "formatting_requirements": """Format your response in Markdown, using:
- Headers (##) for main sections
- Lists (- or *) for key points
- Bold (**) for emphasis on important elements
- Line breaks between paragraphs"""
    })
    
    # Add specific analysis request
    prompt.add_section("analysis_request", 
        "Please provide an evaluation that begins with your categorization of the essay "
        "(e.g. Fail, Satisfactory, Good, Very Good, Excellent), followed by your "
        "justification for that categorization, and then relevant feedback for the student.")

    final_prompt = prompt.build()
    system_msg = "You are an experienced essay evaluator who provides constructive feedback to postgraduate students. Please format all of your responses as valid Markdown."

    api_logger.info(f"Sending prompt for {section_name} to API:")
    api_logger.info(f"System Message: {system_msg}")
    api_logger.info(f"Prompt:\n{final_prompt}\n{'='*50}")
    
    try:
        if model.startswith('claude'):
            response = client_anthropic.messages.create(
                model=model,
                system=system_msg,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=1000
            )
            evaluation = response.content[0].text.strip()
            api_logger.info(f"Claude API Response for {section_name}:\n{evaluation}\n{'='*50}")
        else:
            response = client_openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": final_prompt}
                ]
            )
            evaluation = response.choices[0].message.content.strip()
            api_logger.info(f"OpenAI API Response for {section_name}:\n{evaluation}\n{'='*50}")
        
        evaluation = re.sub(r'<[^>]+>', '', evaluation)
        return evaluation
    
    except Exception as e:
        print(f"Error generating evaluation: {str(e)}")
        return f"Error generating evaluation for {section_name}: {str(e)}"

@app.route('/get_rubric/<int:rubric_id>')
@login_required
def get_rubric(rubric_id):
    try:
        rubric = Rubric.query.get_or_404(rubric_id)
        criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).order_by(RubricCriteria.id).all()
        
        return jsonify({
            "success": True,
            "rubric": {
                "name": rubric.name,
                "description": rubric.description
            },
            "criteria": [
                {"section_name": c.section_name, "criteria_text": c.criteria_text} 
                for c in criteria
            ]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/save_mark/<file_hash>', methods=['POST'])
@login_required
def save_mark(file_hash):
    try:
        data = request.json
        mark = data['mark']
        
        if not (0 <= mark <= 100):
            return jsonify({"success": False, "error": "Mark must be between 0 and 100"})
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        
        if saved_feedback:
            saved_feedback.mark = mark
            saved_feedback.updated_at = datetime.now()
        else:
            saved_feedback = SavedFeedback(
                paper_id=paper.id,
                mark=mark
            )
            db.session.add(saved_feedback)
        
        db.session.commit()
        return jsonify({"success": True})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})


@app.route('/generate_consolidated_feedback', methods=['POST'])
@login_required
def generate_consolidated_feedback():
    try:
        data = request.json
        additional_feedback = data['additional_feedback']
        file_hash = data['file_hash']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get all evaluations for this paper (excluding summary)
        evaluations = (Evaluation.query
                      .join(RubricCriteria)
                      .filter(
                          Evaluation.paper_id == paper.id,
                          Evaluation.criteria_id.isnot(None)
                      )
                      .order_by(RubricCriteria.id)
                      .all())
        
        # Build consolidated feedback string
        consolidated_parts = ["<core_feedback>", additional_feedback, "</core_feedback>" "\n<additional_feedback>"]
        
        # Add each criteria-specific feedback with its section name as a subheading
        for eval in evaluations:
            criteria = RubricCriteria.query.get(eval.criteria_id)
            consolidated_parts.extend([
                f"\n## {criteria.section_name}",
                eval.evaluation_text
            ])
        
        consolidated_feedback = "\n\n".join(consolidated_parts)
        consolidated_feedback = consolidated_feedback + "\n</additional_feedback>"
        
        # Save the consolidated feedback
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if saved_feedback:
            saved_feedback.additional_feedback = additional_feedback
            saved_feedback.consolidated_feedback = consolidated_feedback
            saved_feedback.updated_at = datetime.now()
        else:
            saved_feedback = SavedFeedback(
                paper_id=paper.id,
                additional_feedback=additional_feedback,
                consolidated_feedback=consolidated_feedback
            )
            db.session.add(saved_feedback)
        
        db.session.commit()
        
        return jsonify({"consolidated_feedback": consolidated_feedback})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/save_additional_feedback/<file_hash>', methods=['POST'])
@login_required
def save_additional_feedback(file_hash):
    data = request.json
    additional_feedback = data['additional_feedback']
    
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    try:
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if saved_feedback:
            saved_feedback.additional_feedback = additional_feedback
            saved_feedback.updated_at = datetime.now()
        else:
            saved_feedback = SavedFeedback(
                paper_id=paper.id,
                additional_feedback=additional_feedback
            )
            db.session.add(saved_feedback)
        
        db.session.commit()
        return jsonify({"success": True})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

def remove_personal_identifiers(text):
    """
    Removes personally-identifying information from text.
    Currently removes:
    - Student ID numbers in the format B###### (e.g., B123456)
    - Course codes in the format XXXX##### (e.g., COMP12345)
    
    Args:
        text (str): The input text to process
        
    Returns:
        str: Text with personal identifiers removed
    """
    # Remove student IDs (B######)
    text = re.sub(r'B\d{6}', '[STUDENT_ID]', text)
    
    # Remove course codes (e.g., COMP12345)
    text = re.sub(r'[A-Z]{4}\d{5}', '[COURSE_ID]', text)
    
    return text

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        f = request.files.get('file')
        model = request.form.get('model', 'gpt-4-mini')
        rubric_id = request.form.get('rubric_id')
        
        if not rubric_id:
            return jsonify({"error": "No rubric selected"})
        
        if f:
            try:
                filename = secure_filename(f.filename)
                upload_dir = os.path.join('static', app.config['UPLOAD_FOLDER'])
                file_path = os.path.join(upload_dir, filename)
                
                os.makedirs(upload_dir, exist_ok=True)
                f.save(file_path)
                
                with open(file_path, 'rb') as file:
                    file_hash = hashlib.md5(file.read()).hexdigest()
                
                # Check if paper exists
                existing_paper = Paper.query.filter_by(hash=file_hash).first()
                if existing_paper:
                    return jsonify({'redirect': url_for('paper', file_hash=file_hash)})
                
                # Store PDF in a permanent location
                pdf_storage_path = os.path.join(upload_dir, f"{file_hash}.pdf")
                os.rename(file_path, pdf_storage_path)
                
                # Extract text
                with open(pdf_storage_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text()
                
                # Remove personal identifiers
                full_text = remove_personal_identifiers(full_text)
                
                # Create new paper
                new_paper = Paper(
                    hash=file_hash,
                    filename=filename,
                    full_text=full_text,
                    model=model,
                    pdf_path=pdf_storage_path
                )
                db.session.add(new_paper)
                
                # Create empty evaluations for each criterion
                criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).all()
                for criterion in criteria:
                    new_evaluation = Evaluation(
                        paper_id=new_paper.id,
                        criteria_id=criterion.id,
                        evaluation_text=""  # Start with empty evaluation
                    )
                    db.session.add(new_evaluation)
                
                db.session.commit()
                return jsonify({'redirect': url_for('paper', file_hash=file_hash)})
            
            except Exception as e:
                db.session.rollback()
                print(f"Error processing file: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    # GET request - load rubrics for dropdown
    rubrics = Rubric.query.order_by(Rubric.created_at.desc()).all()
    return render_template('index.html', 
                         rubrics=[{'id': r.id, 'name': r.name} for r in rubrics])

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        action = request.form['action']
        
        try:
            if action == 'add':
                username = request.form['username']
                password = request.form['password']
                is_admin = 'is_admin' in request.form
                
                new_user = User(
                    username=username,
                    password=generate_password_hash(password),
                    is_admin=is_admin
                )
                db.session.add(new_user)
                db.session.commit()
                flash('User added successfully')
                
            elif action == 'delete':
                user_id = request.form['user_id']
                user = User.query.get_or_404(user_id)
                db.session.delete(user)
                db.session.commit()
                flash('User deleted successfully')
                
            elif action == 'delete_all_articles':
                # Delete all PDF files
                papers = Paper.query.all()
                for paper in papers:
                    if os.path.exists(paper.pdf_path):
                        os.remove(paper.pdf_path)
                
                # Delete all related records first
                Chat.query.delete()
                Evaluation.query.delete()
                SavedFeedback.query.delete()
                Paper.query.delete()
                db.session.commit()
                flash('All articles and chats have been deleted.')
                
            elif action == 'delete_specific_article':
                paper_hash = request.form['paper_hash']
                paper = Paper.query.filter_by(hash=paper_hash).first_or_404()
                
                # Delete PDF file
                if os.path.exists(paper.pdf_path):
                    os.remove(paper.pdf_path)
                
                # Delete related records first
                Chat.query.filter_by(paper_id=paper.id).delete()
                Evaluation.query.filter_by(paper_id=paper.id).delete()
                SavedFeedback.query.filter_by(paper_id=paper.id).delete()
                
                # Delete paper
                db.session.delete(paper)
                db.session.commit()
                flash('Article and associated data have been deleted.')
                
            elif action == 'save_descriptors':
                # Delete existing descriptors
                GradeDescriptors.query.delete()
                
                # Add new descriptors
                ranges = [(90,100), (80,89), (70,79), (60,69), (50,59), 
                         (40,49), (30,39), (20,29), (10,19), (0,9)]
                
                for start, end in ranges:
                    descriptor_text = request.form.get(f'descriptor_{start}_{end}')
                    if descriptor_text:
                        descriptor = GradeDescriptors(
                            range_start=start,
                            range_end=end,
                            descriptor_text=descriptor_text
                        )
                        db.session.add(descriptor)
                
                db.session.commit()
                flash('Grade descriptors saved successfully')
                
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}')
    
    # Get existing descriptors for display
    descriptors = {
        (d.range_start, d.range_end): d.descriptor_text
        for d in GradeDescriptors.query.all()
    }
    
    users = User.query.all()
    return render_template('admin.html', users=users, descriptors=descriptors)

# Update the login manager loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize the database
def init_db():
    with app.app_context():
        db.create_all()
        create_admin_user() 

@app.route('/save_feedback', methods=['POST'])
@login_required
def save_feedback():
    try:
        data = request.json
        evaluation_id = data['evaluation_id']
        feedback_text = data['feedback_text']
        
        # Get and update the evaluation
        evaluation = Evaluation.query.get_or_404(evaluation_id)
        evaluation.evaluation_text = feedback_text
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_consolidated_feedback/<file_hash>', methods=['POST'])
@login_required
def save_consolidated_feedback(file_hash):
    try:
        data = request.json
        consolidated_feedback = data['consolidated_feedback']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        
        if saved_feedback:
            saved_feedback.consolidated_feedback = consolidated_feedback
            saved_feedback.updated_at = datetime.now()
        else:
            saved_feedback = SavedFeedback(
                paper_id=paper.id,
                consolidated_feedback=consolidated_feedback
            )
            db.session.add(saved_feedback)
        
        db.session.commit()
        return jsonify({"success": True})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/polish_feedback/<file_hash>', methods=['POST'])
@login_required
def polish_feedback(file_hash):
    try:
        data = request.json
        model = data['model']
        
        # Get the saved feedback
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        
        if not saved_feedback or not saved_feedback.consolidated_feedback:
            return jsonify({"error": "No feedback found to polish"}), 400

        # Create prompt using StructuredPrompt class
        prompt = StructuredPrompt()
        
        # Add task description
        prompt.add_section("task_description", 
            "Polish and enhance the following feedback statement by converting it from "
            "bullet points into prose.")
        
        # Add the feedback content
        prompt.add_section("feedback_to_polish", saved_feedback.consolidated_feedback)
        
        # Add detailed instructions with subsections
        prompt.add_section("detailed_instructions", "", subsections={
            "goal": ("Make the feedback more cohesive, professional, and well-structured while "
                    "preserving the language and tone of the core feedback."),
            "requirements": """
                1. Maintain, word-for-word, the core feedback provided by the grader. This section is indicated by the <core_feedback> and </core_feedback> tags.
                2. Selectively add feedback from the additional feedback section, indicated by the <additional_feedback> and </additional_feedback> tags.
                3. In consolidating these sections, adjust the additional feedback to match the tone, style, and presentation of the core feedback.
                4. Do not repeat points already covered in the core feedback.""",
            "formatting": "Format the response in Markdown with appropriate headers and styling."
        })

        final_prompt = prompt.build()
        system_msg = ("You are an experienced academic writing specialist who excels at "
                     "polishing feedback while maintaining its substance. Format all "
                     "responses in Markdown.")

        api_logger.info("\n=== POLISHING FEEDBACK ===")
        api_logger.info(f"Model: {model}")
        api_logger.debug(f"Prompt:\n{final_prompt}")
        api_logger.debug(f"System message: {system_msg}")

        if model.startswith('claude'):
            response = client_anthropic.messages.create(
                model=model,
                system=system_msg,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=1000
            )
            polished_feedback = response.content[0].text.strip()
            api_logger.debug(f"Response: {polished_feedback}")
        else:
            response = client_openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": final_prompt}
                ]
            )
            polished_feedback = response.choices[0].message.content.strip()
            api_logger.debug(f"Response: {polished_feedback}")

        return jsonify({"polished_feedback": polished_feedback})
    
    except Exception as e:
        print(f"Error polishing feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/moderate_feedback/<file_hash>', methods=['POST'])
@login_required
def moderate_feedback(file_hash):
    try:
        data = request.json
        core_feedback = data['core_feedback']
        model = data['model']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the saved feedback to access the numerical mark
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        proposed_mark = saved_feedback.mark if saved_feedback else None
        
        # Get all criteria evaluations for this paper
        evaluations = (Evaluation.query
                      .join(RubricCriteria)
                      .filter(Evaluation.paper_id == paper.id)
                      .filter(RubricCriteria.section_name != '_summary')
                      .all())
        
        moderated_feedback = {}
        
        for evaluation in evaluations:
            criteria = RubricCriteria.query.get(evaluation.criteria_id)
            
            # Create prompt using StructuredPrompt class
            prompt = StructuredPrompt()
            
            # Add context sections
            prompt.add_section("task_description", 
                "Your task is to evaluate the scope and fairness of the existing feedback of this essay, identifying strengths and weaknesses "
                "that weren't spotted by the original grader. Below you will find the essay text, the core feedback provided by the grader, and the evaluation criteria you should use to evaluate the essay.")
            
            prompt.add_section("essay_text", paper.full_text)
            
            prompt.add_section("core_feedback", core_feedback)
            
            # Add the proposed mark section
            if proposed_mark is not None:
                prompt.add_section("proposed_mark", 
                    f"The grader's proposed mark for this essay is: {proposed_mark}%")
            
            # Add criteria-specific instructions
            prompt.add_section("criteria_focus", "", subsections={
                "section": criteria.section_name,
                "criteria": criteria.criteria_text,
                "requirements": """
                    1. Focus on finding both strengths and weaknesses NOT mentioned in the core feedback, and assessing whether the existing feedback is a fair and accurate assessment of the essay.
                    2. Cite specific examples from the text where possible.
                    3. Avoid repeating points already covered.
                    4. Pay special attention to aspects specific to this marking criterion given above.
                    5. If the core feedback adequately covers this criterion, acknowledge this.
                    6. Finally, provide a brief overall evaluation of the essay, including whether the proposed mark seems appropriate according to the evaluation criteria supplied.
                    """
            })
            
            # Add output format instructions
            prompt.add_section("format_requirements", """
                Format your response in Markdown:
                1. Start with a brief acknowledgment of what the core feedback covered well
                2. Use bullet points for new issues identified
                3. Use bold for key terms or concepts
                4. Include specific examples from the text where possible
                """)

            final_prompt = prompt.build()
            
            # Log the prompt for debugging
            api_logger.info(f"Moderation prompt for {criteria.section_name}:")
            api_logger.info(final_prompt)

            try:
                moderated_text = llm_service.generate_response(
                    model=model,
                    messages=[{"role": "user", "content": final_prompt}],
                    system_msg="You are an expert academic writing moderator..."
                )
                
                # Store the moderated feedback
                moderated_feedback[evaluation.id] = moderated_text
                evaluation.evaluation_text = moderated_text
                db.session.add(evaluation)
                
            except Exception as e:
                api_logger.error(f"Error moderating feedback for evaluation {evaluation.id}: {str(e)}")
                moderated_feedback[evaluation.id] = f"Error: {str(e)}"
        
        db.session.commit()
        
        return jsonify({"moderated_feedback": moderated_feedback})
        
    except Exception as e:
        db.session.rollback()
        api_logger.error(f"Error in moderate_feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_feedback/<file_hash>')
@login_required
def export_feedback(file_hash):
    try:
        # Get current paper and its rubric ID
        current_paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the rubric ID through evaluations
        rubric_id = (db.session.query(RubricCriteria.rubric_id)
                    .join(Evaluation, Evaluation.criteria_id == RubricCriteria.id)
                    .filter(Evaluation.paper_id == current_paper.id)
                    .first())
        
        if not rubric_id:
            return jsonify({"error": "No rubric found"}), 404
            
        # Get all papers with the same rubric
        papers = (Paper.query
                 .join(Evaluation)
                 .join(RubricCriteria)
                 .filter(RubricCriteria.rubric_id == rubric_id[0])
                 .distinct()
                 .all())
        
        # Create CSV in memory
        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(['Filename', 'Mark', 'Consolidated Feedback'])
        
        for paper in papers:
            # Get saved feedback for this paper
            feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
            mark = feedback.mark if feedback else None
            consolidated = feedback.consolidated_feedback if feedback else ''
            
            # Clean consolidated feedback text (remove markdown and newlines)
            if consolidated:
                # Remove markdown headers
                consolidated = re.sub(r'#{1,6}\s+', '', consolidated)
                # Replace newlines with spaces
                consolidated = consolidated.replace('\n', ' ')
                # Remove multiple spaces
                consolidated = re.sub(r'\s+', ' ', consolidated)
            
            writer.writerow([paper.filename, mark, consolidated])
        
        # Get rubric name for filename
        rubric = Rubric.query.get(rubric_id[0])
        safe_rubric_name = re.sub(r'[^\w\s-]', '', rubric.name)
        filename = f"{safe_rubric_name}_feedback.csv"
        
        # Convert to bytes for sending
        output = si.getvalue().encode('utf-8-sig')  # Use UTF-8 with BOM for Excel compatibility
        si.close()
        
        # Create BytesIO object
        mem = BytesIO()
        mem.write(output)
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error exporting feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_average_time/<file_hash>')
@login_required
def get_average_time(file_hash):
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    # Get the rubric ID for the current paper
    current_rubric_id = (db.session.query(RubricCriteria.rubric_id)
                        .join(Evaluation, Evaluation.criteria_id == RubricCriteria.id)
                        .filter(Evaluation.paper_id == paper.id)
                        .first())
    
    if not current_rubric_id:
        return jsonify({"average_time": None})
    
    # Calculate average grading time using first save time
    avg_time = (db.session.query(func.avg(
        func.strftime('%s', func.min(SavedFeedback.updated_at)) - 
        func.strftime('%s', Paper.created_at)
    ))
    .select_from(Paper)
    .join(SavedFeedback)
    .join(Evaluation, Evaluation.paper_id == Paper.id)
    .join(RubricCriteria, RubricCriteria.id == Evaluation.criteria_id)
    .filter(
        RubricCriteria.rubric_id == current_rubric_id[0],
        SavedFeedback.updated_at.isnot(None)
    )
    .group_by(Paper.id)  # Group by paper to get first save time per paper
    .scalar())
    
    if avg_time:
        avg_seconds = int(avg_time)
        average_time = f"{avg_seconds // 3600:02d}:{(avg_seconds % 3600) // 60:02d}:{avg_seconds % 60:02d}"
    else:
        average_time = None
        
    return jsonify({"average_time": average_time})

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    print(f"Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Expected database location: {os.path.join(os.getcwd(), 'users.db')}")
    init_db()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=80)
