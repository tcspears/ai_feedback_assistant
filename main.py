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
import docx  # For handling DOCX files
import subprocess  # For running LibreOffice conversion
import mimetypes  # For detecting file types
from services.llm_service import LLMService
from models.database import (
    db, User, Paper, Rubric, RubricCriteria, Evaluation, 
    Chat, GradeDescriptors, SavedFeedback, FeedbackMacro, AppliedMacro, ModerationSession, CriterionFeedback
)
from utils.prompt_builder import StructuredPrompt
from utils.prompt_loader import PromptLoader
from sqlalchemy import or_

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

# Initialize prompt loader
prompt_loader = PromptLoader()

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
    
    # Get evaluations and criterion feedback
    evaluations = (Evaluation.query
                  .filter_by(paper_id=paper.id)
                  .order_by(Evaluation.criteria_id.nullsfirst())
                  .all())
    
    # Get saved feedback and criterion feedback
    saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
    criterion_feedback = {}
    if saved_feedback:
        for cf in saved_feedback.criterion_feedback:
            criterion_feedback[cf.criteria_id] = cf.feedback_text
    
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
                    'criteria_id': eval.criteria_id,
                    'section_name': criteria.section_name,
                    'evaluation_text': eval.evaluation_text,
                    'core_feedback': criterion_feedback.get(eval.criteria_id, '')
                })
    
    # Get chat history
    chats = [(chat.user_message, chat.ai_response) for chat in 
             Chat.query.filter_by(paper_id=paper.id).order_by(Chat.created_at).all()]
    
    # Get the latest moderation session
    moderation_session = (ModerationSession.query
                        .filter_by(paper_id=paper.id)
                        .order_by(ModerationSession.created_at.desc())
                        .first())
    
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
                         saved_consolidated_feedback=saved_feedback.consolidated_feedback if saved_feedback else None,
                         pdf_path=paper.pdf_path.replace('static/', ''),
                         upload_time=upload_time.isoformat(),
                         average_time=average_time,
                         moderation_session=moderation_session)

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
    ]) if descriptors else "No grade descriptors available."
    
    # Get both prompt and system message
    prompt, system_msg = prompt_loader.create_prompt(
        'chat_prompt',
        essay_content=paper.full_text,
        descriptors_text=descriptors_text
    )
    
    initial_context = prompt.build()
    
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
        
    except ValueError as e:
        # Handle model-related errors
        db.session.rollback()
        error_msg = str(e)
        api_logger.error(f"Model error in chat: {error_msg}")
        return jsonify({"error": f"Model error: {error_msg}"}), 400
        
    except Exception as e:
        db.session.rollback()
        api_logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

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
        criterion_feedback = data.get('criterion_feedback', {})
        model = data.get('model', 'gpt-4')
        file_hash = data.get('file_hash')
        align_to_mark = data.get('align_to_mark', False)
        mark = data.get('mark')
        applied_macros = data.get('applied_macros', [])
        
        if not file_hash:
            return jsonify({'error': 'No file hash provided'})
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the rubric and criteria
        evaluations = (Evaluation.query
                      .filter_by(paper_id=paper.id)
                      .order_by(Evaluation.criteria_id.nullsfirst())
                      .all())
        
        # Build the prompt for consolidated feedback
        prompt_parts = []
        
        # Add criterion-specific feedback
        for eval in evaluations:
            if eval.criteria_id and eval.criteria_id in criterion_feedback:
                criteria = RubricCriteria.query.get(eval.criteria_id)
                if criteria:
                    prompt_parts.append(f"Feedback for {criteria.section_name}:\n{criterion_feedback[eval.criteria_id]}\n")
        
        # Add mark if provided
        if mark is not None:
            prompt_parts.append(f"Overall mark: {mark}")
        
        # Add applied macros if any
        if applied_macros:
            macros = FeedbackMacro.query.filter(FeedbackMacro.id.in_(applied_macros)).all()
            if macros:
                prompt_parts.append("\nApplied feedback macros:")
                for macro in macros:
                    prompt_parts.append(f"- {macro.name}: {macro.text}")
        
        # Build the final prompt
        feedback_text = "\n".join(prompt_parts)
        
        # Use prompt loader to create the prompt and get system message
        prompt, system_msg = prompt_loader.create_prompt(
            'polish_feedback_prompt',
            feedback_to_polish=feedback_text
        )
        
        # Generate consolidated feedback using the selected model
        consolidated_feedback = llm_service.generate_response(
            model=model,
            messages=[{"role": "user", "content": prompt.build()}],
            system_msg=system_msg
        )
        
        return jsonify({
            'success': True,
            'consolidated_feedback': consolidated_feedback
        })
    except Exception as e:
        print(f"Error generating consolidated feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

# Function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to convert DOCX to PDF using LibreOffice
def convert_docx_to_pdf(docx_path, output_dir):
    try:
        libreoffice_path = app.config.get('LIBREOFFICE_PATH', '/usr/bin/soffice')
        cmd = [libreoffice_path, '--headless', '--convert-to', 'pdf', '--outdir', output_dir, docx_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"LibreOffice conversion failed: {stderr.decode()}")
        
        # Return the path to the generated PDF
        pdf_filename = os.path.splitext(os.path.basename(docx_path))[0] + '.pdf'
        return os.path.join(output_dir, pdf_filename)
    except Exception as e:
        raise Exception(f"Error converting DOCX to PDF: {str(e)}")

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
                
                # Determine file type and process accordingly
                file_ext = os.path.splitext(filename)[1].lower()
                pdf_storage_path = os.path.join(upload_dir, f"{file_hash}.pdf")
                full_text = ""
                
                if file_ext == '.pdf':
                    # Just move the PDF to its permanent location
                    os.rename(file_path, pdf_storage_path)
                    
                    # Extract text from PDF
                    with open(pdf_storage_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            full_text += page.extract_text()
                
                elif file_ext == '.docx':
                    # Extract text from DOCX
                    full_text = extract_text_from_docx(file_path)
                    
                    # Convert DOCX to PDF
                    temp_pdf_path = convert_docx_to_pdf(file_path, upload_dir)
                    
                    # Move the converted PDF to its permanent location
                    os.rename(temp_pdf_path, pdf_storage_path)
                    
                    # Remove the original DOCX file
                    os.remove(file_path)
                
                else:
                    return jsonify({"error": "Unsupported file format. Please upload a PDF or DOCX file."}), 400
                
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
                # Clean up any temporary files
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
                AppliedMacro.query.delete()
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
                AppliedMacro.query.filter_by(paper_id=paper.id).delete()
                
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

@app.route('/moderate_feedback/<file_hash>', methods=['POST'])
@login_required
def moderate_feedback(file_hash):
    data = request.json
    core_feedback = data['core_feedback']
    model = data['model']
    applied_macro_ids = data.get('applied_macros', [])
    
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
    
    # Get the text of all applied macros
    applied_macro_texts = []
    if applied_macro_ids:
        applied_macros = FeedbackMacro.query.filter(FeedbackMacro.id.in_(applied_macro_ids)).all()
        applied_macro_texts = [macro.text for macro in applied_macros]
    
    # Combine core feedback with macro texts
    combined_feedback = core_feedback
    if applied_macro_texts:
        combined_feedback += "<feedback_macros>\n"
        for text in applied_macro_texts:
            combined_feedback += f"\n{text}\n"
        combined_feedback += "</feedback_macros>\n"
    
    moderated_feedback = {}
    
    for evaluation in evaluations:
        criteria = RubricCriteria.query.get(evaluation.criteria_id)
        
        # Use prompt loader to create the prompt and get system message
        prompt, system_msg = prompt_loader.create_prompt(
            'moderate_feedback_prompt',
            essay_text=paper.full_text,
            core_feedback=combined_feedback,
            mark=str(proposed_mark) if proposed_mark is not None else "not provided",
            section=criteria.section_name,
            criteria=criteria.criteria_text
        )

        try:
            moderated_text = llm_service.generate_response(
                model=model,
                messages=[{"role": "user", "content": prompt.build()}],
                system_msg=system_msg
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

@app.route('/list_papers')
@login_required
def list_papers():
    # Query to get all papers with their rubric names and marks
    papers = (db.session.query(
        Paper,
        Rubric.name.label('rubric_name'),
        SavedFeedback.mark
    )
    .outerjoin(Evaluation, Paper.id == Evaluation.paper_id)
    .outerjoin(RubricCriteria, Evaluation.criteria_id == RubricCriteria.id)
    .outerjoin(Rubric, RubricCriteria.rubric_id == Rubric.id)
    .outerjoin(SavedFeedback, Paper.id == SavedFeedback.paper_id)
    .group_by(Paper.id)
    .all())
    
    # Format the results for the template
    formatted_papers = [{
        'hash': paper.Paper.hash,
        'filename': paper.Paper.filename,
        'rubric_name': paper.rubric_name or 'No rubric',
        'mark': paper.mark
    } for paper in papers]
    
    return render_template('list_papers.html', papers=formatted_papers)

@app.route('/delete_paper/<file_hash>', methods=['POST'])
@login_required
def delete_paper(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Delete the PDF file
        if os.path.exists(paper.pdf_path):
            os.remove(paper.pdf_path)
        
        # Delete related records
        Chat.query.filter_by(paper_id=paper.id).delete()
        Evaluation.query.filter_by(paper_id=paper.id).delete()
        SavedFeedback.query.filter_by(paper_id=paper.id).delete()
        AppliedMacro.query.filter_by(paper_id=paper.id).delete()
        
        # Delete the paper record
        db.session.delete(paper)
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_macros/<int:rubric_id>')
@login_required
def get_macros(rubric_id):
    try:
        # Get all criteria for this rubric
        criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).order_by(RubricCriteria.id).all()
        
        # Get all macros for this rubric
        macros = FeedbackMacro.query.filter_by(rubric_id=rubric_id).order_by(FeedbackMacro.category, FeedbackMacro.name).all()
        
        return jsonify({
            "success": True,
            "criteria": [
                {
                    "id": c.id,
                    "section_name": c.section_name,
                    "criteria_text": c.criteria_text
                }
                for c in criteria
            ],
            "macros": [
                {
                    "id": macro.id,
                    "name": macro.name,
                    "category": macro.category,
                    "text": macro.text,
                    "criteria_id": macro.criteria_id
                }
                for macro in macros
            ]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_macro/<int:macro_id>')
@login_required
def get_macro(macro_id):
    try:
        macro = FeedbackMacro.query.get_or_404(macro_id)
        
        return jsonify({
            "success": True,
            "macro": {
                "id": macro.id,
                "name": macro.name,
                "category": macro.category,
                "text": macro.text,
                "rubric_id": macro.rubric_id
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_macro', methods=['POST'])
@login_required
def save_macro():
    try:
        data = request.json
        rubric_id = data['rubric_id']
        name = data['name']
        category = data['category']
        text = data['text']
        
        # Create new macro
        new_macro = FeedbackMacro(
            rubric_id=rubric_id,
            name=name,
            category=category,
            text=text
        )
        db.session.add(new_macro)
        db.session.commit()
        
        return jsonify({"success": True, "macro_id": new_macro.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/update_macro/<int:macro_id>', methods=['POST'])
@login_required
def update_macro(macro_id):
    try:
        data = request.json
        macro = FeedbackMacro.query.get_or_404(macro_id)
        
        macro.name = data['name']
        macro.category = data['category']
        macro.text = data['text']
        
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete_macro/<int:macro_id>', methods=['POST'])
@login_required
def delete_macro(macro_id):
    try:
        macro = FeedbackMacro.query.get_or_404(macro_id)
        
        # Delete any applied instances of this macro
        AppliedMacro.query.filter_by(macro_id=macro_id).delete()
        
        # Delete the macro itself
        db.session.delete(macro)
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/copy_macros', methods=['POST'])
@login_required
def copy_macros():
    try:
        data = request.json
        source_rubric_id = data['source_rubric_id']
        destination_rubric_id = data['destination_rubric_id']
        
        # Get all macros from the source rubric
        source_macros = FeedbackMacro.query.filter_by(rubric_id=source_rubric_id).all()
        
        # Copy each macro to the destination rubric
        count = 0
        for macro in source_macros:
            new_macro = FeedbackMacro(
                rubric_id=destination_rubric_id,
                name=macro.name,
                category=macro.category,
                text=macro.text
            )
            db.session.add(new_macro)
            count += 1
        
        db.session.commit()
        
        return jsonify({"success": True, "count": count})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_paper_macros/<file_hash>')
@login_required
def get_paper_macros(file_hash):
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    # Get the rubric ID for the current paper through its evaluations
    current_rubric_id = (db.session.query(RubricCriteria.rubric_id)
                        .join(Evaluation, Evaluation.criteria_id == RubricCriteria.id)
                        .filter(Evaluation.paper_id == paper.id)
                        .first())
    
    if not current_rubric_id:
        return jsonify({
            'success': False,
            'error': 'No rubric found for this paper'
        })
    
    # Get all macros for this rubric (both global and criterion-specific)
    macros = (FeedbackMacro.query
              .filter(
                  or_(
                      FeedbackMacro.rubric_id == current_rubric_id[0],
                      FeedbackMacro.criteria_id.in_(
                          db.session.query(RubricCriteria.id)
                          .filter(RubricCriteria.rubric_id == current_rubric_id[0])
                      )
                  )
              )
              .all())
    
    # Get applied macros for this paper
    applied_macros = (AppliedMacro.query
                     .filter_by(paper_id=paper.id)
                     .all())
    applied_macro_ids = {am.macro_id for am in applied_macros}
    
    # Format macros for response
    formatted_macros = []
    for macro in macros:
        formatted_macro = {
            'id': macro.id,
            'name': macro.name,
            'category': macro.category,
            'text': macro.text,
            'applied': macro.id in applied_macro_ids,
            'criteria_id': macro.criteria_id
        }
        formatted_macros.append(formatted_macro)
    
    return jsonify({
        'success': True,
        'macros': formatted_macros
    })

@app.route('/save_applied_macros/<file_hash>', methods=['POST'])
@login_required
def save_applied_macros(file_hash):
    try:
        data = request.json
        applied_macro_ids = data['applied_macros']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Delete all existing applied macros for this paper
        AppliedMacro.query.filter_by(paper_id=paper.id).delete()
        
        # Add new applied macros
        for macro_id in applied_macro_ids:
            applied_macro = AppliedMacro(
                paper_id=paper.id,
                macro_id=macro_id
            )
            db.session.add(applied_macro)
        
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_macro_from_paper/<file_hash>', methods=['POST'])
@login_required
def save_macro_from_paper(file_hash):
    try:
        data = request.json
        name = data['name']
        category = data['category']
        text = data['text']
        criteria_id = data.get('criteria_id')  # Optional, for criterion-specific macros
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the rubric ID for this paper
        rubric_id = (db.session.query(RubricCriteria.rubric_id)
                    .join(Evaluation, Evaluation.criteria_id == RubricCriteria.id)
                    .filter(Evaluation.paper_id == paper.id)
                    .first())
        
        if not rubric_id:
            return jsonify({"success": False, "error": "No rubric found for this paper"})
        
        # Create new macro
        new_macro = FeedbackMacro(
            rubric_id=rubric_id[0],
            name=name,
            category=category,
            text=text,
            criteria_id=criteria_id
        )
        db.session.add(new_macro)
        db.session.commit()
        
        return jsonify({"success": True, "macro_id": new_macro.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/start_moderation/<file_hash>', methods=['POST'])
@login_required
def start_moderation(file_hash):
    try:
        data = request.json
        model = data.get('model', 'gpt-4')
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the current feedback
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback or not saved_feedback.consolidated_feedback:
            return jsonify({"success": False, "error": "No feedback found to moderate"})
        
        # Create a new moderation session
        moderation_session = ModerationSession(
            paper_id=paper.id,
            original_feedback=saved_feedback.consolidated_feedback,
            status='in_progress'
        )
        db.session.add(moderation_session)
        db.session.commit()
        
        return jsonify({"success": True, "session_id": moderation_session.id})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_moderation_feedback/<file_hash>')
@login_required
def get_moderation_feedback(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the latest moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
        
        if not moderation_session:
            return jsonify({"success": False, "error": "No moderation session found"})
        
        return jsonify({
            "success": True,
            "original_feedback": moderation_session.original_feedback,
            "moderated_feedback": moderation_session.moderated_feedback,
            "status": moderation_session.status
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/complete_moderation/<file_hash>', methods=['POST'])
@login_required
def complete_moderation(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the latest moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
        
        if not moderation_session or moderation_session.status != 'in_progress':
            return jsonify({"success": False, "error": "No active moderation session found"})
        
        # Update the saved feedback with the moderated version
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if saved_feedback:
            saved_feedback.consolidated_feedback = moderation_session.moderated_feedback
            saved_feedback.updated_at = datetime.now()
        
        # Update moderation session status
        moderation_session.status = 'completed'
        moderation_session.completed_at = datetime.now()
        
        db.session.commit()
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/reject_moderation/<file_hash>', methods=['POST'])
@login_required
def reject_moderation(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the latest moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
        
        if not moderation_session or moderation_session.status != 'in_progress':
            return jsonify({"success": False, "error": "No active moderation session found"})
        
        # Update moderation session status
        moderation_session.status = 'rejected'
        moderation_session.completed_at = datetime.now()
        
        db.session.commit()
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_criterion_feedback/<file_hash>', methods=['POST'])
@login_required
def save_criterion_feedback(file_hash):
    try:
        data = request.json
        criteria_id = data['criteria_id']
        feedback_text = data['feedback_text']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get or create SavedFeedback for this paper
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback:
            saved_feedback = SavedFeedback(paper_id=paper.id)
            db.session.add(saved_feedback)
        
        # Get or create CriterionFeedback for this criteria
        criterion_feedback = CriterionFeedback.query.filter_by(
            saved_feedback_id=saved_feedback.id,
            criteria_id=criteria_id
        ).first()
        
        if not criterion_feedback:
            criterion_feedback = CriterionFeedback(
                saved_feedback_id=saved_feedback.id,
                criteria_id=criteria_id,
                feedback_text=feedback_text
            )
            db.session.add(criterion_feedback)
        else:
            criterion_feedback.feedback_text = feedback_text
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Feedback saved successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/toggle_macro/<file_hash>', methods=['POST'])
@login_required
def toggle_macro(file_hash):
    try:
        data = request.json
        macro_id = data['macro_id']
        applied = data['applied']
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the existing applied macro
        applied_macro = AppliedMacro.query.filter_by(
            paper_id=paper.id,
            macro_id=macro_id
        ).first()
        
        if applied:
            # Add the macro if it doesn't exist
            if not applied_macro:
                applied_macro = AppliedMacro(
                    paper_id=paper.id,
                    macro_id=macro_id
                )
                db.session.add(applied_macro)
        else:
            # Remove the macro if it exists
            if applied_macro:
                db.session.delete(applied_macro)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Macro toggled successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/export_rubric/<int:rubric_id>')
@login_required
def export_rubric(rubric_id):
    try:
        # Get the rubric and its criteria
        rubric = Rubric.query.get_or_404(rubric_id)
        criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).all()
        
        # Get all macros associated with this rubric
        macros = FeedbackMacro.query.filter_by(rubric_id=rubric_id).all()
        
        # Create the export data structure
        export_data = {
            'name': rubric.name,
            'description': rubric.description,
            'criteria': [
                {
                    'section_name': c.section_name,
                    'criteria_text': c.criteria_text
                } for c in criteria
            ],
            'macros': [
                {
                    'name': m.name,
                    'category': m.category,
                    'text': m.text,
                    'criteria_id': m.criteria_id
                } for m in macros
            ]
        }
        
        return jsonify({
            'success': True,
            'rubric': export_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/import_rubric', methods=['POST'])
@login_required
def import_rubric():
    try:
        data = request.get_json()
        if not data or 'rubric' not in data:
            return jsonify({
                'success': False,
                'error': 'No rubric data provided'
            })
        
        rubric_data = data['rubric']
        
        # Create new rubric
        new_rubric = Rubric(
            name=rubric_data['name'],
            description=rubric_data['description']
        )
        db.session.add(new_rubric)
        db.session.flush()  # Get the new rubric ID
        
        # Create criteria
        criteria_map = {}  # Map to store old criteria_id -> new criteria_id
        for criterion_data in rubric_data['criteria']:
            new_criterion = RubricCriteria(
                rubric_id=new_rubric.id,
                section_name=criterion_data['section_name'],
                criteria_text=criterion_data['criteria_text']
            )
            db.session.add(new_criterion)
            db.session.flush()
            criteria_map[criterion_data.get('id', '')] = new_criterion.id
        
        # Create macros
        for macro_data in rubric_data['macros']:
            new_macro = FeedbackMacro(
                rubric_id=new_rubric.id,
                name=macro_data['name'],
                category=macro_data['category'],
                text=macro_data['text'],
                criteria_id=macro_data.get('criteria_id')  # This will be null for general macros
            )
            db.session.add(new_macro)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Rubric imported successfully',
            'rubric_id': new_rubric.id
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/delete_rubric/<int:rubric_id>', methods=['POST'])
@login_required
def delete_rubric(rubric_id):
    try:
        # Get the rubric
        rubric = Rubric.query.get_or_404(rubric_id)
        
        # Delete associated criteria
        RubricCriteria.query.filter_by(rubric_id=rubric_id).delete()
        
        # Delete associated macros
        FeedbackMacro.query.filter_by(rubric_id=rubric_id).delete()
        
        # Delete the rubric
        db.session.delete(rubric)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Rubric deleted successfully'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    print(f"Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Expected database location: {os.path.join(os.getcwd(), 'users.db')}")
    init_db()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=80)
