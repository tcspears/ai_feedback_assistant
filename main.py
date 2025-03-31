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
    Chat, GradeDescriptors, SavedFeedback, FeedbackMacro, AppliedMacro, ModerationSession, CriterionFeedback, ModerationResult, AIEvaluation
)
from utils.prompt_builder import StructuredPrompt
from utils.prompt_loader import PromptLoader
from sqlalchemy import or_
import traceback

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
            criterion_feedback[cf.criteria_id] = cf
    
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
                    'criteria_text': criteria.criteria_text,
                    'evaluation_text': eval.evaluation_text,
                    'weight': criteria.weight,
                    'core_feedback': criterion_feedback.get(eval.criteria_id, '').feedback_text if criterion_feedback.get(eval.criteria_id) else ''
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
    
    return render_template('paper.html',
                         filename=paper.filename,
                         full_text=paper.full_text,
                         evaluations=formatted_evaluations,
                         chats=chats,
                         file_hash=file_hash,
                         related_papers=formatted_related,
                         rubric_name=rubric_name,
                         saved_feedback=saved_feedback,
                         saved_consolidated_feedback=saved_feedback.consolidated_feedback if saved_feedback else None,
                         pdf_path=paper.pdf_path.replace('static/', ''),
                         upload_time=upload_time.isoformat(),
                         moderation_session=moderation_session,
                         criterion_feedback=criterion_feedback)

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
                criteria_text=criterion['description'],
                weight=float(criterion['weight'])
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
                {
                    "section_name": c.section_name, 
                    "criteria_text": c.criteria_text,
                    "weight": c.weight
                } 
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
        overall_mark = data['mark']
        criterion_marks = data.get('criterion_marks', [])
        
        # Validate overall mark
        if overall_mark is None:
            return jsonify({"success": False, "error": "Mark cannot be None"})
        
        # Ensure mark is a valid number between 0 and 100
        try:
            overall_mark = float(overall_mark)
            if not (0 <= overall_mark <= 100):
                return jsonify({"success": False, "error": "Mark must be between 0 and 100"})
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid mark value"})
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        
        if saved_feedback:
            saved_feedback.mark = overall_mark
            saved_feedback.updated_at = datetime.now()
        else:
            saved_feedback = SavedFeedback(
                paper_id=paper.id,
                mark=overall_mark
            )
            db.session.add(saved_feedback)
            db.session.flush()  # Get the saved_feedback ID
        
        # Save criterion-specific marks
        for criterion_mark in criterion_marks:
            # Validate criterion mark
            criteria_id = criterion_mark.get('criteria_id')
            mark_value = criterion_mark.get('mark')
            
            if criteria_id is None or mark_value is None:
                continue  # Skip invalid entries
            
            try:
                mark_value = float(mark_value)
                if not (0 <= mark_value <= 100):
                    continue  # Skip out-of-range marks
            except (ValueError, TypeError):
                continue  # Skip non-numeric marks
            
            criterion_feedback = CriterionFeedback.query.filter_by(
                saved_feedback_id=saved_feedback.id,
                criteria_id=criteria_id
            ).first()
            
            if criterion_feedback:
                criterion_feedback.mark = mark_value
                criterion_feedback.updated_at = datetime.now()
            else:
                criterion_feedback = CriterionFeedback(
                    saved_feedback_id=saved_feedback.id,
                    criteria_id=criteria_id,
                    mark=mark_value
                )
                db.session.add(criterion_feedback)
        
        db.session.commit()
        return jsonify({"success": True})
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in save_mark: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


def get_grade_descriptor_for_mark(mark):
    """
    Get the grade descriptor that corresponds to a specific mark.
    
    Args:
        mark (float): The mark to get the descriptor for (0-100)
        
    Returns:
        str: The grade descriptor text, or None if no matching descriptor is found
    """
    if mark is None or not (0 <= mark <= 100):
        return None
        
    # Get all descriptors ordered by range_start descending
    descriptors = GradeDescriptors.query.order_by(GradeDescriptors.range_start.desc()).all()
    
    # Find the first descriptor where the mark falls within its range
    for descriptor in descriptors:
        if descriptor.range_start <= mark <= descriptor.range_end:
            return descriptor.descriptor_text
            
    return None

@app.route('/generate_consolidated_feedback', methods=['POST'])
@login_required
def generate_consolidated_feedback():
    try:
        data = request.json
        app.logger.debug(f"Received data: {data}")
        
        criterion_feedback = data.get('criterion_feedback', {})
        model = data.get('model', 'gpt-4')
        file_hash = data.get('file_hash')
        align_to_mark = data.get('align_to_mark', False)
        applied_macros = data.get('applied_macros', [])
        
        if not file_hash:
            return jsonify({'error': 'No file hash provided'})
        
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the rubric and criteria
        evaluations = (Evaluation.query
                      .filter_by(paper_id=paper.id)
                      .order_by(Evaluation.criteria_id.nullsfirst())
                      .all())
        
        # Get all criteria and their feedback
        feedback_sections = []
        for eval in evaluations:
            if eval.criteria_id:
                criteria = RubricCriteria.query.get(eval.criteria_id)
                if criteria:
                    # Get criterion-specific feedback and macros
                    section_feedback = criterion_feedback.get(str(eval.criteria_id), '')
                    
                    # Handle applied macros - ensure we're working with the correct data structure
                    section_macros = []
                    if applied_macros:
                        for macro in applied_macros:
                            # Handle both dictionary and object formats
                            if isinstance(macro, dict):
                                macro_criteria_id = macro.get('criteria_id')
                                macro_text = macro.get('text', '')
                            else:
                                macro_criteria_id = getattr(macro, 'criteria_id', None)
                                macro_text = getattr(macro, 'text', '')
                            
                            if macro_criteria_id == eval.criteria_id and macro_text:
                                section_macros.append(macro_text)
                    
                    # Format the section data
                    section_data = {
                        'name': criteria.section_name,
                        'feedback': section_feedback,
                        'macros': section_macros
                    }
                    feedback_sections.append(section_data)
        
        # Format feedback sections for the prompt
        formatted_sections = []
        for section in feedback_sections:
            section_text = f"# {section['name']}\n"
            if section['feedback']:
                section_text += f"Core feedback: {section['feedback']}\n"
            if section['macros']:
                section_text += "Additional feedback:\n"
                for macro in section['macros']:
                    section_text += f"- {macro}\n"
            formatted_sections.append(section_text)
        
        # Join all sections with newlines
        feedback_text = "\n\n".join(formatted_sections)
        
        # Get the overall mark and corresponding grade descriptor if we need to align the feedback
        overall_mark = None
        grade_descriptor = None
        if align_to_mark:
            saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
            if saved_feedback and saved_feedback.mark is not None:
                overall_mark = saved_feedback.mark
                grade_descriptor = get_grade_descriptor_for_mark(overall_mark)
        
        # Use prompt loader to create the prompt and get system message
        if align_to_mark and overall_mark is not None:
            prompt, system_msg = prompt_loader.create_prompt('align_feedback_prompt')
            prompt.add_section('mark', str(overall_mark))
            prompt.add_section('feedback', feedback_text)
            if grade_descriptor:
                prompt.add_section('grade_descriptor', grade_descriptor)
        else:
            prompt, system_msg = prompt_loader.create_prompt('polish_feedback_prompt')
            prompt.add_section('feedback_sections', feedback_text)
        
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
        app.logger.error(f"Error in generate_consolidated_feedback: {str(e)}")
        app.logger.error(f"Error traceback: {traceback.format_exc()}")
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
    return db.session.get(User, int(user_id))

# Initialize the database
def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create admin user if it doesn't exist
        create_admin_user()

@app.route('/save_feedback', methods=['POST'])
@login_required
def save_feedback():
    try:
        data = request.json
        app.logger.debug(f"Received save_feedback request with data: {data}")
        
        criteria_id = data['evaluation_id']  # Frontend still sends as evaluation_id
        feedback_text = data['feedback_text']
        app.logger.debug(f"Parsed criteria_id: {criteria_id}, feedback_text length: {len(feedback_text)}")
        
        # First, get the RubricCriteria to ensure it exists
        criteria = RubricCriteria.query.get_or_404(criteria_id)
        app.logger.debug(f"Found criteria: {criteria.section_name}")
        
        # Get the paper from the current session
        paper = Paper.query.filter_by(hash=request.args.get('file_hash')).first_or_404()
        app.logger.debug(f"Found paper with id: {paper.id}")
        
        # Get or create SavedFeedback
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback:
            app.logger.debug("Creating new SavedFeedback record")
            saved_feedback = SavedFeedback(paper_id=paper.id)
            db.session.add(saved_feedback)
            db.session.flush()  # Get the ID before committing
        else:
            app.logger.debug(f"Found existing SavedFeedback with id: {saved_feedback.id}")
        
        # Get or create CriterionFeedback
        criterion_feedback = CriterionFeedback.query.filter_by(
            saved_feedback_id=saved_feedback.id,
            criteria_id=criteria_id
        ).first()
        
        if criterion_feedback:
            app.logger.debug(f"Updating existing CriterionFeedback with id: {criterion_feedback.id}")
            criterion_feedback.feedback_text = feedback_text
            criterion_feedback.updated_at = func.now()
        else:
            app.logger.debug("Creating new CriterionFeedback record")
            criterion_feedback = CriterionFeedback(
                saved_feedback_id=saved_feedback.id,
                criteria_id=criteria_id,
                feedback_text=feedback_text
            )
            db.session.add(criterion_feedback)
        
        db.session.commit()
        app.logger.debug("Successfully committed changes to database")
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in save_feedback: {str(e)}")
        app.logger.error(f"Error details: {type(e).__name__}")
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

@app.route('/start_moderation/<file_hash>', methods=['POST'])
@login_required
def start_moderation(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Clear any existing moderation results for this paper
        session = ModerationSession.query.filter_by(paper_id=paper.id).first()
        if session:
            ModerationResult.query.filter_by(session_id=session.id).delete()
            db.session.delete(session)
        
        # Create a new moderation session
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback:
            return jsonify({'success': False, 'error': 'No saved feedback found'})
            
        new_session = ModerationSession(
            paper_id=paper.id,
            original_feedback=saved_feedback.consolidated_feedback,
            status='pending'
        )
        db.session.add(new_session)
        db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in start_moderation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

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
        
        # Get all moderation results
        results = {
            result.criteria_id: {
                'result': result.result,
                'reasoning': result.reasoning,
                'moderated_feedback': result.moderated_feedback
            }
            for result in moderation_session.results
        }
        
        return jsonify({
            "success": True,
            "original_feedback": moderation_session.original_feedback,
            "moderated_feedback": moderation_session.moderated_feedback,
            "status": moderation_session.status,
            "results": results
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
        
        if not moderation_session:
            return jsonify({'success': False, 'error': 'No moderation session found'})
        
        # Update all criterion feedback with moderated versions
        moderated_feedback = {}
        for result in moderation_session.results:
            criterion_feedback = CriterionFeedback.query.filter_by(
                criteria_id=result.criteria_id,
                saved_feedback_id=SavedFeedback.query.filter_by(paper_id=paper.id).first().id
            ).first()
            
            if criterion_feedback:
                criterion_feedback.feedback_text = result.moderated_feedback
                moderated_feedback[result.criteria_id] = result.moderated_feedback
        
        # Update session status
        moderation_session.status = 'completed'
        moderation_session.completed_at = func.now()
        
        db.session.commit()

        return jsonify({
            'success': True,
            'moderated_feedback': moderated_feedback
        })

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in complete_moderation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

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
        
        if moderation_session:
            # Clear all moderation results
            ModerationResult.query.filter_by(session_id=moderation_session.id).delete()
            
            # Update session status
            moderation_session.status = 'rejected'
            moderation_session.completed_at = func.now()
            
            db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in reject_moderation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/moderate_criterion/<file_hash>/<criteria_id>', methods=['POST'])
@login_required
def moderate_criterion(file_hash, criteria_id):
    try:
        # Get the model from request
        data = request.get_json()
        model = data.get('model', 'gpt-4o')
        app.logger.info(f"Starting moderation for criterion {criteria_id} using model {model}")

        # Get paper and criterion details
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        criterion = RubricCriteria.query.get_or_404(criteria_id)
        
        # Get saved feedback
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback:
            app.logger.error("No saved feedback found for paper")
            return jsonify({'success': False, 'error': 'No saved feedback found'})
            
        criterion_feedback = CriterionFeedback.query.filter_by(
            saved_feedback_id=saved_feedback.id,
            criteria_id=criteria_id
        ).first()
        
        if not criterion_feedback:
            app.logger.error("No criterion feedback found")
            return jsonify({'success': False, 'error': 'No criterion feedback found'})

        # Get or create moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
                            
        if not moderation_session or moderation_session.status != 'pending':
            app.logger.info("Creating new moderation session")
            moderation_session = ModerationSession(
                paper_id=paper.id,
                original_feedback=saved_feedback.consolidated_feedback,
                status='pending'
            )
            db.session.add(moderation_session)
            db.session.flush()  # Get the session ID

        # Get grade descriptors from database
        grade_descriptors = GradeDescriptors.query.order_by(GradeDescriptors.range_start.desc()).all()
        
        # Format grade descriptors as text
        grade_descriptors_text = ""
        if grade_descriptors:
            grade_descriptors_text = "Grade Descriptors:\n"
            for descriptor in grade_descriptors:
                grade_descriptors_text += f"{descriptor.range_start}-{descriptor.range_end}%: {descriptor.descriptor_text}\n"
        else:
            grade_descriptors_text = "No grade descriptors available."
        
        # Get more detailed information about the rubric this criterion belongs to
        rubric = None
        if criterion.rubric_id:
            rubric = Rubric.query.get(criterion.rubric_id)

        # Get all other criteria in the rubric for context
        related_criteria = []
        if rubric:
            related_criteria = RubricCriteria.query.filter_by(rubric_id=rubric.id).all()
            
        # Build detailed criterion info
        criterion_info = f"Criterion: {criterion.section_name}\n\n"
        criterion_info += f"Description: {criterion.criteria_text}\n\n"
        
        # Add rubric context
        if rubric:
            criterion_info += f"This criterion is part of the rubric: '{rubric.name}'\n"
            criterion_info += f"Rubric description: {rubric.description}\n\n"
            
        # Add weight information
        criterion_info += f"This criterion has a weight of {criterion.weight} in the overall assessment.\n"
            
        # Load the criterion moderation prompt
        prompt_loader = PromptLoader('prompts.yaml')
        prompt, system_msg = prompt_loader.create_prompt('criterion_moderation_prompt')

        # Fill in dynamic content
        prompt.add_section('criterion_info', criterion_info)
        prompt.add_section('feedback', criterion_feedback.feedback_text if criterion_feedback else "No feedback provided")
        prompt.add_section('mark_info', f"The proposed mark for this criterion is: {criterion_feedback.mark if criterion_feedback.mark else saved_feedback.mark}%")
        prompt.add_section('grade_descriptors', grade_descriptors_text)

        app.logger.info(f"Sending prompt to model {model}")
        # Get moderation result from LLM
        result = llm_service.generate_response(
            model=model,
            messages=[{"role": "user", "content": prompt.build()}],
            system_msg=system_msg
        )
        app.logger.info(f"Received response from model: {result[:100]}...")

        # Parse the JSON result - first clean up any leading/trailing text that might not be part of the JSON
        result_text = result.strip()
        
        # Try multiple approaches to extract JSON
        json_extracted = False
        json_content = None
        
        # Approach 1: Find JSON content between ```json and ``` markers
        json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', result_text)
        if json_match:
            app.logger.info("Found JSON content inside code blocks")
            json_content = json_match.group(1)
            json_extracted = True
        
        # Approach 2: Find content that looks like a JSON object (between curly braces)
        if not json_extracted:
            json_obj_match = re.search(r'(\{[\s\S]*\})', result_text)
            if json_obj_match:
                app.logger.info("Found JSON-like content between curly braces")
                json_content = json_obj_match.group(1)
                json_extracted = True
        
        # If we couldn't extract JSON, use the whole response
        if not json_extracted:
            app.logger.info("Using entire response as JSON")
            json_content = result_text
        
        app.logger.info(f"Attempting to parse JSON: {json_content}")
        try:
            result_json = json.loads(json_content)
            decision = result_json.get('decision', '').strip().upper()
            reasoning = result_json.get('reasoning', '').strip()
            
            app.logger.info(f"Parsed JSON successfully. Decision: {decision}")
            
            # Validate decision is either PASSES or FAILS
            if decision not in ['PASSES', 'FAILS']:
                app.logger.error(f"Invalid decision value: {decision}")
                return jsonify({'success': False, 'error': f'Invalid moderation decision: {decision}'})
                
            if not reasoning:
                app.logger.error("Missing reasoning in result")
                return jsonify({'success': False, 'error': 'Missing reasoning in moderation result'})
                
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON parse error: {str(e)}")
            app.logger.error(f"Result text: {result_text}")
            return jsonify({'success': False, 'error': f'Failed to parse moderation result as JSON: {str(e)}'})

        # Store the moderation result
        moderation_result = ModerationResult.query.filter_by(
            session_id=moderation_session.id,
            criteria_id=criteria_id
        ).first()
        
        if not moderation_result:
            app.logger.info("Creating new moderation result")
            moderation_result = ModerationResult(
                session_id=moderation_session.id,
                criteria_id=criteria_id,
                result=decision,
                reasoning=reasoning,
                moderated_feedback=criterion_feedback.feedback_text
            )
            db.session.add(moderation_result)
        else:
            app.logger.info("Updating existing moderation result")
            moderation_result.result = decision
            moderation_result.reasoning = reasoning
            moderation_result.moderated_feedback = criterion_feedback.feedback_text
        
        db.session.commit()
        app.logger.info("Moderation completed successfully")

        return jsonify({
            'success': True,
            'result': decision,
            'reasoning': reasoning
        })

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in moderate_criterion: {str(e)}")
        app.logger.error(f"Error traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/accept_criterion_changes/<file_hash>/<criteria_id>', methods=['POST'])
@login_required
def accept_criterion_changes(file_hash, criteria_id):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the latest moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
        
        if not moderation_session:
            return jsonify({'success': False, 'error': 'No moderation session found'})
            
        # Get the moderation result
        moderation_result = ModerationResult.query.filter_by(
            session_id=moderation_session.id,
            criteria_id=criteria_id
        ).first()
        
        if not moderation_result:
            return jsonify({'success': False, 'error': 'No moderation result found'})

        # Update the criterion feedback with the moderated version
        criterion_feedback = CriterionFeedback.query.filter_by(
            saved_feedback_id=SavedFeedback.query.filter_by(paper_id=paper.id).first().id,
            criteria_id=criteria_id
        ).first()
        
        if criterion_feedback:
            criterion_feedback.feedback_text = moderation_result.moderated_feedback
            
        # Delete the moderation result
        db.session.delete(moderation_result)
        db.session.commit()

        return jsonify({
            'success': True,
            'moderated_feedback': moderation_result.moderated_feedback
        })

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in accept_criterion_changes: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reject_criterion_changes/<file_hash>/<criteria_id>', methods=['POST'])
@login_required
def reject_criterion_changes(file_hash, criteria_id):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get the latest moderation session
        moderation_session = (ModerationSession.query
                            .filter_by(paper_id=paper.id)
                            .order_by(ModerationSession.created_at.desc())
                            .first())
        
        if moderation_session:
            # Delete the moderation result
            ModerationResult.query.filter_by(
                session_id=moderation_session.id,
                criteria_id=criteria_id
            ).delete()
            
            db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in reject_criterion_changes: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_ai_evaluations/<file_hash>')
@login_required
def get_ai_evaluations(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get all AI evaluations for this paper
        ai_evaluations = AIEvaluation.query.filter_by(paper_id=paper.id).all()
        
        # Format evaluations for response
        formatted_evaluations = {}
        for eval in ai_evaluations:
            formatted_evaluations[eval.criteria_id] = {
                'evaluation_text': eval.evaluation_text,
                'mark': eval.mark,
                'created_at': eval.created_at.isoformat()
            }
        
        return jsonify({
            'success': True,
            'evaluations': formatted_evaluations
        })
        
    except Exception as e:
        app.logger.error(f"Error getting AI evaluations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/generate_ai_evaluation/<file_hash>/<criteria_id>', methods=['POST'])
@login_required
def generate_ai_evaluation(file_hash, criteria_id):
    try:
        # Get the model from request
        data = request.get_json()
        model = data.get('model', 'gpt-4')
        app.logger.info(f"Starting AI evaluation for criterion {criteria_id} using model {model}")

        # Get paper and criterion details
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        criterion = RubricCriteria.query.get_or_404(criteria_id)
        
        # Get grade descriptors
        grade_descriptors = GradeDescriptors.query.order_by(GradeDescriptors.range_start.desc()).all()
        
        # Format grade descriptors as text
        grade_descriptors_text = ""
        if grade_descriptors:
            grade_descriptors_text = "Grade Descriptors:\n"
            for descriptor in grade_descriptors:
                grade_descriptors_text += f"{descriptor.range_start}-{descriptor.range_end}%: {descriptor.descriptor_text}\n"
        else:
            grade_descriptors_text = "No grade descriptors available."
        
        # Get more detailed information about the rubric this criterion belongs to
        rubric = None
        if criterion.rubric_id:
            rubric = Rubric.query.get(criterion.rubric_id)

        # Build detailed criterion info
        criterion_info = f"Criterion: {criterion.section_name}\n\n"
        criterion_info += f"Description: {criterion.criteria_text}\n\n"
        
        # Add rubric context
        if rubric:
            criterion_info += f"This criterion is part of the rubric: '{rubric.name}'\n"
            criterion_info += f"Rubric description: {rubric.description}\n\n"
            
        # Add weight information
        criterion_info += f"This criterion has a weight of {criterion.weight} in the overall assessment.\n"
            
        # Load the AI evaluation prompt
        prompt, system_msg = prompt_loader.create_prompt('ai_evaluation_prompt')

        # Fill in dynamic content
        prompt.add_section('essay_text', paper.full_text)
        prompt.add_section('criterion_info', criterion_info)
        prompt.add_section('grade_descriptors', grade_descriptors_text)

        app.logger.info(f"Sending prompt to model {model}")
        # Get evaluation from LLM
        result = llm_service.generate_response(
            model=model,
            messages=[{"role": "user", "content": prompt.build()}],
            system_msg=system_msg
        )
        app.logger.info(f"Received response from model: {result[:100]}...")

        # Try to extract JSON from the response
        result_text = result.strip()
        
        # Try multiple approaches to extract JSON
        json_extracted = False
        json_content = None
        
        # Approach 1: Find JSON content between ```json and ``` markers
        json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', result_text)
        if json_match:
            app.logger.info("Found JSON content inside code blocks")
            json_content = json_match.group(1)
            json_extracted = True
        
        # Approach 2: Find content that looks like a JSON object (between curly braces)
        if not json_extracted:
            json_obj_match = re.search(r'(\{[\s\S]*\})', result_text)
            if json_obj_match:
                app.logger.info("Found JSON-like content between curly braces")
                json_content = json_obj_match.group(1)
                json_extracted = True
        
        # If we couldn't extract JSON, use the whole response
        if not json_extracted:
            app.logger.info("Using entire response as JSON")
            json_content = result_text
        
        app.logger.info(f"Attempting to parse JSON: {json_content}")
        try:
            result_json = json.loads(json_content)
            evaluation_text = result_json.get('evaluation', '').strip()
            mark = result_json.get('mark')
            reasoning = result_json.get('reasoning', '').strip()
            
            if not evaluation_text or mark is None:
                app.logger.error("Missing required fields in AI evaluation result")
                return jsonify({'success': False, 'error': 'Invalid AI evaluation result'})
                
            # Validate mark is between 0 and 100
            try:
                mark = float(mark)
                if not (0 <= mark <= 100):
                    return jsonify({'success': False, 'error': 'Invalid mark value'})
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Invalid mark value'})
                
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON parse error: {str(e)}")
            app.logger.error(f"Result text: {result_text}")
            app.logger.error(f"Attempted to parse: {json_content}")
            return jsonify({'success': False, 'error': f'Failed to parse AI evaluation result as JSON: {str(e)}'})

        # Store the AI evaluation
        ai_evaluation = AIEvaluation.query.filter_by(
            paper_id=paper.id,
            criteria_id=criteria_id
        ).first()
        
        if not ai_evaluation:
            app.logger.info("Creating new AI evaluation")
            ai_evaluation = AIEvaluation(
                paper_id=paper.id,
                criteria_id=criteria_id,
                evaluation_text=evaluation_text,
                mark=mark
            )
            db.session.add(ai_evaluation)
        else:
            app.logger.info("Updating existing AI evaluation")
            ai_evaluation.evaluation_text = evaluation_text
            ai_evaluation.mark = mark
        
        db.session.commit()
        app.logger.info("AI evaluation completed successfully")

        return jsonify({
            'success': True,
            'evaluation_text': evaluation_text,
            'mark': mark,
            'reasoning': reasoning
        })

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in generate_ai_evaluation: {str(e)}")
        app.logger.error(f"Error traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/list_papers')
@login_required
def list_papers():
    # Get all papers ordered by filename
    papers = Paper.query.order_by(Paper.filename.collate('NOCASE')).all()
    
    # Format papers for template
    formatted_papers = []
    for paper in papers:
        # Get the rubric name if available
        rubric_name = None
        evaluation = Evaluation.query.filter_by(paper_id=paper.id).first()
        if evaluation and evaluation.criteria_id:
            criteria = RubricCriteria.query.get(evaluation.criteria_id)
            if criteria and criteria.rubric:
                rubric_name = criteria.rubric.name
        
        formatted_papers.append({
            'hash': paper.hash,
            'filename': paper.filename,
            'rubric_name': rubric_name,
            'created_at': paper.created_at
        })
    
    return render_template('list_papers.html', papers=formatted_papers)

@app.route('/export_feedback/<file_hash>')
@login_required
def export_feedback(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get saved feedback
        saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
        if not saved_feedback:
            flash('No feedback found for this paper')
            return redirect(url_for('paper', file_hash=file_hash))
        
        # Get all evaluations
        evaluations = (Evaluation.query
                      .filter_by(paper_id=paper.id)
                      .order_by(Evaluation.criteria_id.nullsfirst())
                      .all())
        
        # Get all criteria and their feedback
        criteria_data = []
        for eval in evaluations:
            if eval.criteria_id:
                criteria = RubricCriteria.query.get(eval.criteria_id)
                if criteria:
                    # Get criterion-specific mark and feedback
                    criterion_feedback = CriterionFeedback.query.filter_by(
                        saved_feedback_id=saved_feedback.id,
                        criteria_id=eval.criteria_id
                    ).first()
                    
                    criterion_mark = criterion_feedback.mark if criterion_feedback and criterion_feedback.mark else 0
                    criterion_feedback_text = criterion_feedback.feedback_text if criterion_feedback else eval.evaluation_text
                    
                    criteria_data.append({
                        'name': criteria.section_name,
                        'mark': criterion_mark,
                        'weight': criteria.weight,
                        'feedback': criterion_feedback_text or 'No feedback'
                    })
        
        # Create CSV data with a single row per essay
        output = StringIO()
        writer = csv.writer(output)
        
        # Create headers
        headers = ['Paper Name', 'Total Mark']
        
        # Add column headers for each criterion
        for i, criterion in enumerate(criteria_data):
            criterion_num = i + 1
            headers.extend([
                f'Criterion {criterion_num} Name',
                f'Criterion {criterion_num} Mark',
                f'Criterion {criterion_num} Weight',
                f'Criterion {criterion_num} Feedback'
            ])
        
        # Add consolidated feedback header
        headers.append('Consolidated Feedback')
        
        # Write headers
        writer.writerow(headers)
        
        # Create a single row with all data
        row_data = [paper.filename]
        
        # Add total mark
        row_data.append(f"{saved_feedback.mark:.1f}%" if saved_feedback.mark else 'Not assigned')
        
        # Add data for each criterion
        for criterion in criteria_data:
            row_data.extend([
                criterion['name'],
                f"{criterion['mark']:.1f}%" if criterion['mark'] else 'Not assigned',
                f"{criterion['weight']:.2f}",
                criterion['feedback']
            ])
        
        # Add consolidated feedback
        row_data.append(saved_feedback.consolidated_feedback if saved_feedback.consolidated_feedback else 'No consolidated feedback')
        
        # Write the row
        writer.writerow(row_data)
        
        # Create the response
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{paper.filename}_feedback.csv'
        )
        
    except Exception as e:
        app.logger.error(f"Error exporting feedback: {str(e)}")
        flash('Error exporting feedback')
        return redirect(url_for('paper', file_hash=file_hash))

@app.route('/get_paper_macros/<file_hash>')
@login_required
def get_paper_macros(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Get all macros for this paper's rubric
        evaluations = Evaluation.query.filter_by(paper_id=paper.id).all()
        criteria_ids = [eval.criteria_id for eval in evaluations if eval.criteria_id]
        
        # Get all macros for these criteria
        macros = (FeedbackMacro.query
                 .filter(FeedbackMacro.criteria_id.in_(criteria_ids))
                 .all())
        
        # Get applied macros for this paper
        applied_macro_ids = {
            am.macro_id for am in AppliedMacro.query.filter_by(paper_id=paper.id).all()
        }
        
        # Format macros for response
        formatted_macros = []
        for macro in macros:
            formatted_macros.append({
                'id': macro.id,
                'name': macro.name,
                'category': macro.category,
                'text': macro.text,
                'criteria_id': macro.criteria_id,
                'applied': macro.id in applied_macro_ids
            })
        
        return jsonify({
            'success': True,
            'macros': formatted_macros
        })
        
    except Exception as e:
        app.logger.error(f"Error getting paper macros: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/toggle_macro/<file_hash>/<int:macro_id>', methods=['POST'])
@login_required
def toggle_macro(file_hash, macro_id):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Check if macro is already applied
        applied_macro = AppliedMacro.query.filter_by(
            paper_id=paper.id,
            macro_id=macro_id
        ).first()
        
        if applied_macro:
            # Remove the macro
            db.session.delete(applied_macro)
        else:
            # Add the macro
            applied_macro = AppliedMacro(
                paper_id=paper.id,
                macro_id=macro_id
            )
            db.session.add(applied_macro)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'applied': not bool(applied_macro)
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error toggling macro: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export_rubric/<int:rubric_id>')
@login_required
def export_rubric(rubric_id):
    try:
        rubric = Rubric.query.get_or_404(rubric_id)
        criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).order_by(RubricCriteria.id).all()
        
        # Create a dictionary with all rubric data including weights and macros
        rubric_data = {
            "name": rubric.name,
            "description": rubric.description,
            "criteria": []
        }
        
        # Add criteria with their associated macros
        for criterion in criteria:
            criterion_data = {
                "section_name": criterion.section_name,
                "criteria_text": criterion.criteria_text,
                "weight": criterion.weight,
                "macros": []
            }
            
            # Get macros for this criterion
            macros = FeedbackMacro.query.filter_by(
                rubric_id=rubric_id,
                criteria_id=criterion.id
            ).all()
            
            # Add macros to criterion data
            for macro in macros:
                criterion_data["macros"].append({
                    "name": macro.name,
                    "category": macro.category,
                    "text": macro.text
                })
            
            rubric_data["criteria"].append(criterion_data)
        
        return jsonify({
            "success": True,
            "rubric": rubric_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/import_rubric', methods=['POST'])
@login_required
def import_rubric():
    try:
        data = request.json
        rubric_data = data['rubric']
        
        # Validate required fields
        if not rubric_data.get('name') or not rubric_data.get('description') or not rubric_data.get('criteria'):
            return jsonify({"success": False, "error": "Missing required rubric data"})
        
        # Create new rubric
        rubric = Rubric(
            name=rubric_data['name'],
            description=rubric_data['description']
        )
        db.session.add(rubric)
        db.session.flush()  # Get the rubric_id
        
        # Add criteria with weights and macros
        total_weight = 0
        criteria_list = []
        
        for criterion_data in rubric_data['criteria']:
            if not criterion_data.get('section_name') or not criterion_data.get('criteria_text'):
                continue
                
            # Get weight from data or use default
            weight = float(criterion_data.get('weight', 1.0))
            total_weight += weight
            
            criterion = RubricCriteria(
                rubric_id=rubric.id,
                section_name=criterion_data['section_name'],
                criteria_text=criterion_data['criteria_text'],
                weight=weight
            )
            criteria_list.append(criterion)
            db.session.add(criterion)
            db.session.flush()  # Get the criterion ID
            
            # Add macros for this criterion if any exist
            macros = criterion_data.get('macros', [])
            for macro_data in macros:
                if not macro_data.get('name') or not macro_data.get('text'):
                    continue
                    
                macro = FeedbackMacro(
                    rubric_id=rubric.id,
                    criteria_id=criterion.id,
                    name=macro_data['name'],
                    category=macro_data.get('category', 'general'),
                    text=macro_data['text']
                )
                db.session.add(macro)
        
        # Validate total weight is close to 1
        if abs(total_weight - 1) > 0.0001:
            return jsonify({
                "success": False, 
                "error": f"Total weight must be 1.0 (current sum: {total_weight:.4f})"
            })
        
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
        criteria_id = data['criteria_id']
        
        # Get the rubric_id from the criteria
        criteria = RubricCriteria.query.get_or_404(criteria_id)
        rubric_id = criteria.rubric_id
        
        # Create new macro
        macro = FeedbackMacro(
            name=name,
            category=category,
            text=text,
            criteria_id=criteria_id,
            rubric_id=rubric_id  # Add the rubric_id
        )
        db.session.add(macro)
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error saving macro: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/update_macro/<int:macro_id>', methods=['POST'])
@login_required
def update_macro(macro_id):
    try:
        data = request.json
        name = data['name']
        category = data['category']
        text = data['text']
        
        # Find and update the macro
        macro = FeedbackMacro.query.get_or_404(macro_id)
        
        # Only allow editing if current user is admin or if it's their macro
        # This check could be modified based on your application's needs
        
        # Update the macro
        macro.name = name
        macro.category = category
        macro.text = text
        
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating macro: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete_macro/<int:macro_id>', methods=['POST'])
@login_required
def delete_macro(macro_id):
    try:
        # Find the macro
        macro = FeedbackMacro.query.get_or_404(macro_id)
        
        # Only allow deletion if current user is admin or if it's their macro
        # This check could be modified based on your application's needs
        
        # Delete any applied instances of this macro
        AppliedMacro.query.filter_by(macro_id=macro_id).delete()
        
        # Delete the macro itself
        db.session.delete(macro)
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting macro: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete_paper/<file_hash>', methods=['POST'])
@login_required
def delete_paper(file_hash):
    try:
        paper = Paper.query.filter_by(hash=file_hash).first_or_404()
        
        # Delete PDF file if it exists
        if os.path.exists(paper.pdf_path):
            os.remove(paper.pdf_path)
        
        # Delete related records first
        Chat.query.filter_by(paper_id=paper.id).delete()
        Evaluation.query.filter_by(paper_id=paper.id).delete()
        SavedFeedback.query.filter_by(paper_id=paper.id).delete()
        AppliedMacro.query.filter_by(paper_id=paper.id).delete()
        AIEvaluation.query.filter_by(paper_id=paper.id).delete()
        
        # Delete any moderation sessions and results
        moderation_sessions = ModerationSession.query.filter_by(paper_id=paper.id).all()
        for session in moderation_sessions:
            ModerationResult.query.filter_by(session_id=session.id).delete()
            db.session.delete(session)
        
        # Finally delete the paper
        db.session.delete(paper)
        db.session.commit()
        
        return jsonify({"success": True})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting paper: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print(f"Current working directory: {os.getcwd()}")
    print(f"Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Expected database location: {os.path.join(os.getcwd(), 'users.db')}")
    init_db()
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=80)
