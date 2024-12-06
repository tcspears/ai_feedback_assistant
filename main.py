# Flask backend (app.py)
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
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
db = SQLAlchemy(app)

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

class SavedFeedback(db.Model):
    __tablename__ = 'saved_feedback'
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey('papers.id'), nullable=False)
    additional_feedback = db.Column(db.Text)
    consolidated_feedback = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, nullable=False, default=func.now())

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
    
    # Get evaluations with their criteria
    evaluations = (db.session.query(Evaluation, RubricCriteria)
                  .join(RubricCriteria, Evaluation.criteria_id == RubricCriteria.id)
                  .filter(Evaluation.paper_id == paper.id)
                  .all())
    
    # Get related papers
    related_papers = (Paper.query
                     .join(Evaluation)
                     .join(RubricCriteria)
                     .filter(Paper.hash != file_hash)
                     .distinct()
                     .order_by(Paper.created_at.desc())
                     .all())
    
    # Format evaluations for template
    md = markdown.Markdown(extensions=['extra'])
    formatted_evaluations = []
    
    # Add summary evaluation if it exists
    summary_eval = Evaluation.query.filter_by(
        paper_id=paper.id,
        criteria_id=None
    ).first()
    if summary_eval:
        formatted_evaluations.append({
            'id': 'summary',
            'section_name': '_summary',
            'evaluation_text': Markup(md.convert(summary_eval.evaluation_text))
        })
    
    # Add criteria evaluations
    for eval, criteria in evaluations:
        formatted_evaluations.append({
            'id': eval.id,
            'section_name': criteria.section_name,
            'evaluation_text': Markup(md.convert(eval.evaluation_text))
        })
    
    # Get chat history
    chats = [(chat.user_message, chat.ai_response) for chat in 
             Chat.query.filter_by(paper_id=paper.id).order_by(Chat.created_at).all()]
    
    # Get saved feedback
    saved_feedback = SavedFeedback.query.filter_by(paper_id=paper.id).first()
    
    # Format related papers for template
    formatted_related = [(p.hash, p.filename) for p in related_papers]
    
    # Get the rubric name
    rubric_name = None
    if evaluations:
        first_eval = evaluations[0]
        if first_eval[1] and first_eval[1].rubric:
            rubric_name = first_eval[1].rubric.name
    
    return render_template('paper.html',
                         filename=paper.filename,
                         full_text=paper.full_text,
                         model=paper.model,
                         evaluations=formatted_evaluations,
                         chats=chats,
                         file_hash=file_hash,
                         related_papers=formatted_related,
                         rubric_name=rubric_name,
                         saved_additional_feedback=saved_feedback.additional_feedback if saved_feedback else None,
                         saved_consolidated_feedback=saved_feedback.consolidated_feedback if saved_feedback else None,
                         pdf_path=paper.pdf_path.replace('static/', ''))  # Remove 'static/' from path if present

@app.route('/chat/<file_hash>', methods=['POST'])
@login_required
def chat(file_hash):
    data = request.json
    user_message = data['message']
    model = data['model']

    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    # Get all evaluations including summary
    evaluations = (db.session.query(
        Evaluation, RubricCriteria
    ).outerjoin(
        RubricCriteria, 
        Evaluation.criteria_id == RubricCriteria.id
    ).filter(
        Evaluation.paper_id == paper.id
    ).order_by(
        Evaluation.criteria_id
    ).all())
    
    # Format evaluations text
    evaluations_text = "\n\n".join([
        f"=== {crit.section_name if crit else 'Summary'} ===\n{eval.evaluation_text}"
        for eval, crit in evaluations
    ])

    system_message = f"""You are an AI assistant specialized in discussing essays. You have access to:
1. The full essay text
2. A comprehensive set of evaluations for different aspects of the essay
3. An overall evaluation

Here are all the evaluations:

{evaluations_text}"""

    # Generate AI response using existing client logic
    if model.startswith('claude'):
        response = client_anthropic.messages.create(
            model=model,
            system=system_message,
            messages=[{
                "role": "user",
                "content": f"""Based on the essay and its evaluations above, please answer the following question:

{user_message}

Note: When referring to specific parts of the evaluations, please mention which section you're drawing from."""
            }],
            max_tokens=1000
        )
        ai_message = response.content[0].text.strip()
    else:
        response = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"""Based on the essay and its evaluations above, please answer the following question:

{user_message}

Note: When referring to specific parts of the evaluations, please mention which section you're drawing from."""}
            ]
        )
        ai_message = response.choices[0].message.content.strip()

    # Save chat message
    new_chat = Chat(
        paper_id=paper.id,
        user_message=user_message,
        ai_response=ai_message
    )
    db.session.add(new_chat)
    db.session.commit()

    return jsonify({"response": ai_message})

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
    
    prompt = f"""Below is a section of the marking rubric for this assignment. Please evaluate the following essay according to the criteria described. Your evaluation should begin with your categorization of the essay (e.g. Fail, Satisfactory, Good, Very Good, Excellent), followed by your justification for that categorization, and then relevant feedback for the student:

Section Name: {section_name}
Criteria: {criteria_text}

Essay text:
{text}

Please provide a detailed evaluation focusing specifically on the {section_name} criterion described above
"""

    try:
        if model.startswith('claude'):
            response = client_anthropic.messages.create(
                model=model,
                system="You are an experienced essay evaluator providing detailed feedback.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            evaluation = response.content[0].text.strip()
        else:
            response = client_openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an experienced essay evaluator providing detailed feedback."},
                    {"role": "user", "content": prompt}
                ]
            )
            evaluation = response.choices[0].message.content.strip()
        
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

@app.route('/generate_consolidated_feedback', methods=['POST'])
@login_required
def generate_consolidated_feedback():
    data = request.json
    selected_feedback = data['selected_feedback']
    additional_feedback = data['additional_feedback']
    model = data['model']
    file_hash = data['file_hash']
    
    paper = Paper.query.filter_by(hash=file_hash).first_or_404()
    
    # Format feedback text
    feedback_text = "Selected Feedback Points:\n\n"
    for section, points in selected_feedback.items():
        feedback_text += f"=== {section} ===\n"
        feedback_text += "\n".join(points) + "\n\n"
    
    if additional_feedback:
        feedback_text += "=== Additional Feedback ===\n"
        feedback_text += additional_feedback + "\n"

    # Generate consolidated feedback using the selected model
    prompt = f"""Based on the following selected feedback points, generate a well-structured, 
    cohesive but concise feedback statement for the student. Use bullet points where appropriate. 
    Include both strengths and areas for improvement.

    {feedback_text}"""

    if model.startswith('claude'):
        response = client_anthropic.messages.create(
            model=model,
            system="You are an experienced essay grader providing constructive feedback.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        consolidated_feedback = response.content[0].text.strip()
    else:
        response = client_openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an experienced essay grader providing constructive feedback."},
                {"role": "user", "content": prompt}
            ]
        )
        consolidated_feedback = response.choices[0].message.content.strip()

    # Save or update feedback
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
                # Change upload path to static/uploads
                upload_dir = os.path.join('static', app.config['UPLOAD_FOLDER'])
                file_path = os.path.join(upload_dir, filename)
                
                # Ensure upload directory exists
                os.makedirs(upload_dir, exist_ok=True)
                
                f.save(file_path)
                
                with open(file_path, 'rb') as file:
                    file_hash = hashlib.md5(file.read()).hexdigest()
                
                # Check if paper exists
                existing_paper = Paper.query.filter_by(hash=file_hash).first()
                if existing_paper:
                    return jsonify({'redirect': url_for('paper', file_hash=file_hash)})
                
                # Store PDF in a permanent location within static/uploads
                pdf_storage_path = os.path.join(upload_dir, f"{file_hash}.pdf")
                os.rename(file_path, pdf_storage_path)
                
                # Extract text
                with open(pdf_storage_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    for page in reader.pages:
                        full_text += page.extract_text()
                
                # Create new paper
                new_paper = Paper(
                    hash=file_hash,
                    filename=filename,
                    full_text=full_text,
                    model=model,
                    pdf_path=pdf_storage_path
                )
                db.session.add(new_paper)
                db.session.commit()
                
                # Generate evaluations for each criterion
                criteria = RubricCriteria.query.filter_by(rubric_id=rubric_id).all()
                
                for criterion in criteria:
                    evaluation = generate_evaluation(full_text, criterion.section_name, 
                                                  criterion.criteria_text, model)
                    new_evaluation = Evaluation(
                        paper_id=new_paper.id,
                        criteria_id=criterion.id,
                        evaluation_text=evaluation
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
                
                # Delete all chats and papers
                Chat.query.delete()
                Paper.query.delete()
                db.session.commit()
                flash('All articles and chats have been deleted.')
                
            elif action == 'delete_specific_article':
                paper_hash = request.form['paper_hash']
                paper = Paper.query.filter_by(hash=paper_hash).first_or_404()
                
                # Delete PDF file
                if os.path.exists(paper.pdf_path):
                    os.remove(paper.pdf_path)
                
                # Delete paper and associated chats
                db.session.delete(paper)
                db.session.commit()
                flash('Article and associated chats have been deleted.')
                
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}')
    
    users = User.query.all()
    return render_template('admin.html', users=users)

# Update the login manager loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize the database
def init_db():
    with app.app_context():
        db.create_all()
        create_admin_user() 

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
    app.run(host='0.0.0.0', port=80)