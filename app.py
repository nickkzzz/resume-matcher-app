import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import spacy
import re
import io
import json
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.job-match-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'parsed_resume' not in st.session_state:
    st.session_state.parsed_resume = {}
if 'job_matches' not in st.session_state:
    st.session_state.job_matches = []

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load NLP models"""
    try:
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        st.stop()
    
    # Load sentence transformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return nlp, sentence_model

# Sample job descriptions dataset
@st.cache_data
def load_job_dataset():
    """Load sample job descriptions dataset"""
    jobs = [
        {
            "id": 1,
            "title": "Data Scientist",
            "company": "Tech Corp",
            "description": "We are seeking a skilled Data Scientist with experience in machine learning, Python, SQL, and statistical analysis. The ideal candidate will have experience with pandas, scikit-learn, and data visualization tools like matplotlib and seaborn. Knowledge of deep learning frameworks such as TensorFlow or PyTorch is preferred.",
            "requirements": ["Python", "Machine Learning", "SQL", "Statistics", "pandas", "scikit-learn", "TensorFlow", "PyTorch"],
            "location": "San Francisco, CA",
            "salary": "$120,000 - $180,000"
        },
        {
            "id": 2,
            "title": "Software Engineer",
            "company": "StartupXYZ",
            "description": "Looking for a Software Engineer with strong programming skills in Python, JavaScript, and experience with web frameworks like React, Django, or Flask. The candidate should have experience with databases, API development, and version control systems like Git.",
            "requirements": ["Python", "JavaScript", "React", "Django", "Flask", "Git", "API Development", "Databases"],
            "location": "New York, NY",
            "salary": "$100,000 - $150,000"
        },
        {
            "id": 3,
            "title": "ML Engineer",
            "company": "AI Solutions",
            "description": "We need an ML Engineer to deploy and maintain machine learning models in production. Experience with Docker, Kubernetes, cloud platforms (AWS, GCP), and MLOps tools is required. Strong Python skills and experience with model deployment and monitoring are essential.",
            "requirements": ["Python", "Machine Learning", "Docker", "Kubernetes", "AWS", "GCP", "MLOps", "Model Deployment"],
            "location": "Remote",
            "salary": "$130,000 - $190,000"
        },
        {
            "id": 4,
            "title": "Business Analyst",
            "company": "Finance Pro",
            "description": "Business Analyst role focusing on data analysis, reporting, and business intelligence. Proficiency in SQL, Excel, and data visualization tools like Tableau or Power BI is required. Experience with statistical analysis and business process improvement is preferred.",
            "requirements": ["SQL", "Excel", "Tableau", "Power BI", "Business Intelligence", "Statistical Analysis", "Process Improvement"],
            "location": "Chicago, IL",
            "salary": "$80,000 - $110,000"
        },
        {
            "id": 5,
            "title": "DevOps Engineer",
            "company": "CloudTech",
            "description": "DevOps Engineer position requiring experience with cloud infrastructure, CI/CD pipelines, and automation tools. Strong knowledge of Docker, Kubernetes, Jenkins, and cloud platforms (AWS, Azure) is essential. Scripting skills in Python or Bash are required.",
            "requirements": ["DevOps", "Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "CI/CD", "Python", "Bash"],
            "location": "Austin, TX",
            "salary": "$110,000 - $160,000"
        },
        {
            "id": 6,
            "title": "Frontend Developer",
            "company": "WebDesign Co",
            "description": "Frontend Developer with expertise in modern JavaScript frameworks, HTML5, CSS3, and responsive design. Experience with React, Vue.js, or Angular is required. Knowledge of state management, testing frameworks, and build tools is preferred.",
            "requirements": ["JavaScript", "HTML5", "CSS3", "React", "Vue.js", "Angular", "Responsive Design", "Testing"],
            "location": "Seattle, WA",
            "salary": "$90,000 - $130,000"
        }
    ]
    return pd.DataFrame(jobs)

class ResumeParser:
    """Class to handle resume parsing and information extraction"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from resume text"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Phone extraction
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        contact_info['phone'] = phones[0] if phones else None
        
        # LinkedIn extraction
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        contact_info['linkedin'] = linkedin[0] if linkedin else None
        
        return contact_info
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common tech skills
        tech_skills = [
            'python', 'java', 'javascript', 'html', 'css', 'sql', 'r', 'c++', 'c#',
            'react', 'angular', 'vue', 'django', 'flask', 'nodejs', 'express',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git',
            'tableau', 'power bi', 'excel', 'powerpoint', 'word',
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            'hadoop', 'spark', 'kafka', 'airflow', 'linux', 'bash'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in tech_skills:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text"""
        doc = self.nlp(text)
        experiences = []
        
        # Look for common experience indicators
        experience_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|present)',
            r'(\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|present)',
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                experiences.append({
                    'duration': f"{match[0]} - {match[1]}",
                    'text': match
                })
        
        return experiences
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'mba', 'bs', 'ms', 'ma', 'ba',
            'university', 'college', 'institute', 'school', 'degree'
        ]
        
        education = []
        text_lower = text.lower()
        
        for keyword in education_keywords:
            if keyword in text_lower:
                # Extract sentences containing education keywords
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        education.append(sentence.strip())
                        break
        
        return list(set(education))
    
    def parse_resume(self, resume_text: str) -> Dict:
        """Parse resume and extract all information"""
        parsed_data = {
            'contact_info': self.extract_contact_info(resume_text),
            'skills': self.extract_skills(resume_text),
            'experience': self.extract_experience(resume_text),
            'education': self.extract_education(resume_text),
            'raw_text': resume_text
        }
        
        return parsed_data

class JobMatcher:
    """Class to handle job matching using similarity scoring"""
    
    def __init__(self, sentence_model):
        self.sentence_model = sentence_model
    
    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate cosine similarity between resume and job description"""
        # Create embeddings
        resume_embedding = self.sentence_model.encode([resume_text])
        job_embedding = self.sentence_model.encode([job_description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        return similarity
    
    def match_jobs(self, resume_data: Dict, job_df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Match resume to jobs and return top matches"""
        resume_text = resume_data['raw_text']
        matches = []
        
        for _, job in job_df.iterrows():
            # Calculate similarity
            similarity_score = self.calculate_similarity(resume_text, job['description'])
            
            # Calculate skill match percentage
            resume_skills = [skill.lower() for skill in resume_data['skills']]
            job_requirements = [req.lower() for req in job['requirements']]
            
            matched_skills = set(resume_skills) & set(job_requirements)
            skill_match_percentage = (len(matched_skills) / len(job_requirements)) * 100 if job_requirements else 0
            
            matches.append({
                'job_id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'salary': job['salary'],
                'description': job['description'],
                'requirements': job['requirements'],
                'similarity_score': similarity_score,
                'skill_match_percentage': skill_match_percentage,
                'matched_skills': list(matched_skills),
                'missing_skills': list(set(job_requirements) - set(resume_skills))
            })
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:top_n]

def main():
    """Main application function"""
    
    # Load models
    nlp, sentence_model = load_models()
    
    # Initialize classes
    resume_parser = ResumeParser(nlp)
    job_matcher = JobMatcher(sentence_model)
    
    # Load job dataset
    job_df = load_job_dataset()
    
    # App title
    st.markdown('<h1 class="main-header">🤖 AI Resume Analyzer & Job Matcher</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Application Settings")
        
        # Number of job matches to show
        top_n_jobs = st.slider("Number of job matches to show:", 1, 10, 5)
        
        # Job role filter
        available_roles = job_df['title'].unique()
        selected_roles = st.multiselect(
            "Filter by job roles:",
            available_roles,
            default=available_roles
        )
        
        # Location filter
        available_locations = job_df['location'].unique()
        selected_locations = st.multiselect(
            "Filter by locations:",
            available_locations,
            default=available_locations
        )
    
    # Filter job dataset
    filtered_jobs = job_df[
        (job_df['title'].isin(selected_roles)) & 
        (job_df['location'].isin(selected_locations))
    ]
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Resume Upload & Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF format)",
            type=['pdf'],
            help="Upload a PDF file of your resume for analysis"
        )
        
        # Text input alternative
        st.subheader("Or paste your resume text:")
        resume_text_input = st.text_area(
            "Resume Text",
            height=300,
            placeholder="Paste your resume text here..."
        )
        
        # Process resume
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                resume_text = resume_parser.extract_text_from_pdf(uploaded_file)
                st.session_state.resume_text = resume_text
        elif resume_text_input:
            st.session_state.resume_text = resume_text_input
        
        # Parse resume if text is available
        if st.session_state.resume_text:
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Parsing resume..."):
                    parsed_resume = resume_parser.parse_resume(st.session_state.resume_text)
                    st.session_state.parsed_resume = parsed_resume
                    
                    # Find job matches
                    job_matches = job_matcher.match_jobs(
                        parsed_resume, 
                        filtered_jobs, 
                        top_n_jobs
                    )
                    st.session_state.job_matches = job_matches
                    
                st.success("✅ Resume analysis completed!")
    
    with col2:
        st.header("📋 Resume Summary")
        
        if st.session_state.parsed_resume:
            parsed_data = st.session_state.parsed_resume
            
            # Contact Information
            st.subheader("📞 Contact Information")
            contact_info = parsed_data['contact_info']
            
            contact_col1, contact_col2 = st.columns(2)
            with contact_col1:
                if contact_info['email']:
                    st.write(f"**Email:** {contact_info['email']}")
                if contact_info['phone']:
                    st.write(f"**Phone:** {contact_info['phone']}")
            with contact_col2:
                if contact_info['linkedin']:
                    st.write(f"**LinkedIn:** {contact_info['linkedin']}")
            
            # Skills
            st.subheader("🛠️ Skills")
            if parsed_data['skills']:
                skills_text = ", ".join(parsed_data['skills'])
                st.write(skills_text)
                
                # Skills chart
                skills_df = pd.DataFrame({
                    'Skill': parsed_data['skills'],
                    'Count': [1] * len(parsed_data['skills'])
                })
                if len(skills_df) > 0:
                    fig = px.bar(
                        skills_df.head(10), 
                        x='Count', 
                        y='Skill',
                        orientation='h',
                        title="Top Skills"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No skills detected.")
            
            # Education
            st.subheader("🎓 Education")
            if parsed_data['education']:
                for edu in parsed_data['education'][:3]:  # Show top 3
                    st.write(f"• {edu}")
            else:
                st.write("No education information detected.")
    
    # Job Matches Section
    if st.session_state.job_matches:
        st.header("🎯 Job Matches")
        
        # Overview metrics
        matches = st.session_state.job_matches
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(matches))
        with col2:
            avg_similarity = np.mean([m['similarity_score'] for m in matches])
            st.metric("Avg Similarity", f"{avg_similarity:.2%}")
        with col3:
            avg_skill_match = np.mean([m['skill_match_percentage'] for m in matches])
            st.metric("Avg Skill Match", f"{avg_skill_match:.1f}%")
        with col4:
            best_match = max(matches, key=lambda x: x['similarity_score'])
            st.metric("Best Match", f"{best_match['similarity_score']:.2%}")
        
        # Matches visualization
        match_df = pd.DataFrame(matches)
        
        fig = px.scatter(
            match_df,
            x='similarity_score',
            y='skill_match_percentage',
            size='similarity_score',
            color='title',
            title="Job Match Analysis",
            labels={
                'similarity_score': 'Similarity Score',
                'skill_match_percentage': 'Skill Match %',
                'title': 'Job Title'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed job matches
        st.subheader("📋 Detailed Job Matches")
        
        for i, match in enumerate(matches, 1):
            with st.expander(f"#{i} {match['title']} at {match['company']} - {match['similarity_score']:.1%} match"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Location:** {match['location']}")
                    st.write(f"**Salary:** {match['salary']}")
                    st.write(f"**Description:** {match['description'][:200]}...")
                    
                    # Matched skills
                    if match['matched_skills']:
                        st.write("**✅ Matched Skills:**")
                        st.write(", ".join(match['matched_skills']))
                    
                    # Missing skills
                    if match['missing_skills']:
                        st.write("**❌ Missing Skills:**")
                        st.write(", ".join(match['missing_skills']))
                
                with col2:
                    # Similarity gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = match['similarity_score'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Similarity Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Skill match percentage
                    st.metric("Skill Match", f"{match['skill_match_percentage']:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, spaCy, and SentenceTransformers | "
        "Upload your resume to find the perfect job match!"
    )

if __name__ == "__main__":
    main()