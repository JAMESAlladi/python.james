"""
FastAPI Resume Screening Backend
Ultra-fast resume analysis with Groq AI
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.schemas import (
    JobDescription, 
    ResumeAnalysis, 
    CandidateProfile,
    AnalysisResults,
    BulkAnalysisRequest,
    BulkAnalysisResponse
)
from services.resume_parser import ResumeParser
from services.job_matcher import JobMatcher
from utils.file_utils import save_uploaded_file, cleanup_temp_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Screening API",
    description="AI-powered resume screening with ultra-fast Groq processing",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
resume_parser = ResumeParser()
job_matcher = JobMatcher()

# Store for analysis results and candidate profiles
analysis_store: Dict[str, Any] = {}
candidate_store: Dict[str, Any] = {}

# Simple data models for backward compatibility
class CandidateMatch(BaseModel):
    filename: str
    score: float
    rank: int
    strengths: List[str]
    weaknesses: List[str] 
    summary: str
    detailed_analysis: str
    skills_match: float
    experience_match: float
    education_match: float
    
    # Enhanced ranking fields
    competitive_score: float = 0.0
    detailed_metrics: Dict[str, Any] = {}
    ranking_justification: str = ""
    
    # Optional fields with defaults
    success: bool = True
    message: str = ""

async def smart_ranking_with_justification(matches: List[CandidateMatch], job_desc: JobDescription) -> List[CandidateMatch]:
    """
    Intelligent ranking system that compares resumes against each other
    and provides detailed justifications for ranking decisions
    """
    logger.info(f"Starting intelligent ranking of {len(matches)} candidates")
    
    # Phase 1: Enhanced scoring with detailed analysis
    for match in matches:
        # Calculate detailed metrics
        match.detailed_metrics = await calculate_detailed_metrics(match, job_desc)
        
        # Recalculate score with competitive factors
        match.competitive_score = await calculate_competitive_score(match, job_desc)
    
    # Phase 2: Comparative ranking (always by AI score if present, else fallback)
    def get_sort_score(x):
        # Prefer AI score if present, else fallback
        return getattr(x, 'score', None) or getattr(x, 'overall_score', None) or getattr(x, 'competitive_score', 0)
    matches.sort(key=get_sort_score, reverse=True)
    
    # Phase 3: Generate comparative justifications
    for i, match in enumerate(matches):
        match.rank = i + 1
        
        # Generate ranking justification
        match.ranking_justification = await generate_ranking_justification(
            match, matches, i, job_desc
        )
        
        # Update detailed analysis with competitive insights
        match.detailed_analysis = await enhance_analysis_with_comparison(
            match, matches, i, job_desc
        )
    
    logger.info("Intelligent ranking completed")
    return matches

async def calculate_detailed_metrics(match: CandidateMatch, job_desc: JobDescription) -> Dict[str, Any]:
    """Calculate detailed metrics beyond basic scoring"""
    
    # Extract resume content for analysis (would need to be passed from parser)
    metrics = {
        "keyword_density": 0,
        "experience_relevance": 0,
        "skill_depth": 0,
        "achievement_score": 0,
        "education_relevance": 0,
        "certification_value": 0,
        "career_progression": 0,
        "industry_match": 0
    }
    
    # This would analyze the actual resume content
    # For now, derive from existing scores
    metrics["keyword_density"] = match.skills_match
    metrics["experience_relevance"] = match.experience_match
    metrics["education_relevance"] = match.education_match
    
    return metrics

async def calculate_competitive_score(match: CandidateMatch, job_desc: JobDescription) -> float:
    """Calculate competitive score considering multiple factors"""
    
    base_score = match.score
    
    # Bonus factors
    bonuses = 0
    
    # Achievement bonus (extracted from detailed analysis)
    if "project" in match.detailed_analysis.lower():
        bonuses += 5
    if "lead" in match.detailed_analysis.lower() or "manage" in match.detailed_analysis.lower():
        bonuses += 8
    if "award" in match.detailed_analysis.lower() or "recognition" in match.detailed_analysis.lower():
        bonuses += 10
    
    # Skills depth bonus
    skills_mentioned = len([s for s in job_desc.skills if s.lower() in match.detailed_analysis.lower()])
    bonuses += skills_mentioned * 2
    
    # Experience quality bonus
    if match.experience_match > 85:
        bonuses += 10
    elif match.experience_match > 70:
        bonuses += 5
    
    competitive_score = min(100, base_score + bonuses)
    
    return competitive_score

async def generate_ranking_justification(
    match: CandidateMatch, 
    all_matches: List[CandidateMatch], 
    current_rank: int, 
    job_desc: JobDescription
) -> str:
    """Generate detailed justification for why this candidate ranks where they do"""
    
    justifications = []
    
    # Why this rank?
    if current_rank == 0:
        justifications.append("🏆 **TOP CANDIDATE** - Best overall match for this role")
        
        # Compare with #2
        if len(all_matches) > 1:
            second_best = all_matches[1]
            score_diff = match.competitive_score - second_best.competitive_score
            justifications.append(f"• Outperforms 2nd candidate by {score_diff:.1f} points")
            
            if match.skills_match > second_best.skills_match:
                justifications.append(f"• Superior skills match ({match.skills_match:.0f}% vs {second_best.skills_match:.0f}%)")
            
            if match.experience_match > second_best.experience_match:
                justifications.append(f"• More relevant experience ({match.experience_match:.0f}% vs {second_best.experience_match:.0f}%)")
    
    elif current_rank == 1:
        justifications.append("🥈 **STRONG SECOND CHOICE** - High-quality candidate")
        top_candidate = all_matches[0]
        score_diff = top_candidate.competitive_score - match.competitive_score
        justifications.append(f"• {score_diff:.1f} points behind top candidate")
        
        # Find what they're missing
        if match.skills_match < top_candidate.skills_match:
            justifications.append(f"• Skills gap: {top_candidate.skills_match - match.skills_match:.0f} points lower")
        if match.experience_match < top_candidate.experience_match:
            justifications.append(f"• Experience gap: {top_candidate.experience_match - match.experience_match:.0f} points lower")
    
    elif current_rank == 2:
        justifications.append("🥉 **SOLID THIRD OPTION** - Good potential candidate")
        top_candidate = all_matches[0]
        justifications.append(f"• Key areas for improvement to reach top tier:")
        
        if match.skills_match < 70:
            justifications.append(f"  - Skills development needed (currently {match.skills_match:.0f}%)")
        if match.experience_match < 70:
            justifications.append(f"  - More relevant experience required (currently {match.experience_match:.0f}%)")
    
    else:
        justifications.append(f"📋 **CANDIDATE #{current_rank + 1}** - Requires careful consideration")
        
        # Find main weaknesses
        if match.skills_match < 50:
            justifications.append("• Major skills gap identified")
        if match.experience_match < 50:
            justifications.append("• Limited relevant experience")
        if match.education_match < 50:
            justifications.append("• Educational background may not align")
        
        # Position in ranking
        total_candidates = len(all_matches)
        percentile = ((total_candidates - current_rank) / total_candidates) * 100
        justifications.append(f"• Ranks in {percentile:.0f}th percentile of applicants")
    
    return "\n".join(justifications)

async def enhance_analysis_with_comparison(
    match: CandidateMatch, 
    all_matches: List[CandidateMatch], 
    current_rank: int, 
    job_desc: JobDescription
) -> str:
    """Enhance the detailed analysis with competitive comparison"""
    
    base_analysis = match.detailed_analysis
    
    # Add competitive insights
    competitive_insights = []
    
    # Compare with average
    avg_score = sum(m.competitive_score for m in all_matches) / len(all_matches)
    score_vs_avg = match.competitive_score - avg_score
    
    if score_vs_avg > 15:
        competitive_insights.append(f"📈 **SIGNIFICANTLY ABOVE AVERAGE** (+{score_vs_avg:.1f} points)")
    elif score_vs_avg > 5:
        competitive_insights.append(f"📊 **ABOVE AVERAGE** (+{score_vs_avg:.1f} points)")
    elif score_vs_avg < -15:
        competitive_insights.append(f"📉 **BELOW AVERAGE** ({score_vs_avg:.1f} points)")
    elif score_vs_avg < -5:
        competitive_insights.append(f"📊 **SLIGHTLY BELOW AVERAGE** ({score_vs_avg:.1f} points)")
    else:
        competitive_insights.append(f"📊 **AVERAGE PERFORMANCE** ({score_vs_avg:+.1f} points)")
    
    # Identify unique strengths
    unique_strengths = []
    if current_rank == 0:
        unique_strengths.append("🎯 **BEST OVERALL CANDIDATE** - Top choice for immediate consideration")
    
    if match.skills_match == max(m.skills_match for m in all_matches):
        unique_strengths.append("🔧 **STRONGEST TECHNICAL SKILLS** among all candidates")
    
    if match.experience_match == max(m.experience_match for m in all_matches):
        unique_strengths.append("💼 **MOST RELEVANT EXPERIENCE** in the candidate pool")
    
    if match.education_match == max(m.education_match for m in all_matches):
        unique_strengths.append("🎓 **BEST EDUCATIONAL FIT** for role requirements")
    
    # Combine all insights
    enhanced_analysis = f"""
{base_analysis}

**COMPETITIVE ANALYSIS:**
{chr(10).join(competitive_insights)}

**UNIQUE ADVANTAGES:**
{chr(10).join(unique_strengths) if unique_strengths else "• Standard qualifications relative to other candidates"}

**RANKING JUSTIFICATION:**
{match.ranking_justification if hasattr(match, 'ranking_justification') else "Ranking calculation in progress..."}
"""
    
    return enhanced_analysis.strip()

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    matches: List[CandidateMatch] = []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Resume Screening API is running!",
        "status": "healthy",
        "version": "2.0.0",
        "ai_provider": "Groq (Ultra-fast)"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test Groq connection
        test_result = await job_matcher.test_connection()
        return {
            "status": "healthy",
            "ai_service": "operational" if test_result else "degraded",
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/parse-resume", response_model=CandidateProfile)
async def parse_resume(file: UploadFile = File(...)):
    """Parse a single resume and extract candidate information"""
    try:
        logger.info(f"Parsing resume: {file.filename}")
        
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        
        try:
            # Parse resume
            candidate = await resume_parser.parse_resume(file_path)
            
            logger.info(f"Successfully parsed resume for {candidate.name}")
            return candidate
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {str(e)}")

@app.post("/analyze-single", response_model=ResumeAnalysis)
async def analyze_single_resume(
    job_description: str = Form(...),
    file: UploadFile = File(...)
):
    """Analyze a single resume against a job description"""
    try:
        logger.info(f"Analyzing resume: {file.filename}")
        
        # Parse job description
        job_desc = JobDescription.from_text(job_description)
        
        # Save and parse resume
        file_path = await save_uploaded_file(file)
        
        try:
            # Parse resume
            candidate = await resume_parser.parse_resume(file_path)
            
            # Analyze match
            analysis = await job_matcher.analyze_candidate(candidate, job_desc)
            
            logger.info(f"Analysis complete for {candidate.name}: {analysis.overall_score}%")
            return analysis
            
        finally:
            # Cleanup
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"Single analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk_resumes(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Analyze multiple resumes against a job description"""
    try:
        logger.info(f"Starting bulk analysis of {len(files)} resumes")
        
        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 resumes allowed per batch for optimal performance")
        
        # Parse job description
        job_desc = JobDescription.from_text(job_description)
        
        results = []
        failed_files = []
        
        for file in files:
            try:
                # Save and parse resume
                file_path = await save_uploaded_file(file)
                
                try:
                    # Parse resume
                    candidate = await resume_parser.parse_resume(file_path)
                    
                    # Analyze match
                    analysis = await job_matcher.analyze_candidate(candidate, job_desc)
                    analysis.filename = file.filename
                    
                    results.append(analysis)
                    
                finally:
                    # Cleanup individual file
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        
            except Exception as e:
                logger.error(f"Failed to analyze {file.filename}: {e}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info(f"Bulk analysis complete: {len(results)} successful, {len(failed_files)} failed")
        
        return BulkAnalysisResponse(
            total_resumes=len(files),
            successful_analyses=len(results),
            failed_analyses=len(failed_files),
            results=results,
            failed_files=failed_files,
            job_description=job_desc
        )
        
    except Exception as e:
        logger.error(f"Bulk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")

@app.post("/process-resumes", response_model=ProcessingResponse)
async def process_resumes(
    resumes: List[UploadFile] = File(...),
    jobTitle: str = Form(...),
    requirements: str = Form(...),
    skills: str = Form(default=""),
    experience: str = Form(default=""),
    education: str = Form(default=""),
    certifications: str = Form(default=""),
    responsibilities: str = Form(default="")
):
    """🚀 OPTIMIZED: Process resumes with enhanced bulk performance"""
    import asyncio
    import time
    try:
        start_time = time.time()
        total_resumes = len(resumes)
        logger.info(f"🚀 BULK PROCESSING: {total_resumes} resumes for position: {jobTitle}")

        # 📊 Performance tracking
        processing_stats = {
            "total_resumes": total_resumes,
            "successful": 0,
            "failed": 0,
            "start_time": start_time,
            "ai_analyses": 0,
            "fallback_analyses": 0
        }

        # Create comprehensive job description
        job_description_text = f"""
        Job Title: {jobTitle}

        Requirements: {requirements}

        Required Skills: {skills}
        Experience Required: {experience}
        Education Required: {education}
        Certifications: {certifications}
        Key Responsibilities: {responsibilities}
        """

        # Parse job description
        job_desc = JobDescription.from_text(job_description_text)

        matches = []
        
        # 🔥 ENHANCED CONCURRENT PROCESSING with controlled concurrency
        semaphore = asyncio.Semaphore(10)  # Limit concurrent AI calls to prevent overload

        async def process_single_resume_optimized(resume_file):
            async with semaphore:  # Control concurrency
                try:
                    # Save uploaded file
                    file_path = await save_uploaded_file(resume_file)
                    try:
                        # ⚡ OPTIMIZED: Parse resume with timeout
                        candidate = await asyncio.wait_for(
                            resume_parser.parse_resume(file_path), 
                            timeout=30  # 30s timeout for parsing
                        )
                        candidate_store[resume_file.filename] = candidate
                        
                        # ⚡ OPTIMIZED: Analyze match with stricter timeout
                        try:
                            analysis = await asyncio.wait_for(
                                job_matcher.analyze_candidate(candidate, job_desc), 
                                timeout=45  # 45s timeout for AI analysis
                            )
                            processing_stats["ai_analyses"] += 1
                            processing_stats["successful"] += 1
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"⏰ AI timeout for {resume_file.filename} - using fallback")
                            # Use fallback scoring instead of failing completely
                            analysis = await job_matcher.get_fallback_analysis(candidate, job_desc)
                            processing_stats["fallback_analyses"] += 1
                            processing_stats["successful"] += 1
                            
                        match = CandidateMatch(
                            filename=resume_file.filename,
                            score=analysis.get("overall_score", 0) if isinstance(analysis, dict) else analysis.overall_score,
                            rank=0,
                            strengths=analysis.get("strengths", []) if isinstance(analysis, dict) else analysis.strengths,
                            weaknesses=analysis.get("weaknesses", []) if isinstance(analysis, dict) else analysis.weaknesses,
                            summary=analysis.get("fit_summary", "") if isinstance(analysis, dict) else analysis.fit_summary,
                            detailed_analysis=analysis.get("detailed_reasoning", "") if isinstance(analysis, dict) else analysis.detailed_feedback,
                            skills_match=analysis.get("skills_score", 0) if isinstance(analysis, dict) else analysis.skills_score,
                            experience_match=analysis.get("experience_score", 0) if isinstance(analysis, dict) else analysis.experience_score,
                            education_match=analysis.get("education_score", 0) if isinstance(analysis, dict) else analysis.education_score
                        )
                        matches.append(match)
                        
                    finally:
                        # 🧹 CLEANUP: Remove temp file immediately
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            
                except Exception as e:
                    processing_stats["failed"] += 1
                    logger.error(f"❌ Failed to process {resume_file.filename}: {e}")
                    failed_match = CandidateMatch(
                        filename=resume_file.filename,
                        score=0,
                        rank=999,
                        strengths=["Resume uploaded successfully"],
                        weaknesses=["AI processing failed - requires manual review"],
                        summary=f"Processing failed: {str(e)}",
                        detailed_analysis="AI analysis unavailable - fallback scoring used",
                        skills_match=0,
                        experience_match=0,
                        education_match=0
                    )
                    matches.append(failed_match)

        # 🚀 PROCESS ALL RESUMES CONCURRENTLY with progress tracking
        logger.info(f"⚡ Starting concurrent processing of {total_resumes} resumes...")
        await asyncio.gather(*(process_single_resume_optimized(resume_file) for resume_file in resumes))

        # 📊 PERFORMANCE METRICS
        total_time = time.time() - start_time
        avg_time_per_resume = total_time / total_resumes if total_resumes > 0 else 0
        
        logger.info(f"🎯 BULK PROCESSING COMPLETE:")
        logger.info(f"   ✅ Total Processed: {processing_stats['successful']}/{total_resumes}")
        logger.info(f"   ❌ Failed: {processing_stats['failed']}")
        logger.info(f"   🤖 AI Analyses: {processing_stats['ai_analyses']}")
        logger.info(f"   🔄 Fallback Analyses: {processing_stats['fallback_analyses']}")
        logger.info(f"   ⏱️ Total Time: {total_time:.1f}s")
        logger.info(f"   📈 Avg Time/Resume: {avg_time_per_resume:.1f}s")

        # Sort by score and assign ranks with intelligent comparison
        matches = await smart_ranking_with_justification(matches, job_desc)

        return ProcessingResponse(
            success=True,
            message=f"🚀 Successfully processed {processing_stats['successful']}/{total_resumes} resumes in {total_time:.1f}s (avg: {avg_time_per_resume:.1f}s/resume). AI: {processing_stats['ai_analyses']}, Fallback: {processing_stats['fallback_analyses']}",
            matches=matches
        )

    except Exception as e:
        logger.error(f"💥 Bulk processing error: {e}", exc_info=True)
        # Always return a response, even on error
        return ProcessingResponse(
            success=False,
            message=f"❌ Error processing resumes: {str(e)}",
            matches=[]
        )

class CandidateContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    score: float
    rank: int

@app.post("/get-candidate-contacts")
async def get_candidate_contacts(request: dict) -> List[CandidateContactInfo]:
    """Extract contact information from candidate matches - FIXED VERSION"""
    try:
        matches_data = request.get("matches", [])
        threshold = request.get("threshold", 70.0)
        
        logger.info(f"Processing {len(matches_data)} matches with threshold {threshold}")
        
        # Convert dict data to match objects if needed
        qualified_candidates = []
        for match_data in matches_data:
            if isinstance(match_data, dict):
                score = match_data.get("score", 0)
                if score >= threshold:
                    qualified_candidates.append(match_data)
            else:
                # If it's already an object, check score
                if hasattr(match_data, 'score') and match_data.score >= threshold:
                    qualified_candidates.append(match_data)
        
        logger.info(f"Found {len(qualified_candidates)} qualified candidates")
        
        contact_list = []
        for match in qualified_candidates:
            # Handle both dict and object formats
            if isinstance(match, dict):
                filename = match.get("filename", "Unknown")
                score = match.get("score", 0)
                rank = match.get("rank", 999)
                summary = match.get("summary", "")
                detailed_analysis = match.get("detailed_analysis", "")
            else:
                filename = getattr(match, 'filename', 'Unknown')
                score = getattr(match, 'score', 0)
                rank = getattr(match, 'rank', 999)
                summary = getattr(match, 'summary', '')
                detailed_analysis = getattr(match, 'detailed_analysis', '')
            
            # Default values
            name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '').replace('_', ' ').replace('-', ' ').strip().title()
            email = "Not found"
            phone = "Not found"
            
            logger.info(f"Processing: {filename}")
            
            # Strategy 1: Get from stored candidate (this is where the good data is!)
            stored_candidate = candidate_store.get(filename)
            if stored_candidate:
                name = getattr(stored_candidate, 'name', name) or name
                stored_email = getattr(stored_candidate, 'email', '')
                stored_phone = getattr(stored_candidate, 'phone', '')
                
                if stored_email and stored_email != "Not found" and stored_email.strip():
                    email = stored_email.strip()
                if stored_phone and stored_phone != "Not found" and stored_phone.strip():
                    phone = stored_phone.strip()
                
                logger.info(f"From stored candidate: name='{name}', email='{email}', phone='{phone}'")
            
            # Strategy 2: If still not found, use enhanced extraction
            if email == "Not found" or phone == "Not found":
                combined_text = f"{summary} {detailed_analysis} {filename}"
                extracted = enhanced_extract_contact_from_text(combined_text)
                
                if email == "Not found" and extracted.get("email"):
                    email = extracted["email"]
                if phone == "Not found" and extracted.get("phone"):
                    phone = extracted["phone"]
                
                logger.info(f"After extraction: email='{email}', phone='{phone}'")
            
            contact_info = CandidateContactInfo(
                name=name,
                email=email,
                phone=phone,
                score=float(score),
                rank=int(rank)
            )
            
            contact_list.append(contact_info)
            logger.info(f"Added contact: {contact_info.name} | {contact_info.email} | {contact_info.phone}")
        
        logger.info(f"Returning {len(contact_list)} contacts")
        return contact_list
        
    except Exception as e:
        logger.error(f"Error in get_candidate_contacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Contact extraction failed: {str(e)}")


def enhanced_extract_contact_from_text(text: str) -> Dict[str, str]:
    """Simple but robust contact extraction using data scraping approach"""
    import re
    
    if not text:
        return {"email": "", "phone": ""}
    
    contact_info = {"email": "", "phone": ""}
    
    # Method 1: Your specific format - email — phone or email - phone
    # This handles formats like: kattakarthik8008@gmail.com — 8008456790
    specific_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s*[—–\-]+\s*(\d{8,15})'
    specific_matches = re.findall(specific_pattern, text, re.IGNORECASE)
    if specific_matches:
        email, phone = specific_matches[0]
        contact_info["email"] = email.strip()
        contact_info["phone"] = phone.strip()
        return contact_info
    
    # Method 2: Find email and phone separately
    # Email extraction - simple and reliable
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    email_matches = re.findall(email_pattern, text, re.IGNORECASE)
    if email_matches:
        contact_info["email"] = email_matches[0].strip()
    
    # Phone extraction - focus on 10-digit numbers
    phone_patterns = [
        r'\b(\d{10})\b',  # Exactly 10 digits
        r'\+91\s*(\d{10})',  # +91 format
        r'91\s*(\d{10})',  # 91 format
        r'(\d{8,15})',  # 8-15 digits (catch-all)
    ]
    
    for pattern in phone_patterns:
        phone_matches = re.findall(pattern, text)
        if phone_matches:
            phone = phone_matches[0].strip()
            # Prefer 10-digit numbers starting with 6,7,8,9 (Indian mobile)
            if len(phone) == 10 and phone[0] in '6789':
                contact_info["phone"] = phone
                break
            elif len(phone) >= 10:
                contact_info["phone"] = phone
                break
    
    return contact_info
    
    # Extract emails
    emails = set()
    for pattern in email_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                email = match[0] if match[0] else match[-1]
            else:
                email = match
            
            email = email.strip()
            # Basic validation
            if '@' in email and '.' in email.split('@')[-1] and len(email) > 5:
                emails.add(email)
    
    # Extract phones
    phones = set()
    for pattern in phone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                phone = ''.join(str(x) for x in match if x)
            else:
                phone = str(match)
            
            # Clean and validate
            phone = phone.strip()
            phone_digits = re.sub(r'[^\d]', '', phone)
            
            if len(phone_digits) >= 10:
                phones.add(phone)
    
    # Select best email
    if emails:
        best_email = sorted(emails, key=lambda x: (
            len(x.split('@')[0]),
            any(domain in x.lower() for domain in ['gmail', 'yahoo', 'hotmail', 'outlook', 'company']),
            len(x)
        ), reverse=True)[0]
        contact_info["email"] = best_email
    
    # Select best phone
    if phones:
        best_phone = sorted(phones, key=lambda x: (
            x.strip().startswith('+'),
            len(re.sub(r'[^\d]', '', x)),
            len(x)
        ), reverse=True)[0]
        contact_info["phone"] = best_phone.strip()
    
    return contact_info

@app.get("/models")
async def get_available_models():
    """Get information about available AI models"""
    return {
        "current_provider": "Groq",
        "current_model": "llama-3.1-8b-instant",
        "features": [
            "Ultra-fast processing (1-2 seconds)",
            "High accuracy analysis",
            "Bulk processing support",
            "Real-time feedback"
        ],
        "performance": {
            "avg_processing_time": "1-2 seconds",
            "max_batch_size": 100,
            "concurrent_requests": "up to 10 simultaneous AI analyses",
            "performance": "optimized for 40-100 resumes",
            "fallback_protection": "automatic fallback if AI times out"
        }
    }

@app.delete("/cleanup")
async def cleanup_temp_files_endpoint():
    """Cleanup temporary files"""
    try:
        cleanup_temp_files()
        return {"message": "Temporary files cleaned up successfully"}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Resume Screening API...")
    print("🤖 AI Provider: Groq (Ultra-fast)")
    print("🌐 Frontend: http://localhost:3000")
    print("📋 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
