#!/usr/bin/env python3
"""
Analyze transcript files using LLM (OpenAI or Anthropic) to generate insights.
For each .txt file in the specified directory, generates an 'insights-{filename}.txt' file.

Supports both OpenAI (GPT-4, GPT-4o, O1, etc.) and Anthropic (Claude) models.
Automatically detects the provider from the model name and uses the appropriate API key.
"""

import os
import sys
import argparse
import logging
import json
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def detect_provider(model_name):
    """Detect which provider (OpenAI or Anthropic) a model belongs to."""
    model_lower = model_name.lower()
    
    # Anthropic models typically start with 'claude'
    if model_lower.startswith('claude'):
        return 'anthropic'
    
    # OpenAI models typically start with 'gpt' or 'o1' or are in specific formats
    if (model_lower.startswith('gpt') or 
        model_lower.startswith('o1') or 
        model_lower.startswith('o3') or
        'gpt-' in model_lower or
        'gpt4' in model_lower or
        'gpt3' in model_lower):
        return 'openai'
    
    # Default to Anthropic if uncertain (backward compatibility)
    return 'anthropic'


def create_client(provider, api_key):
    """Create the appropriate client based on provider."""
    if provider == 'openai':
        return OpenAI(api_key=api_key)
    elif provider == 'anthropic':
        return Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_api_key(provider, api_key_arg=None):
    """Get API key from argument, environment variable, or .env file."""
    if provider == 'openai':
        env_key = 'OPENAI_API_KEY'
    elif provider == 'anthropic':
        env_key = 'ANTHROPIC_API_KEY'
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Priority: command-line argument > environment variable > .env file
    api_key = api_key_arg or os.getenv(env_key)
    
    if not api_key:
        raise ValueError(
            f"{provider.capitalize()} API key is required. "
            f"Set it via --api-key argument, {env_key} environment variable, "
            f"or {env_key} in .env file"
        )
    
    return api_key

# Original (comprehensive) system message for Claude
ORIGINAL_SYSTEM_MESSAGE = """You are an expert call analyst specializing in extracting actionable insights from customer service transcripts. Your role is to:

1. IDENTIFY ISSUES: Detect problems, friction points, customer dissatisfaction, process breakdowns, miscommunications, or areas where customer needs were not met.

2. GENERATE INSIGHTS: Analyze patterns, root causes, emotional indicators, and underlying factors that contribute to call outcomes (positive or negative).

3. PROVIDE RECOMMENDATIONS: Suggest specific, actionable improvements to processes, scripts, training, or systems that could enhance customer experience and operational efficiency.

ANALYSIS FRAMEWORK:

- Call Outcome: Success/Partial Success/Failure

- Customer Sentiment: Positive/Neutral/Negative/Mixed

- Key Pain Points: What went wrong or could be improved

- Strengths: What went well

- Root Cause Analysis: Why issues occurred

- Impact Assessment: Business impact (customer retention, efficiency, compliance)

- Priority Level: High/Medium/Low for each recommendation

OUTPUT FORMAT:

Provide structured analysis in clear sections with bullet points for readability. Be concise but comprehensive. Focus on actionable insights rather than mere description.

PRINCIPLES:

- Be objective and data-driven

- Identify systemic issues, not just individual errors

- Consider both customer and agent perspectives

- Prioritize recommendations by potential impact

- Flag compliance or risk issues immediately"""

# Original (comprehensive) user message template
ORIGINAL_USER_MESSAGE_TEMPLATE = """Analyze the following call transcript(s) to identify issues, generate insights, and provide actionable recommendations.

═════════════════════════════════════════════════════════
CONTEXT INFORMATION
═════════════════════════════════════════════════════════

Use the following general context (adapt or ignore based on the transcript):

Overview: This call involves an interaction between a customer (or user) and a representative (human or AI). The purpose, industry, and specific objectives may vary across recordings.

Success Definition:
Define success based on what the transcript suggests (e.g., issue resolved, customer question answered, purchase completed, appointment scheduled, process completed, escalation avoided, etc.)

Failure Definition:
Define failure based on the transcript (e.g., unresolved issue, customer dissatisfaction, incorrect information provided, customer drop-off, escalation required, miscommunication, etc.)

Potential Focus Areas for Analysis (as applicable):

Clarity of information delivery

Resolution effectiveness

Customer experience and emotional journey

Agent communication quality

Script adherence and flexibility

Accuracy of information

Process clarity and next-step communication

Conversion or completion barriers

Misunderstandings or breakdowns

Tone, rapport, and empathy

Technical, policy, or compliance adherence

(The model should infer and adapt based on the transcript provided.)

════════════════════════════════════════════════════════
TRANSCRIPT(S)
═════════════════════════════════════════════════════════

{{input}}

═════════════════════════════════════════════════════════
REQUIRED ANALYSIS
═════════════════════════════════════════════════════════

Please provide a comprehensive analysis structured as follows:

1. EXECUTIVE SUMMARY

Provide a 2–3 sentence overview including:

Who the customer is (if identifiable) and the purpose of the call

Call outcome (Success / Partial Success / Failure)

Primary factor driving the outcome

Overall customer sentiment

1. ISSUES IDENTIFIED

Categorize all problems by severity and provide concrete examples from the transcript.

🔴 CRITICAL ISSUES

Issues that directly caused call failure, created customer dissatisfaction, introduced risk, or significantly harmed the experience.

Example concerns to look for:

Incorrect or misleading information

Compliance/policy steps missing (if applicable)

Customer issue not addressed or resolved

Severe misunderstanding or communication breakdown

Major unmet expectations

Technical issues that block progress

🟡 MAJOR ISSUES

Significant problems that negatively affected efficiency, clarity, or experience, but did not fully derail the call.

Example concerns:

Confusing or incomplete explanations

Late introduction of important information

Multiple clarifications required

Customer confusion due to wording, tone, or pacing

Poor handling of questions

🟢 MINOR ISSUES

Small friction points or optimization opportunities.

Example concerns:

Slightly repetitive or robotic phrasing

Missed opportunities to build rapport

Information that could have been proactive

Minor pacing or flow issues

For each issue, specify:

What occurred

Where it occurred

The impact on customer understanding, satisfaction, or outcome

1. KEY INSIGHTS

Provide deeper analysis addressing:

Root Cause Analysis:

Why did the issues happen?

Are they due to script design, agent skill, customer behavior, tools, or unclear processes?

Pattern Recognition:

Recurring misunderstandings or communication gaps

Repeating themes or bottlenecks

Common customer frustrations or needs

Customer & Agent Behavior:

Customer emotional journey and responsiveness

Agent performance—clarity, empathy, organization, adaptability

Interaction quality and tone

Outcome Barriers:

What specifically prevented or supported call success?

Could barriers have been anticipated or avoided?

Technology / AI Performance (if applicable):

Response accuracy and naturalness

Ability to handle unexpected questions

Script alignment with customer needs

Opportunities for more adaptive or personalized interaction

1. RECOMMENDATIONS

Organize recommendations by priority and category:

HIGH PRIORITY (Immediate – 1–2 Weeks)

Impact: Addresses critical issues and yields quick improvements.

Communication / Script Improvements:

- Specific wording or structural changes

- Key information to reposition or emphasize

Process Enhancements:

- Workflow adjustments

- Better qualification, verification, or troubleshooting steps

Customer Experience Improvements:

- How to reduce confusion or friction

- Clearer next-step instructions

MEDIUM PRIORITY (Next 1–2 Months)

Impact: Enhances reliability, consistency, and experience.

Training & Coaching:

- Skills agents or AI models should strengthen

- Handling of edge cases and questions

Information Delivery Optimization:

- Sequencing improvements

- Clearer structuring of details

System / Tool Improvements:

- System fixes, integrations, or UI improvements

Multi-language / Accessibility (if relevant):

- Enhancements for broader support

LOW PRIORITY (Long-Term Enhancements)

Impact: Strategic or future-forward improvements.

- Advanced features

- Long-term process redesign

- Next-generation AI or system improvements

- Future opportunities for automation or personalization

1. SENTIMENT & EXPERIENCE ANALYSIS

Overall Sentiment: [Positive / Neutral / Negative / Mixed]

Sentiment Journey:

Beginning: Initial emotion

Middle: Key emotional shifts

End: Final sentiment

Critical Emotional Moments:

- Points of frustration, confusion, relief, satisfaction

Customer Effort Analysis:

Was the interaction easy or difficult?

Did the customer understand the path to resolution?

1. QUICK WINS

List 3–5 low-effort, high-impact improvements:

- Quick win

- Quick win

- Quick win

- (Optional)

- (Optional)

1. METRICS TO TRACK

Universal Metrics (usable in any industry):

Resolution/Success rate

Average Handle Time (AHT)

Customer Satisfaction Score (CSAT)

First Contact Resolution (FCR)

Escalation rate

Abandonment or drop-off rates

Information accuracy rate

Sentiment trend across calls

Context-Dependent Metrics (adapt as needed):

Completion or conversion rates

Appointment scheduling or follow-through

Policy/script adherence

Frequency of repeated questions

Time to resolution

Customer effort score

1. FOLLOW-UP ACTIONS

Immediate Actions:

- Urgent follow-ups or corrections

- Any risks or compliance concerns to address

Team / Agent Actions:

- Training updates

- Script adjustments

- Escalation or routing improvements

System / Management Actions:

- Policy or process review

- Tool or system changes

- Cross-team coordination needs

1. COMPARATIVE ANALYSIS (If Multiple Transcripts Provided)

Common Themes Across Calls:

- Patterns observed

Best Practices Identified:

- What consistently worked well

Recurring Failure Points:

- Common issues or bottlenecks

Pattern-Based Recommendations:

- Derived insights from multiple conversations

═════════════════════════════════════════════════════════
END OF ANALYSIS
═════════════════════════════════════════════════════════"""

# Simple system message for Claude
SIMPLE_SYSTEM_MESSAGE = """You are a call analyst extracting actionable insights from customer service transcripts. Analyze issues, identify patterns, and recommend improvements."""

# Simple user message template
SIMPLE_USER_MESSAGE_TEMPLATE = """Analyze this call transcript:

{{input}}

Provide your analysis in this format:

1. EXECUTIVE SUMMARY

Call purpose and outcome (Success/Partial/Failure)

Customer sentiment (Positive/Neutral/Negative/Mixed)

Key takeaway

2. ISSUES IDENTIFIED

🔴 Critical - Caused failure or major dissatisfaction

Issue + impact

🟡 Major - Significantly affected experience or efficiency

Issue + impact

🟢 Minor - Small friction points or missed opportunities

Issue + impact

3. KEY INSIGHTS

Root causes of issues

Patterns or recurring themes

What prevented/enabled success

Agent/customer behavior observations

4. RECOMMENDATIONS

High Priority (Immediate)

Specific, actionable changes to scripts, processes, or training

Medium Priority (1-2 months)

Training needs, system improvements

Low Priority (Long-term)

Strategic enhancements

5. QUICK WINS

3-5 low-effort, high-impact improvements

6. FOLLOW-UP ACTIONS

Immediate actions needed

Metrics to track going forward

Key principles:

Be objective and specific

Focus on systemic issues, not just individual errors

Prioritize by impact

Cite examples from the transcript"""


def analyze_transcript(client, provider, transcript_content, model, prompt_version='original', logger=None):
    """Analyze a transcript using the specified LLM provider."""
    try:
        # Select prompt version
        if prompt_version == 'simple':
            system_message = SIMPLE_SYSTEM_MESSAGE
            user_message_template = SIMPLE_USER_MESSAGE_TEMPLATE
        else:  # default to 'original'
            system_message = ORIGINAL_SYSTEM_MESSAGE
            user_message_template = ORIGINAL_USER_MESSAGE_TEMPLATE
        
        # Replace {{input}} placeholder with actual transcript content
        user_message = user_message_template.replace("{{input}}", transcript_content)
        
        # Log request details
        if logger:
            logger.info("=" * 80)
            logger.info(f"API Request - Provider: {provider}, Model: {model}, Prompt Version: {prompt_version}")
            logger.info(f"System message length: {len(system_message)} characters")
            logger.info(f"User message length: {len(user_message)} characters")
            logger.info(f"Transcript content length: {len(transcript_content)} characters")
            logger.info("-" * 80)
            logger.debug(f"System message: {system_message[:500]}..." if len(system_message) > 500 else f"System message: {system_message}")
            logger.debug(f"User message preview (first 500 chars): {user_message[:500]}...")
            logger.info(f"Request timestamp: {datetime.now().isoformat()}")
        
        # Call API based on provider
        if provider == 'openai':
            # OpenAI API call
            response = client.chat.completions.create(
                model=model,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract text from OpenAI response
            analysis_text = response.choices[0].message.content if response.choices else None
            
            # Log response details
            if logger:
                logger.info("-" * 80)
                logger.info("API Response received")
                if hasattr(response, 'usage'):
                    logger.info(f"Usage - Input tokens: {response.usage.prompt_tokens}, Output tokens: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                if hasattr(response, 'id'):
                    logger.info(f"Response ID: {response.id}")
                if hasattr(response, 'model'):
                    logger.info(f"Model used: {response.model}")
                logger.info(f"Response timestamp: {datetime.now().isoformat()}")
        
        elif provider == 'anthropic':
            # Anthropic API call
            message = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            
            # Extract text from Anthropic response
            analysis_text = ""
            if message.content and len(message.content) > 0:
                for content_block in message.content:
                    if hasattr(content_block, 'text'):
                        analysis_text += content_block.text
            else:
                analysis_text = None
            
            # Log response details
            if logger:
                logger.info("-" * 80)
                logger.info("API Response received")
                if hasattr(message, 'usage'):
                    logger.info(f"Usage - Input tokens: {message.usage.input_tokens}, Output tokens: {message.usage.output_tokens}, Total: {message.usage.input_tokens + message.usage.output_tokens}")
                if hasattr(message, 'id'):
                    logger.info(f"Response ID: {message.id}")
                if hasattr(message, 'model'):
                    logger.info(f"Model used: {message.model}")
                logger.info(f"Response timestamp: {datetime.now().isoformat()}")
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Log response content preview
        if analysis_text:
            if logger:
                logger.info(f"Response content length: {len(analysis_text)} characters")
                logger.debug(f"Response preview (first 500 chars): {analysis_text[:500]}...")
                logger.info("=" * 80)
            return analysis_text
        else:
            if logger:
                logger.warning("Empty response received from API")
            return None
            
    except Exception as e:
        error_msg = f"Error analyzing transcript: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        return None


def save_insights(transcript_path, insights):
    """Save insights to a new file with 'insights-' prefix."""
    transcript_path_obj = Path(transcript_path)
    insights_path = transcript_path_obj.parent / f"insights-{transcript_path_obj.name}"
    
    try:
        with open(insights_path, 'w', encoding='utf-8') as f:
            f.write(insights)
        print(f"Saved insights: {insights_path}")
        return True
    except Exception as e:
        print(f"Error saving insights to {insights_path}: {str(e)}", file=sys.stderr)
        return False


def process_single_file(txt_file, client, provider, model, prompt_version, skip_existing, logger, stats_lock, stats):
    """Process a single transcript file. Returns (success, skipped, failed)."""
    insights_file = txt_file.parent / f"insights-{txt_file.name}"
    
    # Skip if insights file already exists
    if skip_existing and insights_file.exists():
        with stats_lock:
            stats['skipped'] += 1
        print(f"Skipping (insights file exists): {txt_file}")
        return 'skipped'
    
    # Read transcript content
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
    except Exception as e:
        with stats_lock:
            stats['failed'] += 1
        print(f"Error reading {txt_file}: {str(e)}", file=sys.stderr)
        return 'failed'
    
    if not transcript_content.strip():
        with stats_lock:
            stats['skipped'] += 1
        print(f"Warning: {txt_file} is empty, skipping.")
        return 'skipped'
    
    print(f"Analyzing: {txt_file}")
    
    # Analyze transcript
    insights = analyze_transcript(client, provider, transcript_content, model, prompt_version, logger)
    
    if insights:
        if save_insights(txt_file, insights):
            with stats_lock:
                stats['processed'] += 1
            return 'success'
        else:
            with stats_lock:
                stats['failed'] += 1
            return 'failed'
    else:
        with stats_lock:
            stats['failed'] += 1
        return 'failed'


def process_directory(directory, client, provider, model, prompt_version='original', skip_existing=True, logger=None, max_workers=5):
    """Recursively process all .txt files in a directory with parallel processing."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        return
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        return
    
    # Find all .txt files (excluding those that already start with 'insights-')
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == '.txt' and not file.startswith('insights-'):
                txt_files.append(file_path)
    
    if not txt_files:
        print(f"No .txt files found in '{directory}'")
        return
    
    print(f"Found {len(txt_files)} transcript file(s) to process.")
    print(f"Processing with {max_workers} worker thread(s).\n")
    
    # Thread-safe statistics
    stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    stats_lock = threading.Lock()
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file,
                txt_file,
                client,
                provider,
                model,
                prompt_version,
                skip_existing,
                logger,
                stats_lock,
                stats
            ): txt_file
            for txt_file in txt_files
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_file):
            txt_file = future_to_file[future]
            try:
                result = future.result()
            except Exception as e:
                with stats_lock:
                    stats['failed'] += 1
                print(f"Error processing {txt_file}: {str(e)}", file=sys.stderr)
    
    print(f"\nSummary:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze transcript files using Claude Sonnet 4.5 LLM to generate insights'
    )
    parser.add_argument(
        'directory',
        help='Directory to process (will recursively visit all subdirectories)'
    )
    parser.add_argument(
        '--api-key',
        help='API key (will auto-detect provider from model, or set OPENAI_API_KEY/ANTHROPIC_API_KEY environment variable)',
        default=None
    )
    parser.add_argument(
        '--model',
        help='LLM model to use (e.g., gpt-4, gpt-4o, claude-sonnet-4-20250514, o1-preview). Auto-detects provider from model name. Can also set via OPENAI_MODEL or ANTHROPIC_MODEL environment variable.',
        default=None
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-analyze files even if insights file already exists'
    )
    parser.add_argument(
        '--log-file',
        help='File to write API request/response logs (default: anthropic_api.log)',
        default='anthropic_api.log'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Disable logging to file (logs will still appear on console)'
    )
    parser.add_argument(
        '--prompt-version',
        choices=['original', 'simple'],
        default='original',
        help='Prompt version to use: "original" (comprehensive) or "simple" (concise) (default: original)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of parallel workers for processing files (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = None
    if not args.no_log:
        log_level = getattr(logging, args.log_level.upper())
        
        # Create logger
        logger = logging.getLogger('anthropic_api')
        logger.setLevel(log_level)
        
        # Create file handler
        file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # Create console handler for errors/warnings
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized - Level: {args.log_level}, File: {args.log_file}")
    
    # Get model from argument, environment variable, or use default
    model = (args.model or 
             os.getenv('OPENAI_MODEL') or 
             os.getenv('ANTHROPIC_MODEL') or 
             'claude-sonnet-4-20250514')
    
    # Detect provider from model name
    provider = detect_provider(model)
    
    if logger:
        logger.info(f"Detected provider: {provider} for model: {model}")
    
    # Get API key for the detected provider
    try:
        api_key = get_api_key(provider, args.api_key)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Create the appropriate client
    try:
        client = create_client(provider, api_key)
    except Exception as e:
        print(f"Error creating {provider} client: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using {provider.capitalize()} model: {model}\n")
    
    # Process directory
    process_directory(
        args.directory, 
        client,
        provider=provider,
        model=model,
        prompt_version=args.prompt_version,
        skip_existing=not args.no_skip_existing,
        logger=logger,
        max_workers=args.max_workers
    )
    
    if logger:
        logger.info("Processing completed")


if __name__ == '__main__':
    main()

