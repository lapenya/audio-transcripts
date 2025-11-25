#!/usr/bin/env python3
"""
Query an LLM with insights file content and save the response.
Takes a directory and model name as arguments, finds all 'insights-*.txt' files,
sends each file's content to the LLM, and saves responses with 'response-' prefix.
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

# Vertex AI imports (with error handling for optional dependency)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Google AI Studio imports (with error handling for optional dependency)
try:
    import google.generativeai as genai
    GOOGLE_AI_STUDIO_AVAILABLE = True
except ImportError:
    GOOGLE_AI_STUDIO_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()


def detect_provider(model_name):
    """Detect which provider (OpenAI, Anthropic, or Google) a model belongs to."""
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
    
    # Google Gemini models (can use either Vertex AI or Google AI Studio)
    if (model_lower.startswith('gemini') or 
        'gemini-' in model_lower or
        model_lower.startswith('vertex') or
        'vertex-' in model_lower):
        return 'google'
    
    # Default to Anthropic if uncertain
    return 'anthropic'


def get_google_provider():
    """Determine which Google provider to use (Vertex AI or AI Studio) based on env var."""
    provider = os.getenv('GOOGLE_AI_PROVIDER', 'vertex').lower()
    if provider in ['studio', 'ai-studio', 'aistudio', 'google-ai-studio']:
        return 'google-ai-studio'
    elif provider in ['vertex', 'vertex-ai', 'vertexai']:
        return 'vertex-ai'
    else:
        # Default to vertex if unknown value
        return 'vertex-ai'


def create_client(provider, api_key=None, project_id=None, location=None):
    """Create the appropriate client based on provider."""
    if provider == 'openai':
        return OpenAI(api_key=api_key)
    elif provider == 'anthropic':
        return Anthropic(api_key=api_key)
    elif provider == 'google':
        # Determine which Google provider to use
        google_provider = get_google_provider()
        
        if google_provider == 'google-ai-studio':
            # Google AI Studio (simpler, just needs API key)
            if not GOOGLE_AI_STUDIO_AVAILABLE:
                raise ImportError(
                    "google-generativeai is not installed. "
                    "Install it with: poetry add google-generativeai"
                )
            if not api_key:
                raise ValueError(
                    "API key is required for Google AI Studio. "
                    "Set GOOGLE_VERTEX_API_KEY or use --api-key"
                )
            # Configure the API key for Google AI Studio
            # This must be done before creating any model instances
            genai.configure(api_key=api_key)
            return {
                'provider': 'google-ai-studio',
                'api_key': api_key
            }
        else:
            # Vertex AI (needs project ID and location)
            if not VERTEX_AI_AVAILABLE:
                raise ImportError(
                    "google-cloud-aiplatform is not installed. "
                    "Install it with: poetry add google-cloud-aiplatform"
                )
            if not project_id:
                raise ValueError("Google Cloud project_id is required for Vertex AI")
            if not location:
                location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            return {
                'provider': 'vertex-ai',
                'project_id': project_id,
                'location': location,
                'api_key': api_key
            }
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_api_key(provider, api_key_arg=None, project_id_arg=None, location_arg=None):
    """Get API key or credentials from argument, environment variable, or .env file."""
    if provider == 'openai':
        env_key = 'OPENAI_API_KEY'
        api_key = api_key_arg or os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"{provider.capitalize()} API key is required. "
                f"Set it via --api-key argument, {env_key} environment variable, "
                f"or {env_key} in .env file"
            )
        return api_key, None, None
    
    elif provider == 'anthropic':
        env_key = 'ANTHROPIC_API_KEY'
        api_key = api_key_arg or os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"{provider.capitalize()} API key is required. "
                f"Set it via --api-key argument, {env_key} environment variable, "
                f"or {env_key} in .env file"
            )
        return api_key, None, None
    
    elif provider == 'google':
        # Determine which Google provider to use
        google_provider = get_google_provider()
        
        if google_provider == 'google-ai-studio':
            # Google AI Studio only needs API key
            # Priority: command-line argument > GOOGLE_AI_STUDIO_API_KEY > GOOGLE_VERTEX_API_KEY (for backward compatibility)
            api_key = (api_key_arg or 
                      os.getenv('GOOGLE_AI_STUDIO_API_KEY') or 
                      os.getenv('GOOGLE_VERTEX_API_KEY'))
            if not api_key:
                raise ValueError(
                    "API key is required for Google AI Studio. "
                    "Set it via --api-key argument, GOOGLE_AI_STUDIO_API_KEY environment variable "
                    "(or GOOGLE_VERTEX_API_KEY for backward compatibility), "
                    "or GOOGLE_AI_STUDIO_API_KEY in .env file"
                )
            return api_key, None, None
        else:
            # Vertex AI needs project_id and optionally location
            # Priority: command-line argument > environment variable > .env file
            project_id = (project_id_arg or 
                         os.getenv('GOOGLE_CLOUD_PROJECT_ID'))
            location = (location_arg or 
                       os.getenv('GOOGLE_CLOUD_LOCATION') or 
                       'us-central1')
            
            if not project_id:
                raise ValueError(
                    "Google Cloud project_id is required for Vertex AI. "
                    "Set it via --project-id argument, GOOGLE_CLOUD_PROJECT_ID environment variable, "
                    "or GOOGLE_CLOUD_PROJECT_ID in .env file"
                )
            
            # Check for API key (GOOGLE_VERTEX_API_KEY) or credentials
            api_key = api_key_arg or os.getenv('GOOGLE_VERTEX_API_KEY')
            if not api_key:
                # Fall back to checking for service account credentials
                creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if not creds_path and not os.path.exists(os.path.expanduser('~/.config/gcloud/application_default_credentials.json')):
                    print("Warning: No Google Cloud credentials found. Set GOOGLE_VERTEX_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, or run 'gcloud auth application-default login'", file=sys.stderr)
            
            return api_key, project_id, location
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


# System message for aggregation
SYSTEM_MESSAGE = """You are a strategic operations analyst specializing in aggregating call analysis data to identify systemic recruiting patterns, prioritize improvements, and provide actionable recommendations for operations, product, and engineering teams.

Analyze candidate screening call reports to generate a strategic recruitment optimization report in strict JSON format."""

# User message template for aggregation
USER_MESSAGE_TEMPLATE = """Aggregate the following individual call analysis reports to identify recurring issues, patterns, and recommendations across all calls.

═══════════════════════════════════════════════════════════════
GENERAL CONTEXT
═══════════════════════════════════════════════════════════════

This dataset contains multiple call analysis reports produced using a standardized call evaluation framework.

Extract and reference specific business details from the call reports such as:
- Company names, products, or services mentioned
- Specific processes, tools, or systems referenced
- Industry-specific terminology or workflows
- Unique business rules, policies, or procedures
- Customer types, segments, or personas identified
- Agent roles, departments, or team structures mentioned

Success Definition:
Infer from the analyses (e.g., issue resolved, conversion completed, next steps confirmed, customer satisfied).

Failure Definition:
Infer from the analyses (e.g., unresolved issue, confusion, customer dissatisfaction, breakdown in process, no follow-through).

═══════════════════════════════════════════════════════════════
INDIVIDUAL CALL REPORTS
═══════════════════════════════════════════════════════════════

{{input}}

═══════════════════════════════════════════════════════════════
AGGREGATION OBJECTIVE
═══════════════════════════════════════════════════════════════

Aggregate all individual call analyses to produce a unified dataset that identifies:

- Recurring patterns and systemic issues
- Shared communication or process failures
- Conversion, resolution, or experience blockers
- Root causes behind cross-call trends
- Cross-functional improvement opportunities
- Common agent and customer behavior patterns
- Priority recommendations applicable across many calls

═══════════════════════════════════════════════════════════════
ANALYSIS METHODOLOGY
═══════════════════════════════════════════════════════════════

Apply the following:

Pattern Identification:
Detect repeated issues, behaviors, or bottlenecks appearing across multiple calls.

Issue Clustering:
Group similar issues by nature (communication, process, technology, training, clarity, compliance, experience, systems) and severity (critical, major, minor).

Impact Assessment:
Estimate effect on outcomes such as resolution, customer satisfaction, efficiency, or clarity.

Root Cause Analysis:
Identify systemic drivers such as unclear instructions, script gaps, agent training gaps, missing tools, customer uncertainty, or process inconsistencies.

Recommendations:
Provide aggregated, high-impact actions mapped to relevant categories or teams (e.g., Operations, Training, Product, Engineering, Management, Support, Sales, or any inferred functions).

Confidence Scoring:
Assign numeric confidence values (0–1) based on frequency and consistency across calls.

Priority Distribution:
- High priority: Only 1-2 most critical issues that affect multiple calls and have severe impact
- Medium priority: 3-5 issues with significant but manageable impact
- Low priority: Remaining issues that are minor or have limited impact

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY)
═══════════════════════════════════════════════════════════════

Output must be a single valid JSON object — no prose, no comments, no Markdown.

All keys must be present even if values are null.

Use the following schema:

{
"count": {TOTAL_CALLS},
"model": "call-analysis-aggregator",
"source": "multi-call-insight-aggregate",
"usage": {
"prompt_tokens": null,
"completion_tokens": null,
"total_tokens": null
},
"aggregatedMetrics": {
"overall_resolution_rate": {
"current": "X%",
"target": "Y%",
"potential_uplift": "+Z%",
"impact_priority": "high | medium | low"
},
"customer_satisfaction_trend": {
"current": "X%",
"target": "Y%",
"potential_uplift": "+Z%",
"impact_priority": "high | medium | low"
},
"average_handle_time": {
"current": "X min",
"target": "Y min",
"potential_change": "-Z min",
"impact_priority": "high | medium | low"
},
"drop_off_or_failure_rate": {
"current": "X%",
"target": "Y%",
"potential_change": "-Z%",
"impact_priority": "medium | low"
},
"clarity_and_information_accuracy_score": {
"current": "X/100",
"target": "Y/100",
"potential_uplift": "+Z",
"impact_priority": "medium | low"
}
},
"cross_session_insights": {
"summary": [
"• Concise bullet point summarizing key systemic pattern or trend",
"• Specific business detail or process issue identified",
"• Another key insight with concrete example from the calls",
"• Final summary point with actionable focus"
],
"issueClusters": [
{
"clusterId": "IC01",
"canonicalName": "Specific issue name referencing actual business context (e.g., [Product Name] Onboarding Confusion)",
"description": "Concise explanation (2-3 sentences) of the recurring issue, referencing specific business elements (products, processes, tools) mentioned in the calls, and its impact.",
"journeySteps": [
{
"stepName": "Specific interaction stage with business context (e.g., [Product] Setup Phase, [Service] Qualification)",
"stepInsights": [
"Concise insight referencing specific business elements",
"Another brief insight with concrete detail",
"Final insight point"
]
}
],
"representativeIssues": [
"Specific issue example with business context (e.g., 'Agents struggle to explain [Product Feature] pricing during [Process Step]')",
"Another concrete example from actual calls"
],
"aggregatedImpact": "high | medium | low",
"frequency": "Number of calls in which this issue appeared",
"supportingEvidence": [
{
"callId": "unique_call_identifier_1",
"evidenceSummary": "Brief summary or excerpt showing the issue in that call."
},
{
"callId": "unique_call_identifier_2",
"evidenceSummary": "Another concrete example from a different call."
}
],
"aggregatedRecommendations": [
{
"action": "Specific, actionable recommendation referencing business context (e.g., 'Update [Product Name] onboarding script to clarify [Specific Feature] at [Specific Step]')",
"rationale": "Brief explanation (1-2 sentences) of how this addresses the issue with reference to specific business elements.",
"priority": "high | medium | low"
}
],
"rationaleSummary": "Concise explanation (1-2 sentences) of why this issue recurs, referencing specific business processes, tools, or systems mentioned in calls.",
"confidenceScore": "0.0–1.0"
}
]
}
}

═══════════════════════════════════════════════════════════════
GENERATION RULES
═══════════════════════════════════════════════════════════════

Output must be strictly valid JSON — no commentary or additional text.

Include 3–7 issueClusters, each with concise but specific details. Prioritize clusters that reference actual business context from the calls.

Each cluster must include:

frequency

aggregatedImpact

confidenceScore

At least 2 supportingEvidence items

At least 1 aggregatedRecommendation

cross_session_insights.summary must be a bullet list (array) of 4-6 concise points that reference specific business details, products, processes, or systems mentioned in the calls. Avoid generic statements.

aggregatedMetrics should reflect general performance metrics, even if expressed with placeholders.

Confidence scores should reflect how consistently issues appeared across calls.

Priority Distribution Rules:
- High priority: Maximum 1-2 clusters only. These must be the most critical issues affecting multiple calls with severe business impact.
- Medium priority: 3-5 clusters with significant but manageable impact.
- Low priority: Remaining clusters that are minor or have limited impact.

Content Requirements:
- Reference specific business names, products, services, processes, tools, or systems mentioned in the call reports
- Include concrete examples and details that make the analysis unique to this business
- Avoid generic statements that could apply to any business
- Keep all descriptions concise (2-3 sentences maximum per field)
- Use bullet points where appropriate for readability

Maintain an executive, analytical, neutral tone."""


def query_llm(client, provider, aggregated_content, model, max_tokens=4096):
    """Send aggregated content to LLM and get response."""
    try:
        # Replace {{input}} placeholder with aggregated content
        user_message = USER_MESSAGE_TEMPLATE.replace("{{input}}", aggregated_content)
        
        if provider == 'openai':
            # OpenAI API call
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content if response.choices else None
        
        elif provider == 'anthropic':
            # Anthropic API call
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_MESSAGE,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            
            # Extract text from Anthropic response
            if message.content and len(message.content) > 0:
                analysis_text = ""
                for content_block in message.content:
                    if hasattr(content_block, 'text'):
                        analysis_text += content_block.text
                return analysis_text
            return None
        
        elif provider == 'google':
            # Google Gemini API call (Vertex AI or AI Studio)
            google_provider = client.get('provider', 'vertex-ai')
            
            # Normalize model name (remove 'vertex-' prefix if present, handle gemini versions)
            model_name = model.lower()
            if model_name.startswith('vertex-'):
                model_name = model_name.replace('vertex-', '')
            
            # Handle various Gemini model name formats
            # Support: gemini-3-pro, gemini-1.5-pro, gemini-pro, gemini-2.0-flash, etc.
            if not model_name.startswith('gemini-'):
                if 'gemini' in model_name:
                    # Extract version if present (e.g., "gemini3pro" -> "gemini-3-pro")
                    match = re.search(r'gemini[-\s]?(\d+)[-\s]?pro', model_name)
                    if match:
                        version = match.group(1)
                        model_name = f"gemini-{version}-pro"
                    else:
                        model_name = 'gemini-1.5-pro'  # Default
                else:
                    model_name = 'gemini-1.5-pro'  # Default
            
            if google_provider == 'google-ai-studio':
                # Google AI Studio API call
                # Ensure API key is configured (in case it wasn't set during client creation)
                api_key = client.get('api_key')
                if api_key:
                    genai.configure(api_key=api_key)
                
                # Combine system message and user message for Gemini
                full_prompt = f"{SYSTEM_MESSAGE}\n\n{user_message}"
                
                # Create the model instance
                # For AI Studio, use the model name directly (e.g., "gemini-2.5-pro")
                gemini_model = genai.GenerativeModel(model_name)
                
                # Generate content
                response = gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                    )
                )
                
                # Extract text from Gemini response
                if response and response.text:
                    return response.text
                return None
            else:
                # Vertex AI API call
                # Create the model instance
                gemini_model = GenerativeModel(model_name)
                
                # Combine system message and user message for Gemini
                # Gemini doesn't have separate system messages, so we combine them
                full_prompt = f"{SYSTEM_MESSAGE}\n\n{user_message}"
                
                # Generate content
                response = gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": 0.7,
                    }
                )
                
                # Extract text from Gemini response
                if response and response.text:
                    return response.text
                return None
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        print(f"Error querying LLM: {str(e)}", file=sys.stderr)
        return None


def save_response(file_path, response):
    """Save LLM response to a new file with 'response-' prefix."""
    file_path_obj = Path(file_path)
    response_path = file_path_obj.parent / f"response-{file_path_obj.name}"
    
    try:
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Response saved to: {response_path}")
        return True
    except Exception as e:
        print(f"Error saving response to {response_path}: {str(e)}", file=sys.stderr)
        return False


def read_insights_files(insights_files):
    """Read all insights files and aggregate their content."""
    aggregated_content = []
    file_count = 0
    
    for insights_file in insights_files:
        try:
            with open(insights_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Add separator and file identifier
                    aggregated_content.append(f"\n\n{'='*60}\n")
                    aggregated_content.append(f"CALL REPORT: {insights_file.name}\n")
                    aggregated_content.append(f"{'='*60}\n\n")
                    aggregated_content.append(content)
                    file_count += 1
        except Exception as e:
            print(f"Warning: Error reading {insights_file}: {str(e)}", file=sys.stderr)
    
    return "\n".join(aggregated_content), file_count


def main():
    parser = argparse.ArgumentParser(
        description='Query an LLM with insights file content and save responses'
    )
    parser.add_argument(
        'directory',
        help='Directory to process (will recursively find all insights-*.txt files)'
    )
    parser.add_argument(
        'model',
        help='LLM model to use (e.g., gpt-4, gpt-4o, claude-sonnet-4-20250514, gemini-3-pro, o1-preview). For Gemini models, set GOOGLE_AI_PROVIDER=studio for AI Studio or GOOGLE_AI_PROVIDER=vertex for Vertex AI (default: vertex)'
    )
    parser.add_argument(
        '--api-key',
        help='API key (will auto-detect provider from model, or set OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_AI_STUDIO_API_KEY/GOOGLE_VERTEX_API_KEY environment variable)',
        default=None
    )
    parser.add_argument(
        '--project-id',
        help='Google Cloud project ID (required for Vertex AI, or set GOOGLE_CLOUD_PROJECT_ID environment variable)',
        default=None
    )
    parser.add_argument(
        '--location',
        help='Google Cloud location/region (for Vertex AI, default: us-central1, or set GOOGLE_CLOUD_LOCATION)',
        default=None
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum number of tokens in the response (default: 4096)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Overwrite response file even if it already exists'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    directory_path = Path(args.directory)
    if not directory_path.exists():
        print(f"Error: Directory '{args.directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not directory_path.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    # Find all insights-*.txt files
    insights_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == '.txt' and file.startswith('insights-'):
                insights_files.append(file_path)
    
    if not insights_files:
        print(f"No 'insights-*.txt' files found in '{args.directory}'")
        sys.exit(0)
    
    print(f"Found {len(insights_files)} insights file(s) to aggregate.")
    
    # Read and aggregate all insights files
    print("Reading insights files...")
    aggregated_content, file_count = read_insights_files(insights_files)
    
    if file_count == 0:
        print("No valid insights content found to process.")
        sys.exit(0)
    
    print(f"Aggregated content from {file_count} file(s) ({len(aggregated_content)} characters)")
    
    # Detect provider from model name
    provider = detect_provider(args.model)
    print(f"Detected provider: {provider} for model: {args.model}")
    
    # Get API key/credentials for the detected provider
    try:
        api_key, project_id, location = get_api_key(
            provider, 
            args.api_key, 
            args.project_id, 
            args.location
        )
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Create the appropriate client
    try:
        if provider == 'google':
            client = create_client(provider, api_key, project_id, location)
        else:
            client = create_client(provider, api_key)
    except Exception as e:
        print(f"Error creating {provider} client: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Show which Google provider is being used
    if provider == 'google':
        google_provider = get_google_provider()
        print(f"Using {google_provider.replace('-', ' ').title()}")
    
    print(f"Using {provider.capitalize()} model: {args.model}")
    print("Sending aggregated content to LLM for analysis...\n")
    
    # Query LLM with aggregated content
    response = query_llm(client, provider, aggregated_content, args.model, args.max_tokens)
    
    if response:
        # Save response to a single aggregated report file
        output_file = directory_path / "response-aggregated-report.json"
        
        # Check if file exists and skip if requested
        if not args.no_skip_existing and output_file.exists():
            print(f"Skipping (response file exists): {output_file}")
            print("Use --no-skip-existing to overwrite.")
            sys.exit(0)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Response saved to: {output_file}")
            print(f"Response length: {len(response)} characters")
            print(f"\nSummary:")
            print(f"  Files aggregated: {file_count}")
            print(f"  Output file: {output_file}")
            sys.exit(0)
        except Exception as e:
            print(f"Error saving response to {output_file}: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: No response received from LLM.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

