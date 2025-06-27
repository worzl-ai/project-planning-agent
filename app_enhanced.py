"""
Project Planning Agent with BRD Generation - Enhanced Implementation
Integrates BRD generation capabilities with project planning functions

Capabilities:
- Project planning and roadmap creation
- Task breakdown and management  
- BRD generation using the worzl BRDA system prompt
- Resource allocation planning
- Agent workflow coordination
"""

import os
import logging
from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with modern configuration
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key'),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

CORS(app, supports_credentials=True)

# BRD System Prompt (from worzl BRDA Agent)
BRD_SYSTEM_PROMPT = """
You are an expert Business Requirements Document (BRD) specialist. Your purpose is to meticulously gather, analyse, and articulate business requirements into a clear, concise, and comprehensive Markdown document.

Goals:
- Produce well-structured, professional, and actionable BRDs.
- Ensure all critical business needs and functional/non-functional requirements are clearly documented.
- Facilitate understanding and alignment among business stakeholders and technical teams.
- Maintain a formal, objective, and precise tone throughout the document.

Overall Direction:
- Always generate the BRD in Markdown format.
- Prioritise clarity, conciseness, and completeness.
- If initial user input is insufficient to complete any section, ask specific clarifying questions to gather the necessary information.
- Do not make assumptions about technical implementation details unless explicitly requested and justified by business requirements.
- Focus on "what" the business needs, not "how" it will be built (unless the "how" directly impacts a business requirement).

BRD Structure:
# Business Requirements Document: [PROJECT_NAME/TITLE]

## 1. Introduction
* **Purpose:** Briefly state the purpose of this document
* **Project Overview:** Provide a high-level summary of the project

## 2. Business Objectives
* Clearly articulate the measurable business goals this project is intended to achieve.
* Use SMART (Specific, Measurable, Achievable, Relevant, Time-bound) criteria where possible.

## 3. Scope
* **In-Scope:** Detail the functionalities, features, processes, and systems that are part of this project
* **Out-of-Scope:** Clearly list what is not part of this project

## 4. Proposed Solution Overview
* Provide a high-level description of the envisioned solution from a business perspective
* Explain how the solution addresses the stated business objectives

### 4.1. Key Features/Functionality
* List and describe the core capabilities the system will provide

### 4.2. Non-Functional Requirements
* **Performance:** Requirements related to speed, response time, scalability, and capacity
* **Security:** Requirements for data protection, access control, authentication, and compliance
* **Scalability:** How the system will handle increased user load, data volume, or new client onboarding
* **Usability:** Requirements for user experience, ease of learning, and accessibility
* **Cost Efficiency:** Any specific requirements related to minimizing setup or ongoing operational costs
* **Availability/Reliability:** Uptime, disaster recovery, mean time to repair (MTTR)
* **Compliance/Regulatory:** Any legal, industry, or internal policy compliance needs

## 5. Stakeholders
* Identify all key individuals, groups, or organizations who have an interest in or will be affected by the project

## 6. Assumptions & Constraints
* **Assumptions:** Factors believed to be true for the project to succeed but are not certain
* **Constraints:** Limitations or restrictions impacting the project

## 7. Key Success Factors
* Define what will determine the successful outcome of the project from a business perspective

## 8. Future Considerations
* List any known features, enhancements, or phases that are explicitly out-of-scope for the current project
"""

@app.route('/')
def index():
    """Project Planning Agent dashboard with BRD capabilities."""
    return jsonify({
        'agent': 'project-planning-agent',
        'status': 'operational',
        'version': '2.0.0',
        'capabilities': [
            'project_planning',
            'roadmap_creation',
            'task_breakdown',
            'brd_generation',
            'resource_allocation',
            'workflow_coordination'
        ],
        'integrations': {
            'brd_agent': True,
            'agent_coordinator': True,
            'foundation_agents': ['technical-seo', 'research-content']
        }
    })

@app.route('/api/brd/generate', methods=['POST'])
def generate_brd():
    """Generate a Business Requirements Document using the BRD Agent system prompt."""
    data = request.get_json()
    
    # Extract project information
    project_name = data.get('project_name', 'Untitled Project')
    project_description = data.get('project_description', '')
    business_objectives = data.get('business_objectives', [])
    stakeholders = data.get('stakeholders', [])
    constraints = data.get('constraints', [])
    
    # Generate BRD using the system prompt logic
    brd_content = generate_brd_content(
        project_name=project_name,
        project_description=project_description,
        business_objectives=business_objectives,
        stakeholders=stakeholders,
        constraints=constraints,
        additional_info=data
    )
    
    # Create project record
    project_id = str(uuid.uuid4())
    project_record = {
        'project_id': project_id,
        'project_name': project_name,
        'created_at': datetime.utcnow().isoformat(),
        'status': 'brd_generated',
        'brd_content': brd_content
    }
    
    return jsonify({
        'status': 'success',
        'project_id': project_id,
        'brd_content': brd_content,
        'project_record': project_record,
        'next_steps': [
            'Review and refine BRD',
            'Get stakeholder approval',
            'Create project plan',
            'Break down tasks',
            'Assign resources'
        ]
    })

@app.route('/api/project/plan', methods=['POST'])
def create_project_plan():
    """Create a detailed project plan from a BRD."""
    data = request.get_json()
    
    project_id = data.get('project_id')
    brd_content = data.get('brd_content', '')
    
    # Generate project plan
    project_plan = generate_project_plan_from_brd(brd_content, data)
    
    return jsonify({
        'status': 'success',
        'project_id': project_id,
        'project_plan': project_plan,
        'timeline': project_plan.get('timeline', {}),
        'milestones': project_plan.get('milestones', []),
        'resource_requirements': project_plan.get('resources', {})
    })

@app.route('/api/workflow/coordinate', methods=['POST'])
def coordinate_workflow():
    """Coordinate workflow between foundation agents."""
    data = request.get_json()
    
    workflow_type = data.get('workflow_type')
    agents_required = data.get('agents_required', [])
    project_context = data.get('project_context', {})
    
    # Create agent coordination plan
    coordination_plan = create_agent_coordination_plan(
        workflow_type=workflow_type,
        agents_required=agents_required,
        project_context=project_context
    )
    
    return jsonify({
        'status': 'success',
        'workflow_id': str(uuid.uuid4()),
        'coordination_plan': coordination_plan,
        'agent_assignments': coordination_plan.get('assignments', {}),
        'execution_order': coordination_plan.get('execution_order', [])
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'agent': 'project-planning-agent',
        'version': '2.0.0',
        'architecture': 'modern',
        'features': {
            'oauth': False,  # Will be configured
            'mcp': True,
            'cards': True,
            'a2a': True,
            'brd_integration': True,
            'agent_coordination': True
        },
        'integrations': {
            'brd_agent': 'integrated',
            'foundation_agents': {
                'technical-seo': 'available',
                'research-content': 'available'
            }
        }
    })

def generate_brd_content(project_name, project_description, business_objectives, stakeholders, constraints, additional_info):
    """Generate BRD content using the structured format."""
    
    # Format business objectives
    objectives_text = ""
    for i, obj in enumerate(business_objectives, 1):
        objectives_text += f"{i}. {obj}\n"
    
    # Format stakeholders
    stakeholders_text = ""
    for stakeholder in stakeholders:
        role = stakeholder.get('role', 'Stakeholder')
        name = stakeholder.get('name', 'TBD')
        involvement = stakeholder.get('involvement', 'To be defined')
        stakeholders_text += f"* **{role}**: {name} - {involvement}\n"
    
    # Format constraints
    constraints_text = ""
    for constraint in constraints:
        constraints_text += f"* {constraint}\n"
    
    # Generate comprehensive BRD
    brd_content = f"""# Business Requirements Document: {project_name}

## 1. Introduction

* **Purpose:** This document defines the business requirements for {project_name}.
* **Project Overview:** {project_description}

## 2. Business Objectives

{objectives_text if objectives_text else "* To be defined based on stakeholder input"}

## 3. Scope

### 3.1. In-Scope
* Core project deliverables as defined in project overview
* Integration with existing systems and processes
* User training and documentation
* Testing and quality assurance

### 3.2. Out-of-Scope
* Features not explicitly mentioned in the core requirements
* Integration with third-party systems not specified
* Ongoing maintenance beyond initial deployment

## 4. Proposed Solution Overview

The proposed solution addresses the business objectives by providing a comprehensive system that meets the identified needs while ensuring scalability, security, and usability.

### 4.1. Key Features/Functionality

* **Core Functionality**: Primary features to address business needs
* **User Interface**: Intuitive interface for end users
* **Integration Capabilities**: Seamless integration with existing systems
* **Reporting and Analytics**: Comprehensive reporting capabilities

### 4.2. Non-Functional Requirements

* **Performance**: System must respond within 3 seconds for standard operations
* **Security**: Must comply with industry security standards and data protection requirements
* **Scalability**: System must handle projected user growth over 3 years
* **Usability**: Intuitive interface requiring minimal training
* **Cost Efficiency**: Solution must provide ROI within 18 months
* **Availability/Reliability**: 99.9% uptime with comprehensive backup and recovery
* **Compliance/Regulatory**: Must meet all applicable regulatory requirements

## 5. Stakeholders

{stakeholders_text if stakeholders_text else "* **Project Sponsor**: To be identified\n* **End Users**: To be identified\n* **Technical Team**: To be identified"}

## 6. Assumptions & Constraints

### 6.1. Assumptions
* Stakeholder availability for requirements gathering and testing
* Existing infrastructure can support the new solution
* Required resources will be available when needed

### 6.2. Constraints
{constraints_text if constraints_text else "* Budget constraints to be defined\n* Timeline constraints to be defined\n* Technical constraints to be identified"}

## 7. Key Success Factors

* Successful delivery within agreed timeline and budget
* User adoption rate of 80% within 3 months of deployment
* Achievement of defined business objectives
* Stakeholder satisfaction with solution performance

## 8. Future Considerations

* Potential integration with additional systems
* Enhancement features based on user feedback
* Scalability improvements for future growth
* Advanced analytics and reporting capabilities

---

**Document Version**: 1.0  
**Created**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Status**: Draft - Pending Review  
"""
    
    return brd_content

def generate_project_plan_from_brd(brd_content, additional_data):
    """Generate a detailed project plan from BRD content."""
    
    project_plan = {
        'phases': [
            {
                'phase': 'Planning & Analysis',
                'duration': '2-4 weeks',
                'tasks': [
                    'Stakeholder interviews',
                    'Requirements refinement',
                    'Technical architecture design',
                    'Resource allocation planning'
                ]
            },
            {
                'phase': 'Design & Development',
                'duration': '6-12 weeks',
                'tasks': [
                    'System design',
                    'Development sprint planning',
                    'Core functionality development',
                    'Integration development'
                ]
            },
            {
                'phase': 'Testing & Deployment',
                'duration': '2-4 weeks',
                'tasks': [
                    'Unit testing',
                    'Integration testing',
                    'User acceptance testing',
                    'Production deployment'
                ]
            },
            {
                'phase': 'Training & Handover',
                'duration': '1-2 weeks',
                'tasks': [
                    'User training',
                    'Documentation handover',
                    'Support processes setup',
                    'Go-live support'
                ]
            }
        ],
        'timeline': {
            'estimated_duration': '11-22 weeks',
            'start_date': 'TBD',
            'end_date': 'TBD'
        },
        'milestones': [
            'BRD Approval',
            'Technical Design Complete',
            'Development Complete',
            'Testing Complete',
            'Go-Live'
        ],
        'resources': {
            'project_manager': 1,
            'business_analyst': 1,
            'developers': '2-4',
            'testers': '1-2',
            'stakeholders': 'Various'
        }
    }
    
    return project_plan

def create_agent_coordination_plan(workflow_type, agents_required, project_context):
    """Create a coordination plan for multi-agent workflows."""
    
    coordination_plan = {
        'workflow_type': workflow_type,
        'coordination_strategy': 'sequential_with_dependencies',
        'assignments': {},
        'execution_order': [],
        'dependencies': {},
        'communication_protocol': 'a2a_messaging'
    }
    
    # Define agent coordination based on workflow type
    if workflow_type == 'content_creation_with_seo':
        coordination_plan['execution_order'] = [
            'research-content-agent',
            'technical-seo-agent',
            'project-planning-agent'
        ]
        coordination_plan['assignments'] = {
            'research-content-agent': 'Market research and content generation',
            'technical-seo-agent': 'SEO optimization and technical recommendations',
            'project-planning-agent': 'Project coordination and quality assurance'
        }
        coordination_plan['dependencies'] = {
            'technical-seo-agent': ['research-content-agent'],
            'project-planning-agent': ['research-content-agent', 'technical-seo-agent']
        }
    
    elif workflow_type == 'project_planning_with_brd':
        coordination_plan['execution_order'] = [
            'project-planning-agent',
            'research-content-agent',
            'technical-seo-agent'
        ]
        coordination_plan['assignments'] = {
            'project-planning-agent': 'BRD generation and project planning',
            'research-content-agent': 'Market research and content requirements',
            'technical-seo-agent': 'Technical requirements and SEO considerations'
        }
        coordination_plan['dependencies'] = {
            'research-content-agent': ['project-planning-agent'],
            'technical-seo-agent': ['project-planning-agent']
        }
    
    return coordination_plan

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"🚀 Starting Enhanced Project Planning Agent with BRD on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
