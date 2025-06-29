# Project Planning Agent Implementation

## Overview

The Project Planning Agent serves as a strategic orchestrator within the A2A agent ecosystem, specializing in project management, feature definition, roadmap creation, and task breakdown. This agent leverages advanced planning methodologies, stakeholder analysis, and Google Cloud's infrastructure to deliver comprehensive project plans that align business objectives with technical implementation strategies.

## Agent Architecture

### Core Agent Class Implementation

```python
import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import networkx as nx
from collections import defaultdict

from google.cloud import aiplatform
from google.cloud import pubsub_v1
from google.cloud import storage
from google.cloud import firestore
from google.cloud import scheduler_v1
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel

# Import our A2A communication framework
from a2a_communication import AgentCommunicationManager, AgentRegistry

class ProjectType(Enum):
    SAAS_DEVELOPMENT = "saas_development"
    WEBSITE_BUILD = "website_build"
    MARKETING_CAMPAIGN = "marketing_campaign"
    CONTENT_STRATEGY = "content_strategy"
    SEO_OPTIMIZATION = "seo_optimization"
    AGENT_DEVELOPMENT = "agent_development"
    INTEGRATION_PROJECT = "integration_project"

class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ProjectPhase(Enum):
    DISCOVERY = "discovery"
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"

@dataclass
class Task:
    """Individual task within a project"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    estimated_hours: float
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    due_date: Optional[str] = None
    completion_date: Optional[str] = None
    progress_percentage: float = 0.0
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class ProjectMilestone:
    """Project milestone definition"""
    milestone_id: str
    title: str
    description: str
    target_date: str
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.NOT_STARTED

@dataclass
class ProjectPlan:
    """Comprehensive project plan"""
    project_id: str
    title: str
    description: str
    project_type: ProjectType
    objectives: List[str]
    success_metrics: List[str]
    stakeholders: List[str]
    budget_estimate: Optional[float] = None
    timeline_weeks: Optional[int] = None
    phases: List[ProjectPhase] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    milestones: List[ProjectMilestone] = field(default_factory=list)
    risks: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class PlanningRequest:
    """Request for project planning services"""
    request_type: str  # "new_project", "update_project", "task_breakdown", "roadmap_creation"
    project_context: Dict[str, Any]
    requirements: List[str]
    constraints: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    timeline_requirements: Optional[Dict[str, Any]] = None
    budget_constraints: Optional[Dict[str, Any]] = None
    priority_factors: List[str] = field(default_factory=list)

class ProjectPlanningAgent:
    """
    Advanced Project Planning Agent with comprehensive planning capabilities
    """
    
    def __init__(self, project_id: str, agent_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.agent_id = agent_id
        self.location = location
        
        # Initialize Google Cloud services
        self.storage_client = storage.Client(project=project_id)
        self.firestore_client = firestore.Client(project=project_id)
        self.scheduler_client = scheduler_v1.CloudSchedulerClient()
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Initialize A2A communication
        self.comm_manager = AgentCommunicationManager(project_id)
        self.agent_registry = AgentRegistry(self.firestore_client)
        
        # Model configurations with token allocations
        self.model_configs = {
            "strategic_planning": {
                "model_name": "gemini-1.5-pro",
                "max_tokens": 8192,
                "temperature": 0.3,
                "top_p": 0.8,
                "addon_tokens": 2048
            },
            "task_breakdown": {
                "model_name": "gemini-1.5-pro",
                "max_tokens": 8192,
                "temperature": 0.4,
                "top_p": 0.9,
                "addon_tokens": 2048
            },
            "risk_analysis": {
                "model_name": "gemini-1.5-flash",
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.8,
                "addon_tokens": 1024
            }
        }
        
        # Planning tools and methodologies
        self.tools = self._initialize_planning_tools()
        
        # Project templates and best practices
        self.project_templates = self._load_project_templates()
        
        # Register agent
        self._register_agent()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ProjectPlanningAgent-{agent_id}")
        
    def _initialize_planning_tools(self) -> Dict[str, Any]:
        """Initialize planning tools and methodologies"""
        return {
            "requirement_analyzer": RequirementAnalyzer(self.project_id),
            "task_estimator": TaskEstimator(self.project_id),
            "dependency_mapper": DependencyMapper(self.project_id),
            "risk_assessor": RiskAssessor(self.project_id),
            "resource_planner": ResourcePlanner(self.project_id),
            "timeline_optimizer": TimelineOptimizer(self.project_id),
            "stakeholder_analyzer": StakeholderAnalyzer(self.project_id),
            "milestone_generator": MilestoneGenerator(self.project_id)
        }
    
    def _load_project_templates(self) -> Dict[str, Any]:
        """Load project templates for different project types"""
        return {
            ProjectType.SAAS_DEVELOPMENT: {
                "phases": [
                    ProjectPhase.DISCOVERY,
                    ProjectPhase.PLANNING,
                    ProjectPhase.DESIGN,
                    ProjectPhase.DEVELOPMENT,
                    ProjectPhase.TESTING,
                    ProjectPhase.DEPLOYMENT
                ],
                "typical_duration_weeks": 16,
                "key_milestones": [
                    "Requirements Finalized",
                    "Technical Architecture Approved",
                    "MVP Development Complete",
                    "Beta Testing Complete",
                    "Production Launch"
                ],
                "common_risks": [
                    "Scope creep",
                    "Technical complexity underestimation",
                    "Integration challenges",
                    "Performance issues"
                ]
            },
            ProjectType.WEBSITE_BUILD: {
                "phases": [
                    ProjectPhase.DISCOVERY,
                    ProjectPhase.PLANNING,
                    ProjectPhase.DESIGN,
                    ProjectPhase.DEVELOPMENT,
                    ProjectPhase.TESTING,
                    ProjectPhase.DEPLOYMENT
                ],
                "typical_duration_weeks": 8,
                "key_milestones": [
                    "Content Strategy Approved",
                    "Design Mockups Approved",
                    "Development Complete",
                    "SEO Optimization Complete",
                    "Site Launch"
                ],
                "common_risks": [
                    "Content delays",
                    "Design approval bottlenecks",
                    "SEO requirements changes",
                    "Browser compatibility issues"
                ]
            },
            ProjectType.AGENT_DEVELOPMENT: {
                "phases": [
                    ProjectPhase.DISCOVERY,
                    ProjectPhase.PLANNING,
                    ProjectPhase.DESIGN,
                    ProjectPhase.DEVELOPMENT,
                    ProjectPhase.TESTING,
                    ProjectPhase.DEPLOYMENT
                ],
                "typical_duration_weeks": 12,
                "key_milestones": [
                    "Agent Specifications Defined",
                    "Core Functionality Implemented",
                    "A2A Integration Complete",
                    "Testing and Validation Complete",
                    "Production Deployment"
                ],
                "common_risks": [
                    "Model performance issues",
                    "Integration complexity",
                    "Prompt engineering challenges",
                    "Scalability concerns"
                ]
            }
        }
    
    def _register_agent(self):
        """Register this agent in the agent registry"""
        agent_info = {
            "agent_id": self.agent_id,
            "agent_type": "project_planning",
            "capabilities": [
                "project_planning",
                "task_breakdown",
                "roadmap_creation",
                "risk_assessment",
                "resource_planning",
                "timeline_optimization",
                "stakeholder_analysis",
                "milestone_planning"
            ],
            "input_topics": [
                f"agent_{self.agent_id}_requests",
                "planning_requests",
                "project_updates"
            ],
            "output_topics": [
                f"agent_{self.agent_id}_responses",
                "project_plans",
                "planning_recommendations"
            ],
            "metadata": {
                "version": "1.0.0",
                "supported_project_types": [pt.value for pt in ProjectType],
                "planning_methodologies": ["agile", "waterfall", "hybrid"]
            }
        }
        
        self.agent_registry.register_agent(agent_info)
        self.logger.info(f"Project Planning Agent {self.agent_id} registered successfully")

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Main message processing method for A2A communication"""
        start_time = datetime.utcnow()
        
        try:
            message_type = message.get("type")
            payload = message.get("payload", {})
            
            if message_type == "planning_request":
                result = await self.handle_planning_request(payload)
            elif message_type == "task_breakdown_request":
                result = await self.handle_task_breakdown_request(payload)
            elif message_type == "roadmap_request":
                result = await self.handle_roadmap_request(payload)
            elif message_type == "risk_assessment_request":
                result = await self.handle_risk_assessment_request(payload)
            elif message_type == "project_update":
                result = await self.handle_project_update(payload)
            else:
                raise ValueError(f"Unknown message type: {message_type}")
                
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "data": result,
                "metadata": {
                    "agent_id": self.agent_id,
                    "processing_time": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Error processing message: {str(e)}")
            
            return {
                "success": False,
                "data": {},
                "metadata": {
                    "agent_id": self.agent_id,
                    "processing_time": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "errors": [str(e)]
            }

    async def handle_planning_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive project planning requests"""
        planning_request = PlanningRequest(**payload)
        
        self.logger.info(f"Processing planning request: {planning_request.request_type}")
        
        if planning_request.request_type == "new_project":
            return await self._create_new_project_plan(planning_request)
        elif planning_request.request_type == "update_project":
            return await self._update_existing_project(planning_request)
        elif planning_request.request_type == "task_breakdown":
            return await self._perform_task_breakdown(planning_request)
        elif planning_request.request_type == "roadmap_creation":
            return await self._create_project_roadmap(planning_request)
        else:
            raise ValueError(f"Unsupported planning request type: {planning_request.request_type}")

    async def _create_new_project_plan(self, request: PlanningRequest) -> Dict[str, Any]:
        """Create a comprehensive new project plan"""
        
        # 1. Analyze project context and requirements
        context_analysis = await self._analyze_project_context(request.project_context)
        
        # 2. Determine project type and select appropriate template
        project_type = await self._determine_project_type(request.project_context, request.requirements)
        template = self.project_templates.get(project_type, {})
        
        # 3. Analyze requirements and stakeholders
        requirement_analyzer = self.tools["requirement_analyzer"]
        requirements_analysis = await requirement_analyzer.analyze_requirements(
            requirements=request.requirements,
            context=request.project_context,
            stakeholders=request.stakeholders
        )
        
        # 4. Perform stakeholder analysis
        stakeholder_analyzer = self.tools["stakeholder_analyzer"]
        stakeholder_analysis = await stakeholder_analyzer.analyze_stakeholders(
            stakeholders=request.stakeholders,
            project_context=request.project_context
        )
        
        # 5. Generate initial task breakdown
        task_breakdown = await self._generate_comprehensive_task_breakdown(
            project_type=project_type,
            requirements=requirements_analysis,
            template=template
        )
        
        # 6. Analyze dependencies and create dependency graph
        dependency_mapper = self.tools["dependency_mapper"]
        dependency_analysis = await dependency_mapper.map_dependencies(task_breakdown["tasks"])
        
        # 7. Estimate effort and timeline
        task_estimator = self.tools["task_estimator"]
        effort_estimates = await task_estimator.estimate_tasks(
            tasks=task_breakdown["tasks"],
            project_context=request.project_context,
            historical_data={}  # Would include historical project data
        )
        
        # 8. Optimize timeline
        timeline_optimizer = self.tools["timeline_optimizer"]
        optimized_timeline = await timeline_optimizer.optimize_timeline(
            tasks=task_breakdown["tasks"],
            dependencies=dependency_analysis,
            constraints=request.constraints,
            timeline_requirements=request.timeline_requirements
        )
        
        # 9. Assess risks
        risk_assessor = self.tools["risk_assessor"]
        risk_assessment = await risk_assessor.assess_project_risks(
            project_type=project_type,
            tasks=task_breakdown["tasks"],
            stakeholders=stakeholder_analysis,
            constraints=request.constraints
        )
        
        # 10. Generate milestones
        milestone_generator = self.tools["milestone_generator"]
        milestones = await milestone_generator.generate_milestones(
            project_type=project_type,
            tasks=task_breakdown["tasks"],
            timeline=optimized_timeline
        )
        
        # 11. Create comprehensive project plan
        project_plan = ProjectPlan(
            project_id=str(uuid.uuid4()),
            title=request.project_context.get("title", "New Project"),
            description=request.project_context.get("description", ""),
            project_type=project_type,
            objectives=requirements_analysis.get("objectives", []),
            success_metrics=requirements_analysis.get("success_metrics", []),
            stakeholders=request.stakeholders,
            budget_estimate=request.budget_constraints.get("total_budget") if request.budget_constraints else None,
            timeline_weeks=optimized_timeline.get("total_weeks"),
            phases=template.get("phases", []),
            tasks=[Task(**task) for task in task_breakdown["tasks"]],
            milestones=[ProjectMilestone(**milestone) for milestone in milestones],
            risks=risk_assessment.get("risks", []),
            assumptions=requirements_analysis.get("assumptions", []),
            constraints=request.constraints
        )
        
        # 12. Generate strategic recommendations
        strategic_recommendations = await self._generate_strategic_recommendations(
            project_plan=project_plan,
            context_analysis=context_analysis,
            stakeholder_analysis=stakeholder_analysis
        )
        
        # 13. Save project plan to Firestore
        await self._save_project_plan(project_plan)
        
        return {
            "project_plan": asdict(project_plan),
            "context_analysis": context_analysis,
            "requirements_analysis": requirements_analysis,
            "stakeholder_analysis": stakeholder_analysis,
            "dependency_analysis": dependency_analysis,
            "effort_estimates": effort_estimates,
            "timeline_optimization": optimized_timeline,
            "risk_assessment": risk_assessment,
            "strategic_recommendations": strategic_recommendations,
            "planning_metadata": {
                "planning_methodology": "hybrid_agile",
                "confidence_level": "high",
                "last_updated": datetime.utcnow().isoformat()
            }
        }

    async def _analyze_project_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project context using AI"""
        
        context_prompt = self._build_context_analysis_prompt(context)
        
        model = GenerativeModel(self.model_configs["strategic_planning"]["model_name"])
        context_response = await model.generate_content_async(
            context_prompt,
            generation_config={
                "max_output_tokens": self.model_configs["strategic_planning"]["max_tokens"],
                "temperature": self.model_configs["strategic_planning"]["temperature"]
            }
        )
        
        try:
            context_analysis = json.loads(context_response.text)
        except json.JSONDecodeError:
            # Fallback to structured analysis if JSON parsing fails
            context_analysis = {
                "business_context": context_response.text,
                "complexity_assessment": "medium",
                "strategic_importance": "high",
                "key_challenges": [],
                "success_factors": []
            }
        
        return context_analysis

    async def _determine_project_type(self, context: Dict[str, Any], 
                                    requirements: List[str]) -> ProjectType:
        """Determine the most appropriate project type"""
        
        # Use AI to classify project type based on context and requirements
        classification_prompt = f"""
        Analyze the following project context and requirements to determine the most appropriate project type.

        PROJECT CONTEXT:
        {json.dumps(context, indent=2)}

        REQUIREMENTS:
        {json.dumps(requirements, indent=2)}

        Available project types:
        - saas_development: Building a Software-as-a-Service platform
        - website_build: Creating a website or web application
        - marketing_campaign: Planning and executing marketing initiatives
        - content_strategy: Developing content marketing strategies
        - seo_optimization: Improving search engine optimization
        - agent_development: Building AI agents or automation systems
        - integration_project: Integrating systems or platforms

        Respond with only the project type identifier (e.g., "saas_development").
        """
        
        model = GenerativeModel(self.model_configs["strategic_planning"]["model_name"])
        classification_response = await model.generate_content_async(
            classification_prompt,
            generation_config={
                "max_output_tokens": 50,
                "temperature": 0.1
            }
        )
        
        project_type_str = classification_response.text.strip().lower()
        
        # Map response to enum
        for project_type in ProjectType:
            if project_type.value == project_type_str:
                return project_type
        
        # Default fallback
        return ProjectType.SAAS_DEVELOPMENT

    async def _generate_comprehensive_task_breakdown(self, project_type: ProjectType,
                                                   requirements: Dict[str, Any],
                                                   template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive task breakdown using AI"""
        
        task_breakdown_prompt = self._build_task_breakdown_prompt(
            project_type, requirements, template
        )
        
        model = GenerativeModel(self.model_configs["task_breakdown"]["model_name"])
        breakdown_response = await model.generate_content_async(
            task_breakdown_prompt,
            generation_config={
                "max_output_tokens": self.model_configs["task_breakdown"]["max_tokens"],
                "temperature": self.model_configs["task_breakdown"]["temperature"]
            }
        )
        
        try:
            task_breakdown = json.loads(breakdown_response.text)
        except json.JSONDecodeError:
            # Fallback to basic structure if JSON parsing fails
            task_breakdown = {
                "tasks": [],
                "task_categories": [],
                "estimated_total_hours": 0
            }
        
        # Ensure all tasks have required fields
        for task in task_breakdown.get("tasks", []):
            if "task_id" not in task:
                task["task_id"] = str(uuid.uuid4())
            if "priority" not in task:
                task["priority"] = TaskPriority.MEDIUM.value
            if "status" not in task:
                task["status"] = TaskStatus.NOT_STARTED.value
            if "estimated_hours" not in task:
                task["estimated_hours"] = 8.0
        
        return task_breakdown

    async def _generate_strategic_recommendations(self, project_plan: ProjectPlan,
                                                context_analysis: Dict[str, Any],
                                                stakeholder_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendations for project success"""
        
        recommendations_prompt = self._build_strategic_recommendations_prompt(
            project_plan, context_analysis, stakeholder_analysis
        )
        
        model = GenerativeModel(self.model_configs["strategic_planning"]["model_name"])
        recommendations_response = await model.generate_content_async(
            recommendations_prompt,
            generation_config={
                "max_output_tokens": self.model_configs["strategic_planning"]["max_tokens"],
                "temperature": self.model_configs["strategic_planning"]["temperature"]
            }
        )
        
        try:
            recommendations = json.loads(recommendations_response.text)
        except json.JSONDecodeError:
            recommendations = {
                "strategic_recommendations": recommendations_response.text,
                "success_factors": [],
                "risk_mitigation": [],
                "optimization_opportunities": []
            }
        
        return recommendations

    def _build_context_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for project context analysis"""
        return f"""
        You are an expert project management consultant analyzing a new project context.

        PROJECT CONTEXT:
        {json.dumps(context, indent=2)}

        Please provide a comprehensive analysis in JSON format with the following structure:

        {{
            "business_context": {{
                "industry": "Identified industry or sector",
                "market_position": "Current market position or opportunity",
                "competitive_landscape": "Analysis of competitive environment",
                "business_objectives": ["List of inferred business objectives"]
            }},
            "complexity_assessment": {{
                "overall_complexity": "low|medium|high|very_high",
                "technical_complexity": "Assessment of technical challenges",
                "organizational_complexity": "Assessment of organizational challenges",
                "integration_complexity": "Assessment of integration requirements"
            }},
            "strategic_importance": {{
                "business_impact": "high|medium|low",
                "urgency": "high|medium|low",
                "strategic_alignment": "Assessment of strategic alignment"
            }},
            "key_challenges": [
                "List of anticipated key challenges"
            ],
            "success_factors": [
                "List of critical success factors"
            ],
            "stakeholder_considerations": {{
                "primary_stakeholders": ["List of primary stakeholders"],
                "decision_makers": ["List of key decision makers"],
                "influence_factors": ["Factors that influence stakeholder buy-in"]
            }}
        }}

        Focus on actionable insights that will inform project planning and execution strategy.
        """

    def _build_task_breakdown_prompt(self, project_type: ProjectType,
                                   requirements: Dict[str, Any],
                                   template: Dict[str, Any]) -> str:
        """Build prompt for comprehensive task breakdown"""
        return f"""
        You are an expert project manager creating a detailed task breakdown for a {project_type.value} project.

        PROJECT REQUIREMENTS:
        {json.dumps(requirements, indent=2)}

        PROJECT TEMPLATE GUIDANCE:
        {json.dumps(template, indent=2)}

        Create a comprehensive task breakdown in JSON format with the following structure:

        {{
            "task_categories": [
                {{
                    "category": "Category name",
                    "description": "Category description",
                    "phase": "discovery|planning|design|development|testing|deployment|maintenance"
                }}
            ],
            "tasks": [
                {{
                    "title": "Task title",
                    "description": "Detailed task description",
                    "category": "Task category",
                    "phase": "Project phase",
                    "priority": "critical|high|medium|low",
                    "estimated_hours": 8.0,
                    "dependencies": ["List of task titles this depends on"],
                    "deliverables": ["List of expected deliverables"],
                    "acceptance_criteria": ["List of acceptance criteria"],
                    "assigned_agent": "Suggested agent type or null",
                    "tags": ["List of relevant tags"]
                }}
            ],
            "estimated_total_hours": 0,
            "critical_path_tasks": ["List of critical path task titles"],
            "parallel_execution_opportunities": [
                {{
                    "tasks": ["Tasks that can be executed in parallel"],
                    "description": "Why these tasks can be parallel"
                }}
            ]
        }}

        Ensure tasks are:
        1. Specific and actionable
        2. Properly sequenced with clear dependencies
        3. Appropriately sized (4-40 hours each)
        4. Aligned with project phases
        5. Include clear acceptance criteria
        6. Consider resource constraints and skill requirements

        Focus on creating a realistic and executable plan that accounts for the specific requirements and constraints of this project type.
        """

    def _build_strategic_recommendations_prompt(self, project_plan: ProjectPlan,
                                              context_analysis: Dict[str, Any],
                                              stakeholder_analysis: Dict[str, Any]) -> str:
        """Build prompt for strategic recommendations"""
        return f"""
        You are a senior project management consultant providing strategic recommendations for project success.

        PROJECT PLAN SUMMARY:
        - Project Type: {project_plan.project_type.value}
        - Total Tasks: {len(project_plan.tasks)}
        - Timeline: {project_plan.timeline_weeks} weeks
        - Key Objectives: {project_plan.objectives}

        CONTEXT ANALYSIS:
        {json.dumps(context_analysis, indent=2)}

        STAKEHOLDER ANALYSIS:
        {json.dumps(stakeholder_analysis, indent=2)}

        PROJECT RISKS:
        {json.dumps([asdict(risk) for risk in project_plan.risks], indent=2)}

        Provide strategic recommendations in JSON format:

        {{
            "executive_summary": "High-level summary of key recommendations",
            "success_factors": [
                {{
                    "factor": "Success factor name",
                    "description": "Detailed description",
                    "implementation_approach": "How to implement",
                    "success_metrics": ["How to measure success"]
                }}
            ],
            "risk_mitigation": [
                {{
                    "risk": "Risk description",
                    "mitigation_strategy": "Strategy to mitigate",
                    "contingency_plan": "Backup plan if mitigation fails",
                    "monitoring_approach": "How to monitor this risk"
                }}
            ],
            "optimization_opportunities": [
                {{
                    "opportunity": "Optimization opportunity",
                    "potential_impact": "Expected impact",
                    "implementation_effort": "low|medium|high",
                    "recommended_timing": "When to implement"
                }}
            ],
            "stakeholder_management": {{
                "communication_strategy": "Recommended communication approach",
                "engagement_tactics": ["Specific engagement tactics"],
                "decision_points": ["Key decision points requiring stakeholder input"]
            }},
            "quality_assurance": {{
                "quality_gates": ["Key quality checkpoints"],
                "review_processes": ["Recommended review processes"],
                "testing_strategy": "Overall testing approach"
            }},
            "change_management": {{
                "change_control_process": "Recommended change control approach",
                "scope_management": "Strategy for managing scope changes",
                "communication_protocols": ["Communication protocols for changes"]
            }}
        }}

        Focus on actionable, specific recommendations that address the unique challenges and opportunities of this project.
        """

# Planning Tool Implementations

class RequirementAnalyzer:
    """Advanced requirement analysis and categorization"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def analyze_requirements(self, requirements: List[str], 
                                 context: Dict[str, Any],
                                 stakeholders: List[str]) -> Dict[str, Any]:
        """Analyze and categorize project requirements"""
        
        analysis = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "business_requirements": [],
            "technical_requirements": [],
            "constraints": [],
            "assumptions": [],
            "objectives": [],
            "success_metrics": [],
            "requirement_conflicts": [],
            "missing_requirements": []
        }
        
        # Categorize requirements using AI
        for requirement in requirements:
            category = await self._categorize_requirement(requirement, context)
            analysis[category].append(requirement)
        
        # Identify potential conflicts
        conflicts = await self._identify_requirement_conflicts(requirements)
        analysis["requirement_conflicts"] = conflicts
        
        # Suggest missing requirements
        missing = await self._suggest_missing_requirements(requirements, context)
        analysis["missing_requirements"] = missing
        
        return analysis
    
    async def _categorize_requirement(self, requirement: str, 
                                    context: Dict[str, Any]) -> str:
        """Categorize a single requirement"""
        
        # Use AI to categorize requirement
        # This would be implemented with a classification model or prompt
        
        # Simplified categorization logic for now
        requirement_lower = requirement.lower()
        
        if any(word in requirement_lower for word in ["must", "shall", "will", "should"]):
            if any(word in requirement_lower for word in ["performance", "speed", "scalability", "security"]):
                return "non_functional_requirements"
            elif any(word in requirement_lower for word in ["business", "revenue", "profit", "market"]):
                return "business_requirements"
            elif any(word in requirement_lower for word in ["technical", "system", "database", "api"]):
                return "technical_requirements"
            else:
                return "functional_requirements"
        elif any(word in requirement_lower for word in ["constraint", "limitation", "cannot", "must not"]):
            return "constraints"
        elif any(word in requirement_lower for word in ["assume", "assumption", "expect"]):
            return "assumptions"
        elif any(word in requirement_lower for word in ["goal", "objective", "aim"]):
            return "objectives"
        elif any(word in requirement_lower for word in ["measure", "metric", "kpi", "success"]):
            return "success_metrics"
        else:
            return "functional_requirements"

class TaskEstimator:
    """Advanced task estimation using historical data and AI"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def estimate_tasks(self, tasks: List[Dict[str, Any]], 
                           project_context: Dict[str, Any],
                           historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate effort for project tasks"""
        
        estimates = {
            "task_estimates": [],
            "total_estimated_hours": 0,
            "confidence_levels": {},
            "estimation_methodology": "three_point_estimation",
            "risk_factors": []
        }
        
        for task in tasks:
            task_estimate = await self._estimate_single_task(task, project_context, historical_data)
            estimates["task_estimates"].append(task_estimate)
            estimates["total_estimated_hours"] += task_estimate["estimated_hours"]
        
        # Calculate confidence levels
        estimates["confidence_levels"] = await self._calculate_confidence_levels(estimates["task_estimates"])
        
        return estimates
    
    async def _estimate_single_task(self, task: Dict[str, Any], 
                                  context: Dict[str, Any],
                                  historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate effort for a single task using three-point estimation"""
        
        # Use AI to generate optimistic, most likely, and pessimistic estimates
        estimation_prompt = f"""
        Estimate the effort required for the following task using three-point estimation:

        TASK: {task.get('title', 'Unknown')}
        DESCRIPTION: {task.get('description', 'No description')}
        CATEGORY: {task.get('category', 'General')}
        COMPLEXITY: {task.get('complexity', 'Medium')}

        PROJECT CONTEXT:
        {json.dumps(context, indent=2)}

        Provide estimates in hours for:
        - Optimistic (best case scenario)
        - Most Likely (realistic scenario)
        - Pessimistic (worst case scenario)

        Consider factors like:
        - Task complexity
        - Required skills
        - Dependencies
        - Potential risks
        - Team experience

        Respond in JSON format:
        {{
            "optimistic_hours": 0,
            "most_likely_hours": 0,
            "pessimistic_hours": 0,
            "confidence_level": "high|medium|low",
            "risk_factors": ["List of factors that could affect estimation"],
            "assumptions": ["Key assumptions made in estimation"]
        }}
        """
        
        # For now, use simplified estimation logic
        base_hours = task.get("estimated_hours", 8.0)
        
        return {
            "task_id": task.get("task_id"),
            "task_title": task.get("title"),
            "optimistic_hours": base_hours * 0.7,
            "most_likely_hours": base_hours,
            "pessimistic_hours": base_hours * 1.5,
            "estimated_hours": base_hours,  # PERT formula: (O + 4M + P) / 6
            "confidence_level": "medium",
            "risk_factors": [],
            "assumptions": []
        }

class DependencyMapper:
    """Advanced dependency analysis and critical path calculation"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def map_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map task dependencies and calculate critical path"""
        
        # Create dependency graph
        graph = nx.DiGraph()
        
        # Add tasks as nodes
        for task in tasks:
            graph.add_node(task["task_id"], **task)
        
        # Add dependency edges
        for task in tasks:
            for dependency in task.get("dependencies", []):
                # Find dependency task by title (simplified)
                dep_task = next((t for t in tasks if t["title"] == dependency), None)
                if dep_task:
                    graph.add_edge(dep_task["task_id"], task["task_id"])
        
        # Calculate critical path
        try:
            critical_path = self._calculate_critical_path(graph)
        except:
            critical_path = []
        
        # Identify parallel execution opportunities
        parallel_opportunities = self._identify_parallel_opportunities(graph)
        
        # Check for circular dependencies
        circular_dependencies = list(nx.simple_cycles(graph))
        
        return {
            "dependency_graph": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "is_acyclic": nx.is_directed_acyclic_graph(graph)
            },
            "critical_path": critical_path,
            "parallel_opportunities": parallel_opportunities,
            "circular_dependencies": circular_dependencies,
            "dependency_analysis": {
                "highly_dependent_tasks": self._find_highly_dependent_tasks(graph),
                "bottleneck_tasks": self._find_bottleneck_tasks(graph),
                "independent_tasks": self._find_independent_tasks(graph)
            }
        }
    
    def _calculate_critical_path(self, graph: nx.DiGraph) -> List[str]:
        """Calculate the critical path through the project"""
        
        if not nx.is_directed_acyclic_graph(graph):
            return []
        
        # Simplified critical path calculation
        # In a real implementation, this would use proper CPM algorithm
        try:
            longest_path = nx.dag_longest_path(graph, weight='estimated_hours')
            return longest_path
        except:
            return []
    
    def _identify_parallel_opportunities(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Identify tasks that can be executed in parallel"""
        
        opportunities = []
        
        # Find tasks with no dependencies between them
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                if node1 != node2:
                    # Check if tasks can be parallel (no path between them)
                    if not nx.has_path(graph, node1, node2) and not nx.has_path(graph, node2, node1):
                        # Check if they're in the same phase
                        task1 = graph.nodes[node1]
                        task2 = graph.nodes[node2]
                        
                        if task1.get("phase") == task2.get("phase"):
                            opportunities.append({
                                "task_1": task1.get("title"),
                                "task_2": task2.get("title"),
                                "phase": task1.get("phase"),
                                "potential_time_savings": min(
                                    task1.get("estimated_hours", 0),
                                    task2.get("estimated_hours", 0)
                                )
                            })
        
        return opportunities[:10]  # Limit to top 10 opportunities

class RiskAssessor:
    """Comprehensive project risk assessment"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        
    async def assess_project_risks(self, project_type: ProjectType,
                                 tasks: List[Dict[str, Any]],
                                 stakeholders: Dict[str, Any],
                                 constraints: List[str]) -> Dict[str, Any]:
        """Assess comprehensive project risks"""
        
        risk_assessment = {
            "identified_risks": [],
            "risk_matrix": {},
            "mitigation_strategies": [],
            "contingency_plans": [],
            "risk_monitoring_plan": {}
        }
        
        # Identify risks by category
        technical_risks = await self._assess_technical_risks(tasks, project_type)
        schedule_risks = await self._assess_schedule_risks(tasks)
        resource_risks = await self._assess_resource_risks(tasks, stakeholders)
        business_risks = await self._assess_business_risks(project_type, constraints)
        
        all_risks = technical_risks + schedule_risks + resource_risks + business_risks
        
        # Prioritize risks
        prioritized_risks = await self._prioritize_risks(all_risks)
        
        risk_assessment["identified_risks"] = prioritized_risks
        
        # Generate mitigation strategies
        for risk in prioritized_risks[:10]:  # Top 10 risks
            mitigation = await self._generate_mitigation_strategy(risk)
            risk_assessment["mitigation_strategies"].append(mitigation)
        
        return risk_assessment
    
    async def _assess_technical_risks(self, tasks: List[Dict[str, Any]], 
                                    project_type: ProjectType) -> List[Dict[str, Any]]:
        """Assess technical risks based on tasks and project type"""
        
        risks = []
        
        # Analyze task complexity
        complex_tasks = [t for t in tasks if t.get("estimated_hours", 0) > 40]
        if complex_tasks:
            risks.append({
                "risk_id": str(uuid.uuid4()),
                "category": "technical",
                "title": "High complexity tasks may exceed estimates",
                "description": f"Found {len(complex_tasks)} tasks with high complexity",
                "probability": "medium",
                "impact": "high",
                "risk_score": 6
            })
        
        # Check for integration risks
        integration_tasks = [t for t in tasks if "integration" in t.get("title", "").lower()]
        if integration_tasks:
            risks.append({
                "risk_id": str(uuid.uuid4()),
                "category": "technical",
                "title": "Integration complexity may cause delays",
                "description": f"Found {len(integration_tasks)} integration-related tasks",
                "probability": "medium",
                "impact": "medium",
                "risk_score": 4
            })
        
        return risks
```

This implementation provides a comprehensive foundation for the Project Planning Agent, including detailed project planning capabilities, task breakdown, dependency mapping, risk assessment, and strategic recommendations. The agent integrates seamlessly with the A2A communication framework while providing powerful project management capabilities.

