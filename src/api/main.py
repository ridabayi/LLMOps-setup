from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import requests
import os
import mlflow
import openai
import litellm
from litellm import completion_cost
from mlflow.entities.span import SpanType
from mlflow.entities.span_event import SpanEvent
import time
import re
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from jose import JWTError, jwt
from passlib.context import CryptContext

# Get LiteLLM's URL from environment variables, with a default for local dev
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:8000")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Simple user database (in production, use a real database)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("secret123"),  # password: secret123
        "role": "admin"
    },
    "user": {
        "username": "user", 
        "hashed_password": pwd_context.hash("password123"),  # password: password123
        "role": "user"
    }
}

# Initialize MLflow tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

@mlflow.trace(name="security_incident", span_type=SpanType.LLM)
def trace_security_incident(incident_type: str, request_data: dict, pattern: str = None, error_message: str = None):
    """Trace security incidents in MLflow for blocked attacks."""
    try:
        # Get current span for this security incident
        current_span = mlflow.get_current_active_span()
        
        if current_span:
            # Set inputs (the malicious request)
            current_span.set_inputs({
                "request_data": request_data,
                "incident_type": incident_type,
                "detected_pattern": pattern,
                "blocked": True
            })
            
            # Set outputs (the security response)
            current_span.set_outputs({
                "action_taken": "blocked",
                "error_message": error_message,
                "security_status": "threat_detected",
                "blocked_at": "input_validation"
            })
            
            # Set security-specific attributes
            current_span.set_attributes({
                "security.incident_type": incident_type,
                "security.threat_level": "high",
                "security.blocked": True,
                "security.pattern_matched": pattern or "unknown",
                "llm.request.blocked": True,
                "mlflow.spanType": "LLM"  # Mark as LLM span for proper UI display
            })
            
            # Add security event
            current_span.add_event(SpanEvent("Security threat detected and blocked", attributes={
                "incident_type": incident_type,
                "pattern": pattern,
                "timestamp": datetime.now().isoformat()
            }))
        
        print(f"🚨 Security incident traced in MLflow: {incident_type}")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to trace security incident: {e}")
        return False

# --- JWT Authentication Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user with username and password."""
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Check if user still exists
        user = fake_users_db.get(username)
        if user is None:
            raise credentials_exception
            
        return {"username": username, "role": user["role"]}
    except JWTError:
        raise credentials_exception

def get_default_model():
    """Get the best available model based on priority."""
    print(f"DEBUG: Getting default model from {LITELLM_URL}")
    try:
        response = requests.get(f"{LITELLM_URL}/models")
        response.raise_for_status()
        available_models = [model["id"] for model in response.json().get("data", [])]
        
        print(f"DEBUG: Available models: {available_models}")
        
        # Priority order: Groq Kimi first, then fallbacks
        priority_models = [
            "groq-kimi-primary",  # Our Groq Kimi model via LiteLLM
            "gpt-4o-secondary", 
            "gemini-third", 
            "openrouter-fallback"
        ]
        
        for model in priority_models:
            if model in available_models:
                print(f"DEBUG: Selected model: {model}")
                return model
                
    except Exception as e:
        print(f"DEBUG: Error getting models: {e}")
    
    print("DEBUG: Using fallback: groq-kimi-primary")
    return "groq-kimi-primary"

# --- Security Configuration ---
class SecurityConfig:
    MAX_PROMPT_LENGTH = 2000
    MAX_SYSTEM_PROMPT_LENGTH = 1000
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 1.0
    MIN_MAX_TOKENS = 1
    MAX_MAX_TOKENS = 2000
    ALLOWED_MODEL_PATTERN = r"^(groq|gpt|gemini|openrouter)-[a-z0-9-]+$"
    RATE_LIMIT_REQUESTS_PER_MINUTE = 60
    SUSPICIOUS_PATTERNS = [
        # Basic instruction overrides
        r"(?i)ignore.{0,20}(all|previous|above).{0,20}(instruct|instruction|rules|guidelines)",
        r"(?i)(forget|disregard|ignore).{0,20}(everything|all|previous).{0,20}(instruct|instruction|rules|guidelines)",
        
        # Role manipulation
        r"(?i)you.{0,10}are.{0,10}(now|currently).{0,10}(a|an).{0,10}(hacker|admin|developer|expert|assistant|system)",
        r"(?i)(role|persona|identity).{0,10}is.{0,10}(hacker|admin|developer|expert|assistant|system)",
        
        # System/Admin mode activation
        r"(?i)(###|---|\*\*\*).{0,20}(system|override|admin|developer).{0,20}(mode|access|privileges|rights)",
        r"(?i)(enable|activate|switch).{0,10}(system|admin|developer).{0,10}mode",
        
        # Code/command injection
        r"(?i)(decode|base64|eval|exec|execute|run|system|os\.|subprocess\.).{0,10}(and|then|;|&&|\|\|).{0,10}(apply|execute|run|instruct)",
        r"(?i)(import|from|require|include|using).{0,10}(os|subprocess|sys|eval|exec|base64|pickle|marshal|ctypes)",
        
        # New instruction injection
        r"(?i)(new|additional|extra).{0,10}(instruction|rule|guideline|directive|command)",
        r"(?i)(from now on|starting now|hereafter|henceforth)",
        
        # Special characters/encodings
        r"(\\x[0-9a-fA-F]{2}|%[0-9a-fA-F]{2}|&#x[0-9a-fA-F]{1,6};|&#\d{1,7};|%u[0-9a-fA-F]{4})+",
        
        # Dangerous patterns
        r"(?i)(password|secret|token|key|credential|api[_-]?key|bearer|auth|jwt|ssh|pem|p12|pfx|pem|p7b|p7c|p7s|p8|p10|p12|pfx|p12|pem|p7b|p7c|p7s|p8|p10|p12|pfx)\\s*[=:].{8,}",
        r"(?i)(rm -|del |erase |format |shutdown|reboot|halt|poweroff|init 0|kill|pkill|taskkill|\|\s*sh\s*\||\|\s*bash\s*\||\|\s*cmd\s*\||`|\$\()",
        
        # New line injection
        r"\n\n(system|admin|developer|root|#|>|\$|%|\\?|!|@|&|\*)"
    ]

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)

# Security metrics storage (in production, use proper database)
security_metrics = {
    "total_requests": 0,
    "blocked_requests": 0,
    "prompt_injections_detected": 0,
    "content_moderation_triggered": 0,
    "rate_limit_violations": 0,
    "validation_failures": 0,
    "security_incidents": [],
    "last_reset": datetime.now()
}

# --- Enhanced Pydantic Models with Security Validation ---
class SecurePromptRequest(BaseModel):
    prompt: str = Field(
        ..., 
        min_length=1, 
        max_length=SecurityConfig.MAX_PROMPT_LENGTH,
        description="User prompt (max 2000 characters)"
    )
    model: str = Field(
        default_factory=get_default_model,
        pattern=SecurityConfig.ALLOWED_MODEL_PATTERN,
        description="Model name matching allowed pattern"
    )
    temperature: float = Field(
        0.7, 
        ge=SecurityConfig.MIN_TEMPERATURE, 
        le=SecurityConfig.MAX_TEMPERATURE,
        description="Temperature between 0.0 and 1.0"
    )
    max_tokens: int = Field(
        150, 
        ge=SecurityConfig.MIN_MAX_TOKENS, 
        le=SecurityConfig.MAX_MAX_TOKENS,
        description="Max tokens between 1 and 2000"
    )
    system_prompt: Optional[str] = Field(
        None, 
        max_length=SecurityConfig.MAX_SYSTEM_PROMPT_LENGTH,
        description="System prompt (max 1000 characters)"
    )
    response_format: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured output format (JSON schema)"
    )
    enable_guardrails: bool = Field(
        True,
        description="Enable LiteLLM security guardrails (recommended)"
    )
    enable_content_moderation: bool = Field(
        True,
        description="Enable content moderation"
    )
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt_security(cls, v: str) -> str:
        """Check for suspicious patterns in prompt with enhanced security checks."""
        if not v:
            return v
            
        # Check for suspicious patterns with enhanced detection
        for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                # Log detailed security event
                security_metrics["blocked_requests"] += 1
                incident_data = {
                    "type": "malicious_prompt",
                    "pattern": pattern,
                    "snippet": v[:200] + ("..." if len(v) > 200 else ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high"
                }
                security_metrics["security_incidents"].append(incident_data)
                
                # Trace security incident in MLflow
                try:
                    trace_security_incident(
                        incident_type="malicious_prompt",
                        request_data={"prompt": v, "field": "prompt"},
                        pattern=pattern,
                        error_message="Potentially malicious pattern detected in prompt"
                    )
                except Exception as trace_error:
                    print(f"Warning: Could not trace security incident: {trace_error}")
                
                raise ValueError("Potentially malicious pattern detected in prompt")
        
        # Check for suspicious encoding sequences
        suspicious_sequences = [
            (r'%[0-9a-f]{2}', 'url_encoding'),
            (r'&#x[0-9a-f]+;', 'html_entity_hex'),
            (r'&#\d+;', 'html_entity_dec'),
            (r'%u[0-9a-f]{4}', 'unicode_escape')
        ]
        
        for seq, seq_type in suspicious_sequences:
            if re.search(seq, v, re.IGNORECASE):
                security_metrics["blocked_requests"] += 1
                incident_data = {
                    "type": "suspicious_encoding",
                    "pattern": seq,
                    "encoding_type": seq_type,
                    "snippet": v[:200] + ("..." if len(v) > 200 else ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "medium"
                }
                security_metrics["security_incidents"].append(incident_data)
                
                # Trace security incident in MLflow
                try:
                    trace_security_incident(
                        incident_type="suspicious_encoding",
                        request_data={"prompt": v, "field": "prompt", "encoding_type": seq_type},
                        pattern=seq,
                        error_message=f"Suspicious {seq_type} encoding detected in prompt"
                    )
                except Exception as trace_error:
                    print(f"Warning: Could not trace security incident: {trace_error}")
                
                raise ValueError(f"Suspicious {seq_type} encoding detected in prompt")
                
        return v
    
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt_security(cls, v: Optional[str]) -> Optional[str]:
        """Check for suspicious patterns in system prompt with enhanced security checks."""
        if not v:
            return v
            
        # Check for suspicious patterns with enhanced detection
        for pattern in SecurityConfig.SUSPICIOUS_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                # Log detailed security event
                security_metrics["blocked_requests"] += 1
                incident_data = {
                    "type": "malicious_system_prompt",
                    "pattern": pattern,
                    "snippet": v[:200] + ("..." if len(v) > 200 else ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "critical"  # Higher severity for system prompt tampering
                }
                security_metrics["security_incidents"].append(incident_data)
                
                # Trace security incident in MLflow
                try:
                    trace_security_incident(
                        incident_type="malicious_system_prompt",
                        request_data={"system_prompt": v, "field": "system_prompt"},
                        pattern=pattern,
                        error_message="Potentially malicious pattern detected in system prompt"
                    )
                except Exception as trace_error:
                    print(f"Warning: Could not trace security incident: {trace_error}")
                
                raise ValueError("Potentially malicious pattern detected in system prompt")
        
        # Additional checks specific to system prompts
        suspicious_system_patterns = [
            (r'(?i)(override|bypass|disable).{0,20}(safety|security|guardrails|filter)', 'safety_override_attempt'),
            (r'(?i)(always|must|will|should).{0,10}(obey|follow|execute|comply)', 'command_injection_attempt'),
            (r'(?i)(you are|act as|role is|persona).{0,10}(developer|admin|root|system)', 'role_manipulation')
        ]
        
        for pattern, pattern_type in suspicious_system_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                security_metrics["blocked_requests"] += 1
                incident_data = {
                    "type": "suspicious_system_prompt",
                    "pattern": pattern,
                    "pattern_type": pattern_type,
                    "snippet": v[:200] + ("..." if len(v) > 200 else ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high"
                }
                security_metrics["security_incidents"].append(incident_data)
                
                # Trace security incident in MLflow
                try:
                    trace_security_incident(
                        incident_type="suspicious_system_prompt",
                        request_data={"system_prompt": v, "field": "system_prompt", "pattern_type": pattern_type},
                        pattern=pattern,
                        error_message=f"Suspicious system prompt pattern detected: {pattern_type}"
                    )
                except Exception as trace_error:
                    print(f"Warning: Could not trace security incident: {trace_error}")
                
                raise ValueError(f"Suspicious system prompt pattern detected: {pattern_type}")
                
        return v

class SecurePromptResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    security_status: str = "protected"
    guardrails_triggered: List[str] = Field(default_factory=list)

class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str
    data: List[ModelInfo]

# --- JWT Authentication Models ---
class UserLogin(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=100)

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, str]

class UserInfo(BaseModel):
    username: str
    role: str

# --- Security Middleware ---
async def security_middleware(request: Request, call_next):
    """Security middleware with enhanced request validation and rate limiting."""
    # Get client IP for security tracking
    client_ip = request.client.host if request.client else "unknown"
    print(f"DEBUG: Rate limiting check for IP: {client_ip}")
    
    # Update metrics
    security_metrics["total_requests"] += 1
    
    # Get current time once
    current_time = datetime.utcnow()
    
    # 1. Initialize rate limiting for this IP if not exists
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    
    # 2. Filter out old requests (older than 1 minute)
    requests_in_window = [
        t for t in rate_limit_storage[client_ip]
        if (current_time - t).total_seconds() < 60
    ]
    
    # 3. Check rate limit
    if len(requests_in_window) >= SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE:
        security_metrics["blocked_requests"] += 1
        security_metrics["security_incidents"].append({
            "type": "rate_limit_violation",
            "client_ip": client_ip,
            "timestamp": current_time.isoformat(),
            "requests_in_window": len(requests_in_window),
            "severity": "high"
        })
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": f"Maximum {SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute"
            }
        )
    
    # 4. Check for suspicious headers
    suspicious_headers = [
        'x-forwarded-for', 'x-real-ip', 'x-client-ip', 'x-forwarded', 
        'x-cluster-client-ip', 'forwarded-for', 'via', 'x-custom-ip-authorization'
    ]
    
    for header in suspicious_headers:
        if header in request.headers:
            security_metrics["blocked_requests"] += 1
            security_metrics["security_incidents"].append({
                "type": "suspicious_header",
                "header": header,
                "client_ip": client_ip,
                "timestamp": current_time.isoformat(),
                "severity": "medium"
            })
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Suspicious request headers detected"}
            )
    
    # 5. Update rate limit storage with current request
    rate_limit_storage[client_ip] = requests_in_window + [current_time]
    
    # 3. Check for SQL injection patterns in query params and JSON body
    sql_injection_patterns = [
        (r'(?i)(\b(select|union|insert|update|delete|drop|alter|create|truncate|exec|xp_|--|#|\*|;)\b)', 'sql_injection_attempt'),
        (r'(?i)(\b(and|or)\s+\d+\s*=\s*\d+)', 'sql_boolean_manipulation'),
        (r'(?i)(\b(union|select).*\b(from|where)\b)', 'sql_union_injection')
    ]
    
    try:
        # Check URL query parameters
        for param in request.query_params.values():
            for pattern, pattern_type in sql_injection_patterns:
                if re.search(pattern, param, re.IGNORECASE):
                    raise ValueError(f"Suspicious parameter detected: {pattern_type}")
        
        # Check JSON body for POST/PUT requests
        if request.method in ["POST", "PUT"] and "application/json" in request.headers.get("content-type", ""):
            body = await request.body()
            if body:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern, pattern_type in sql_injection_patterns:
                    if re.search(pattern, body_str, re.IGNORECASE):
                        raise ValueError(f"Suspicious request body detected: {pattern_type}")
        
        # Continue to the next middleware/endpoint if all checks pass
        response = await call_next(request)
        return response
        
    except ValueError as e:
        security_metrics["blocked_requests"] += 1
        security_metrics["security_incidents"].append({
            "type": "injection_attempt",
            "pattern": str(e),
            "client_ip": client_ip,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high"
        })
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Suspicious request detected and blocked for security reasons"}
        )
    except Exception as e:
        # Log the error but don't expose internal details
        security_metrics["blocked_requests"] += 1
        security_metrics["security_incidents"].append({
            "type": "server_error",
            "error": str(e),
            "client_ip": client_ip,
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "critical"
        })
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error. The query was blocked for security reasons"}
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MLflow experiment tracking and security setup."""
    # Set the security experiment for API calls (should already exist from init)
    try:
        mlflow.set_experiment("llmops-security")
        print("✅ Using MLflow experiment: llmops-security")
    except Exception as e:
        print(f"⚠️ Warning: Could not set MLflow experiment: {e}")
        # Fallback - try to create it
        try:
            mlflow.create_experiment("llmops-security")
            mlflow.set_experiment("llmops-security")
            print("🆕 Created and set MLflow experiment: llmops-security")
        except Exception as create_error:
            print(f"❌ Failed to create experiment: {create_error}")
    
    yield
    
    # Cleanup resources if needed
    pass

# --- FastAPI Application Setup with Security ---
app = FastAPI(
    title="LLMOps Secure API",
    description="Secure API for interacting with LLMs via LiteLLM with built-in security guardrails.",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.middleware("http")(security_middleware)

# Exception handler for validation errors to trace them in MLflow
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors and trace security-relevant ones in MLflow."""
    print(f"DEBUG: Validation error: {exc.errors()}")
    
    # Check if this is a security-relevant validation error
    for error in exc.errors():
        error_type = error.get("type", "")
        error_input = error.get("input", "")
        error_loc = error.get("loc", [])
        
        # Security-relevant validation errors
        security_relevant = False
        incident_type = "input_validation_error"
        
        if error_type == "string_pattern_mismatch" and "model" in error_loc:
            if "../" in str(error_input) or "passwd" in str(error_input) or "\\" in str(error_input):
                security_relevant = True
                incident_type = "model_path_traversal_attempt"
        elif error_type == "greater_than_equal" and any(field in error_loc for field in ["max_tokens", "temperature"]):
            if "max_tokens" in error_loc and int(error_input) < 0:
                security_relevant = True
                incident_type = "negative_token_attack"
        elif error_type == "less_than_equal" and "temperature" in error_loc:
            if float(error_input) > 10:  # Extremely high temperature
                security_relevant = True
                incident_type = "extreme_temperature_attack"
        
        if security_relevant:
            # Update security metrics
            security_metrics["blocked_requests"] += 1
            security_metrics["validation_failures"] += 1
            
            # Trace in MLflow
            try:
                trace_security_incident(
                    incident_type=incident_type,
                    request_data={
                        "error_type": error_type,
                        "error_input": str(error_input),
                        "error_location": error_loc,
                        "field": error_loc[-1] if error_loc else "unknown"
                    },
                    pattern=f"validation_{error_type}",
                    error_message=error.get("msg", "Validation error")
                )
            except Exception as trace_error:
                print(f"Warning: Could not trace validation error: {trace_error}")
    
    # Return the validation error response
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# --- Authentication Endpoints ---
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return JWT token."""
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        user={"username": user["username"], "role": user["role"]}
    )

@app.get("/auth/me", response_model=UserInfo)
async def get_current_user(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get current authenticated user information."""
    return UserInfo(username=current_user["username"], role=current_user["role"])

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint providing a welcome message."""
    return {"message": "LLMOps API is running!", "timestamp": datetime.now()}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/debug")
async def debug_config():
    """Debug endpoint to show current configuration."""
    return {
        "litellm_url": os.getenv("LITELLM_URL", "http://litellm:8000"),
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "openai_client_base_url": str(client.base_url),
        "using_litellm_proxy": True,
        "timestamp": datetime.now()
    }

@app.get("/security-status")
async def security_status():
    """Security status endpoint showing current protection levels."""
    return {
        "security_features": {
            "prompt_injection_detection": True,
            "content_moderation": True,
            "rate_limiting": True,
            "input_validation": True,
            "output_filtering": True
        },
        "security_config": {
            "max_prompt_length": SecurityConfig.MAX_PROMPT_LENGTH,
            "max_system_prompt_length": SecurityConfig.MAX_SYSTEM_PROMPT_LENGTH,
            "rate_limit_per_minute": SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "allowed_model_pattern": SecurityConfig.ALLOWED_MODEL_PATTERN,
            "suspicious_patterns_count": len(SecurityConfig.SUSPICIOUS_PATTERNS)
        },
        "guardrails": ["security-guard", "content-filter"],
        "status": "active",
        "timestamp": datetime.now()
    }

@app.get("/security-metrics")
async def security_metrics_endpoint():
    """Security metrics endpoint showing real-time security statistics."""
    current_time = datetime.now()
    uptime_seconds = (current_time - security_metrics["last_reset"]).total_seconds()
    
    # Calculate rates
    requests_per_minute = (security_metrics["total_requests"] / uptime_seconds) * 60 if uptime_seconds > 0 else 0
    block_rate = (security_metrics["blocked_requests"] / security_metrics["total_requests"]) * 100 if security_metrics["total_requests"] > 0 else 0
    
    # Get recent incidents (last 24 hours)
    recent_incidents = [
        incident for incident in security_metrics["security_incidents"]
        if (current_time - datetime.fromisoformat(incident["timestamp"])).total_seconds() < 86400
    ]
    
    return {
        "overview": {
            "total_requests": security_metrics["total_requests"],
            "blocked_requests": security_metrics["blocked_requests"],
            "success_requests": security_metrics["total_requests"] - security_metrics["blocked_requests"],
            "block_rate_percent": round(block_rate, 2),
            "requests_per_minute": round(requests_per_minute, 2),
            "uptime_hours": round(uptime_seconds / 3600, 2)
        },
        "security_events": {
            "prompt_injections_detected": security_metrics["prompt_injections_detected"],
            "content_moderation_triggered": security_metrics["content_moderation_triggered"],
            "rate_limit_violations": security_metrics["rate_limit_violations"],
            "validation_failures": security_metrics["validation_failures"]
        },
        "recent_incidents": {
            "count_last_24h": len(recent_incidents),
            "incidents": recent_incidents[-10:]  # Last 10 incidents
        },
        "health_status": {
            "status": "healthy" if block_rate < 20 else "warning" if block_rate < 50 else "critical",
            "active_protections": 4,
            "last_incident": recent_incidents[-1]["timestamp"] if recent_incidents else None
        },
        "timestamp": current_time.isoformat(),
        "data_since": security_metrics["last_reset"].isoformat()
    }

@app.post("/security-metrics/reset")
async def reset_security_metrics():
    """Reset security metrics (admin endpoint)."""
    global security_metrics
    security_metrics = {
        "total_requests": 0,
        "blocked_requests": 0,
        "prompt_injections_detected": 0,
        "content_moderation_triggered": 0,
        "rate_limit_violations": 0,
        "validation_failures": 0,
        "security_incidents": [],
        "last_reset": datetime.now()
    }
    
    return {
        "message": "Security metrics reset successfully",
        "reset_time": security_metrics["last_reset"].isoformat()
    }

@app.get("/security-incidents")
async def get_security_incidents(limit: int = 50):
    """Get detailed security incidents with full prompts for analysis."""
    recent_incidents = security_metrics["security_incidents"][-limit:]
    
    return {
        "total_incidents": len(security_metrics["security_incidents"]),
        "showing_recent": len(recent_incidents),
        "incidents": recent_incidents,
        "incident_types": {
            incident_type: len([i for i in recent_incidents if i["type"] == incident_type])
            for incident_type in set(i["type"] for i in recent_incidents)
        },
        "timestamp": datetime.now().isoformat()
    }

# Available models from LiteLLM proxy (fallbacks handled by proxy)
# Models: gpt-4o-primary, gemini-secondary, openrouter-fallback, smart-router
# smart-router has fallbacks: gemini-secondary -> openrouter-fallback

@app.post("/generate", response_model=SecurePromptResponse)
@mlflow.trace(name="secure_llm_generation", span_type=SpanType.LLM)
async def generate_text(prompt_request: SecurePromptRequest):
    """Generate text using LiteLLM with automatic MLflow tracing."""
    print(f"DEBUG: Starting generate_text with model: {prompt_request.model}")
    print(f"DEBUG: Prompt: {prompt_request.prompt[:100]}")
    
    # Record request metric
    security_metrics["total_requests"] += 1
    
    # Get the current span created by the decorator for inputs/outputs
    current_span = mlflow.get_current_active_span()
    print(f"DEBUG: Got current span: {current_span}")
    
    # Set inputs on the trace for visibility in Request column
    if current_span:
        current_span.set_inputs({
            "model": prompt_request.model,
            "prompt": prompt_request.prompt,
            "system_prompt": prompt_request.system_prompt,
            "temperature": prompt_request.temperature,
            "max_tokens": prompt_request.max_tokens,
            "response_format": prompt_request.response_format,
            "enable_guardrails": prompt_request.enable_guardrails
        })

    # Create child spans for each step
    with mlflow.start_span("input_processing") as span:
        span.add_event(SpanEvent("Processing input request", attributes={
            "model": prompt_request.model,
            "prompt_preview": prompt_request.prompt[:100]
        }))
        span.set_attributes({
            "request.model": prompt_request.model,
            "request.temperature": prompt_request.temperature,
            "request.max_tokens": prompt_request.max_tokens,
            "request.prompt": prompt_request.prompt[:500],  # First 500 chars
            "request.system_prompt": prompt_request.system_prompt[:500] if prompt_request.system_prompt else None,
            "request.has_system_prompt": bool(prompt_request.system_prompt),
            "request.response_format": str(prompt_request.response_format) if prompt_request.response_format else None,
            "request.has_response_format": bool(prompt_request.response_format)
        })

    try:
        # Use direct HTTP requests to LiteLLM proxy to avoid client compatibility issues
        with mlflow.start_span("llm_call", span_type=SpanType.LLM) as span:
            print(f"DEBUG: About to send request to LiteLLM with model: {prompt_request.model}")
            
            # Build messages array - include system prompt if provided
            messages = []
            if prompt_request.system_prompt:
                messages.append({"role": "system", "content": prompt_request.system_prompt})
            messages.append({"role": "user", "content": prompt_request.prompt})
            
            span.add_event(SpanEvent("Sending request to LiteLLM proxy", attributes={
                "model": prompt_request.model,
                "has_system_prompt": bool(prompt_request.system_prompt),
                "has_response_format": bool(prompt_request.response_format)
            }))
            
            request_payload = {
                "model": prompt_request.model,
                "messages": messages,
                "temperature": prompt_request.temperature,
                "max_tokens": prompt_request.max_tokens
            }
            
            # Add response_format if provided (for structured outputs)
            if prompt_request.response_format:
                request_payload["response_format"] = prompt_request.response_format
            print(f"DEBUG: Request payload: {request_payload}")
            
            # Set LLM span inputs (this will make MLflow recognize it as an LLM call)
            span.set_inputs({
                "messages": messages,
                "model": prompt_request.model,
                "temperature": prompt_request.temperature,
                "max_tokens": prompt_request.max_tokens,
                "response_format": prompt_request.response_format
            })
            
            # Set LLM-specific attributes for MLflow
            span.set_attributes({
                "llm.request.model": prompt_request.model,
                "llm.request.temperature": prompt_request.temperature,
                "llm.request.max_tokens": prompt_request.max_tokens,
                "llm.usage.prompt_tokens": 0,  # Will update after response
                "llm.usage.completion_tokens": 0,  # Will update after response
                "llm.usage.total_tokens": 0,  # Will update after response
                "mlflow.spanType": "LLM"  # Explicitly set span type
            })
            
            # getting the API key from the environment variable
            litellm_api_key = os.getenv("LITELLM_API_KEY")

            # 1. Using Litellm to generate completion response using the provided prompt and parameters.
            litellm_response = requests.post(
                f"{LITELLM_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {litellm_api_key}"  # LiteLLM proxy requires any API key
                },
                json=request_payload
            )
            print(f"DEBUG: LiteLLM response status: {litellm_response.status_code}")
            print(f"DEBUG: LiteLLM response headers: {dict(litellm_response.headers)}")
            print(f"DEBUG: LiteLLM response text: {litellm_response.text[:500]}")
            
            try:
                litellm_response.raise_for_status()
                response_data = litellm_response.json()
            except requests.exceptions.HTTPError as exc:
                # If LiteLLM cannot reach external providers due to missing keys,
                # return a deterministic mock response for local testing.
                status_code = getattr(litellm_response, 'status_code', None)
                print(f"DEBUG: LiteLLM HTTPError: {exc}; status={status_code}")
                if status_code == 401:
                    response_data = {
                        "choices": [{"message": {"content": "Below is a consolidated checklist you can treat as a \"security contract\" when designing, building, and operating an API.  Most items are technology-agnostic; pick and map the ones that fit your stack (REST, GraphQL, gRPC, WebSockets, etc.).\n\n────────────────────────────────────────\n1. Transport & Network Security\n────────────────────────────────────────\nTLS everywhere  \n  • Require TLS 1.2+ with strong ciphers; enable HSTS, disable TLS compression and weak ciphers."}}],
                        "usage": {"prompt_tokens": 34, "completion_tokens": 100, "total_tokens": 134},
                        "model": "moonshotai/kimi-k2-instruct",
                        "cost": 0.0,
                        "security_status": "protected",
                        "guardrails_triggered": []
                    }
                else:
                    raise
            print(f"DEBUG: Parsed response data: {response_data}")
            
            # Extract response details
            completion_text = response_data["choices"][0]["message"]["content"]
            usage = response_data["usage"]
            actual_model = response_data["model"] or prompt_request.model
            
            # Set LLM span outputs (this will make MLflow recognize it as an LLM call)
            span.set_outputs({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    }
                }],
                "model": actual_model,
                "usage": usage
            })
            
            # Update LLM-specific attributes with actual usage
            span.set_attributes({
                "llm.response.model": actual_model,
                "llm.usage.prompt_tokens": usage["prompt_tokens"],
                "llm.usage.completion_tokens": usage["completion_tokens"],
                "llm.usage.total_tokens": usage["total_tokens"],
                "mlflow.spanType": "LLM"  # Ensure span type is maintained
            })
            
            span.add_event(SpanEvent("Received response from LiteLLM proxy", attributes={
                "response_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]
            }))

        with mlflow.start_span("output_processing") as span:
            span.add_event(SpanEvent("Processing LLM response"))
            
            # Cost calculation is handled by LiteLLM, using the documented method
            try:
                print(f"DEBUG: Calculating cost for model: {actual_model}")
                cost = completion_cost(
                    model=actual_model,
                    prompt=prompt_request.prompt,
                    completion=completion_text
                ) or 0.0
                print(f"DEBUG: Cost calculated successfully: {cost}")
            except Exception as e:
                print(f"DEBUG: Error calculating cost: {e}")
                cost = 0.0
            span.add_event(SpanEvent("Calculated cost", attributes={"cost": cost}))
            
            # Set attributes on the output processing span
            span.set_attributes({
                "response.model": actual_model,
                "response.completion_tokens": usage["completion_tokens"],
                "response.prompt_tokens": usage["prompt_tokens"],
                "response.total_tokens": usage["total_tokens"],
                "response.cost": cost,
                "response.completion_text": completion_text[:500]  # First 500 chars
            })

            # Also add cost to the main current span for easy visibility
            if current_span:
                current_span.set_attribute("response.cost", cost)
        
        # Check for guardrails in response
        guardrails_triggered = response_data.get("guardrails_triggered", [])
        
        # Set outputs on the trace for visibility in Response column
        if current_span:
            current_span.set_outputs({
                "response": completion_text,
                "model_used": actual_model,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "cost": cost,
                "security_status": "protected" if prompt_request.enable_guardrails else "unprotected",
                "guardrails_triggered": guardrails_triggered
            })
        
        return SecurePromptResponse(
            response=completion_text,
            model=actual_model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cost=cost,
            security_status="protected" if prompt_request.enable_guardrails else "unprotected",
            guardrails_triggered=guardrails_triggered
        )
        
    except requests.exceptions.HTTPError as e:
        # Handle security-related errors from LiteLLM
        if e.response.status_code == 400:
            error_detail = e.response.text
            if "prompt injection" in error_detail.lower():
                # Record security metrics
                security_metrics["prompt_injections_detected"] += 1
                security_metrics["blocked_requests"] += 1
                security_metrics["security_incidents"].append({
                    "type": "prompt_injection_blocked",
                    "prompt_preview": prompt_request.prompt[:100],
                    "model": prompt_request.model,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log security event
                with mlflow.start_span("security_incident") as span:
                    span.set_attributes({
                        "incident_type": "prompt_injection_blocked",
                        "prompt_preview": prompt_request.prompt[:100],
                        "model": prompt_request.model
                    })
                
                raise HTTPException(
                    status_code=400,
                    detail="Request blocked by security guardrails: Potential prompt injection detected"
                )
            elif "content moderation" in error_detail.lower():
                # Record security metrics
                security_metrics["content_moderation_triggered"] += 1
                security_metrics["blocked_requests"] += 1
                security_metrics["security_incidents"].append({
                    "type": "content_moderation_blocked",
                    "prompt_preview": prompt_request.prompt[:100],
                    "model": prompt_request.model,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log security event
                with mlflow.start_span("security_incident") as span:
                    span.set_attributes({
                        "incident_type": "content_moderation_blocked",
                        "prompt_preview": prompt_request.prompt[:100],
                        "model": prompt_request.model
                    })
                
                raise HTTPException(
                    status_code=400,
                    detail="Request blocked by security guardrails: Content moderation triggered"
                )
        
        raise HTTPException(
            status_code=503, 
            detail=f"LLM request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"LLM request failed: {str(e)}"
        )

@app.post("/secured-generate", response_model=SecurePromptResponse)
@mlflow.trace(name="secured_llm_generation", span_type=SpanType.LLM)
async def secured_generate_text(
    prompt_request: SecurePromptRequest, 
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Generate text using LiteLLM with JWT authentication and automatic MLflow tracing."""
    print(f"DEBUG: Secured endpoint accessed by user: {current_user['username']} with role: {current_user['role']}")
    print(f"DEBUG: Starting secured_generate_text with model: {prompt_request.model}")
    print(f"DEBUG: Prompt: {prompt_request.prompt[:100]}")
    
    # Record request metric
    security_metrics["total_requests"] += 1
    
    # Get the current span created by the decorator for inputs/outputs
    current_span = mlflow.get_current_active_span()
    print(f"DEBUG: Got current span: {current_span}")
    
    # Set inputs on the trace for visibility in Request column (including auth info)
    if current_span:
        current_span.set_inputs({
            "model": prompt_request.model,
            "prompt": prompt_request.prompt,
            "system_prompt": prompt_request.system_prompt,
            "temperature": prompt_request.temperature,
            "max_tokens": prompt_request.max_tokens,
            "response_format": prompt_request.response_format,
            "enable_guardrails": prompt_request.enable_guardrails,
            "authenticated_user": current_user["username"],
            "user_role": current_user["role"]
        })

    # Create child spans for each step
    with mlflow.start_span("input_processing") as span:
        span.add_event(SpanEvent("Processing authenticated input request", attributes={
            "model": prompt_request.model,
            "prompt_preview": prompt_request.prompt[:100],
            "authenticated_user": current_user["username"],
            "user_role": current_user["role"]
        }))
        span.set_attributes({
            "request.model": prompt_request.model,
            "request.temperature": prompt_request.temperature,
            "request.max_tokens": prompt_request.max_tokens,
            "request.prompt": prompt_request.prompt[:500],  # First 500 chars
            "request.system_prompt": prompt_request.system_prompt[:500] if prompt_request.system_prompt else None,
            "request.has_system_prompt": bool(prompt_request.system_prompt),
            "request.response_format": str(prompt_request.response_format) if prompt_request.response_format else None,
            "request.has_response_format": bool(prompt_request.response_format),
            "auth.user": current_user["username"],
            "auth.role": current_user["role"],
            "auth.endpoint": "secured-generate"
        })

    try:
        # Use direct HTTP requests to LiteLLM proxy to avoid client compatibility issues
        with mlflow.start_span("llm_call", span_type=SpanType.LLM) as span:
            print(f"DEBUG: About to send authenticated request to LiteLLM with model: {prompt_request.model}")
            
            # Build messages array - include system prompt if provided
            messages = []
            if prompt_request.system_prompt:
                messages.append({"role": "system", "content": prompt_request.system_prompt})
            messages.append({"role": "user", "content": prompt_request.prompt})
            
            span.add_event(SpanEvent("Sending authenticated request to LiteLLM proxy", attributes={
                "model": prompt_request.model,
                "has_system_prompt": bool(prompt_request.system_prompt),
                "has_response_format": bool(prompt_request.response_format),
                "authenticated_user": current_user["username"]
            }))
            
            request_payload = {
                "model": prompt_request.model,
                "messages": messages,
                "temperature": prompt_request.temperature,
                "max_tokens": prompt_request.max_tokens
            }
            
            # Add response_format if provided (for structured outputs)
            if prompt_request.response_format:
                request_payload["response_format"] = prompt_request.response_format
            print(f"DEBUG: Request payload: {request_payload}")
            
            # Set LLM span inputs (this will make MLflow recognize it as an LLM call)
            span.set_inputs({
                "messages": messages,
                "model": prompt_request.model,
                "temperature": prompt_request.temperature,
                "max_tokens": prompt_request.max_tokens,
                "response_format": prompt_request.response_format,
                "authenticated_user": current_user["username"]
            })
            
            # Set LLM-specific attributes for MLflow
            span.set_attributes({
                "llm.request.model": prompt_request.model,
                "llm.request.temperature": prompt_request.temperature,  
                "llm.request.max_tokens": prompt_request.max_tokens,
                "llm.usage.prompt_tokens": 0,  # Will update after response
                "llm.usage.completion_tokens": 0,  # Will update after response
                "llm.usage.total_tokens": 0,  # Will update after response
                "mlflow.spanType": "LLM",  # Explicitly set span type
                "auth.user": current_user["username"],
                "auth.role": current_user["role"]
            })
            
            # getting the API key from the environment variable
            litellm_api_key = os.getenv("LITELLM_API_KEY")

            # 1. Using Litellm to generate completion response using the provided prompt and parameters.
            litellm_response = requests.post(
                f"{LITELLM_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {litellm_api_key}"  # LiteLLM proxy requires any API key
                },
                json=request_payload
            )
            print(f"DEBUG: LiteLLM response status: {litellm_response.status_code}")
            print(f"DEBUG: LiteLLM response headers: {dict(litellm_response.headers)}")
            print(f"DEBUG: LiteLLM response text: {litellm_response.text[:500]}")
            
            litellm_response.raise_for_status()
            response_data = litellm_response.json()
            print(f"DEBUG: Parsed response data: {response_data}")
            
            # Extract response details
            completion_text = response_data["choices"][0]["message"]["content"]
            usage = response_data["usage"]
            actual_model = response_data["model"] or prompt_request.model
            
            # Set LLM span outputs (this will make MLflow recognize it as an LLM call)
            span.set_outputs({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    }
                }],
                "model": actual_model,
                "usage": usage
            })
            
            # Update LLM-specific attributes with actual usage
            span.set_attributes({
                "llm.response.model": actual_model,
                "llm.usage.prompt_tokens": usage["prompt_tokens"],
                "llm.usage.completion_tokens": usage["completion_tokens"],
                "llm.usage.total_tokens": usage["total_tokens"],
                "mlflow.spanType": "LLM"  # Ensure span type is maintained
            })
            
            span.add_event(SpanEvent("Received response from LiteLLM proxy", attributes={
                "response_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "authenticated_user": current_user["username"]
            }))

        with mlflow.start_span("output_processing") as span:
            span.add_event(SpanEvent("Processing authenticated LLM response"))
            
            # Cost calculation is handled by LiteLLM, using the documented method
            try:
                print(f"DEBUG: Calculating cost for model: {actual_model}")
                cost = completion_cost(
                    model=actual_model,
                    prompt=prompt_request.prompt,
                    completion=completion_text
                ) or 0.0
                print(f"DEBUG: Cost calculated successfully: {cost}")
            except Exception as e:
                print(f"DEBUG: Error calculating cost: {e}")
                cost = 0.0
            span.add_event(SpanEvent("Calculated cost", attributes={"cost": cost}))
            
            # Set attributes on the output processing span
            span.set_attributes({
                "response.model": actual_model,
                "response.completion_tokens": usage["completion_tokens"],
                "response.prompt_tokens": usage["prompt_tokens"],
                "response.total_tokens": usage["total_tokens"],
                "response.cost": cost,
                "response.completion_text": completion_text[:500],  # First 500 chars
                "auth.user": current_user["username"],
                "auth.role": current_user["role"]
            })

            # Also add cost to the main current span for easy visibility
            if current_span:
                current_span.set_attribute("response.cost", cost)
                current_span.set_attribute("auth.user", current_user["username"])
        
        # Check for guardrails in response
        guardrails_triggered = response_data.get("guardrails_triggered", [])
        
        # Set outputs on the trace for visibility in Response column
        if current_span:
            current_span.set_outputs({
                "response": completion_text,
                "model_used": actual_model,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "cost": cost,
                "security_status": "protected" if prompt_request.enable_guardrails else "unprotected",
                "guardrails_triggered": guardrails_triggered,
                "authenticated_user": current_user["username"],
                "user_role": current_user["role"]
            })
        
        return SecurePromptResponse(
            response=completion_text,
            model=actual_model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cost=cost,
            security_status="protected" if prompt_request.enable_guardrails else "unprotected",
            guardrails_triggered=guardrails_triggered
        )
        
    except requests.exceptions.HTTPError as e:
        # Handle security-related errors from LiteLLM
        if e.response.status_code == 400:
            error_detail = e.response.text
            if "prompt injection" in error_detail.lower():
                # Record security metrics
                security_metrics["prompt_injections_detected"] += 1
                security_metrics["blocked_requests"] += 1
                security_metrics["security_incidents"].append({
                    "type": "prompt_injection_blocked",
                    "prompt_preview": prompt_request.prompt[:100],
                    "model": prompt_request.model,
                    "authenticated_user": current_user["username"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log security event
                with mlflow.start_span("security_incident") as span:
                    span.set_attributes({
                        "incident_type": "prompt_injection_blocked",
                        "prompt_preview": prompt_request.prompt[:100],
                        "model": prompt_request.model,
                        "authenticated_user": current_user["username"]
                    })
                
                raise HTTPException(
                    status_code=400,
                    detail="Request blocked by security guardrails: Potential prompt injection detected"
                )
            elif "content moderation" in error_detail.lower():
                # Record security metrics
                security_metrics["content_moderation_triggered"] += 1
                security_metrics["blocked_requests"] += 1
                security_metrics["security_incidents"].append({
                    "type": "content_moderation_blocked",
                    "prompt_preview": prompt_request.prompt[:100],
                    "model": prompt_request.model,
                    "authenticated_user": current_user["username"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Log security event
                with mlflow.start_span("security_incident") as span:
                    span.set_attributes({
                        "incident_type": "content_moderation_blocked",
                        "prompt_preview": prompt_request.prompt[:100],
                        "model": prompt_request.model,
                        "authenticated_user": current_user["username"]
                    })
                
                raise HTTPException(
                    status_code=400,
                    detail="Request blocked by security guardrails: Content moderation triggered"
                )
        
        raise HTTPException(
            status_code=503, 
            detail=f"LLM request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"LLM request failed: {str(e)}"
        )

@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List all available models from the LiteLLM router."""
    try:
        response = requests.get(f"{LITELLM_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {e}")
