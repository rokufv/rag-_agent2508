"""
Logging configuration and utilities for Agent RAG Studio
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional
import traceback

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        max_bytes: Maximum log file size
        backup_count: Number of backup files to keep
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                log_path / "app.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        ]
    )
    
    # Create specific loggers
    loggers = [
        'rag',
        'agent', 
        'streamlit',
        'evaluation'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Add dedicated file handler for each module
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{logger_name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

class StructuredLogger:
    """Structured logging for analytics and monitoring"""
    
    def __init__(self, logger_name: str = "analytics", log_dir: str = "logs"):
        self.logger = logging.getLogger(logger_name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup JSON file handler
        json_handler = logging.FileHandler(
            self.log_dir / "analytics.jsonl",
            encoding='utf-8'
        )
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(json_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_query(
        self,
        user_id: str,
        query: str,
        response: str,
        sources: list,
        latency: float,
        model: str,
        confidence: float = None,
        **kwargs
    ):
        """Log a query and response"""
        log_entry = {
            "event_type": "query",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query_hash": hash(query) % 1000000,  # Anonymized query hash
            "query_length": len(query),
            "response_length": len(response),
            "num_sources": len(sources),
            "latency": latency,
            "model": model,
            "confidence": confidence,
            **kwargs
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None,
        **kwargs
    ):
        """Log an error with context"""
        log_entry = {
            "event_type": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "traceback": traceback.format_exc(),
            **kwargs
        }
        
        self.logger.error(json.dumps(log_entry, ensure_ascii=False))
    
    def log_evaluation(
        self,
        eval_type: str,
        metrics: Dict[str, float],
        dataset_size: int,
        **kwargs
    ):
        """Log evaluation results"""
        log_entry = {
            "event_type": "evaluation",
            "timestamp": datetime.now().isoformat(),
            "eval_type": eval_type,
            "metrics": metrics,
            "dataset_size": dataset_size,
            **kwargs
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_system_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        **kwargs
    ):
        """Log general system events"""
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            **kwargs
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))

class AuditLogger:
    """Audit logging for security and compliance"""
    
    def __init__(self, log_dir: str = "logs"):
        self.logger = logging.getLogger("audit")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup audit file handler
        audit_handler = logging.FileHandler(
            self.log_dir / "audit.log",
            encoding='utf-8'
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        ip_address: str = None,
        user_agent: str = None,
        **kwargs
    ):
        """Log access events"""
        log_message = json.dumps({
            "event": "access",
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }, ensure_ascii=False)
        
        self.logger.info(log_message)
    
    def log_data_operation(
        self,
        user_id: str,
        operation: str,
        data_type: str,
        count: int = None,
        **kwargs
    ):
        """Log data operations"""
        log_message = json.dumps({
            "event": "data_operation",
            "user_id": user_id,
            "operation": operation,
            "data_type": data_type,
            "count": count,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }, ensure_ascii=False)
        
        self.logger.info(log_message)
    
    def log_configuration_change(
        self,
        user_id: str,
        setting: str,
        old_value: Any,
        new_value: Any,
        **kwargs
    ):
        """Log configuration changes"""
        log_message = json.dumps({
            "event": "config_change",
            "user_id": user_id,
            "setting": setting,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }, ensure_ascii=False)
        
        self.logger.warning(log_message)

# Global logger instances
structured_logger = StructuredLogger()
audit_logger = AuditLogger()

def mask_sensitive_data(data: str, patterns: list = None) -> str:
    """Mask sensitive data in log messages"""
    import re
    
    if patterns is None:
        patterns = [
            r'([Aa]pi[_-]?[Kk]ey["\s]*[:=]["\s]*)([a-zA-Z0-9-_]{20,})',
            r'([Tt]oken["\s]*[:=]["\s]*)([a-zA-Z0-9-_]{20,})',
            r'([Pp]assword["\s]*[:=]["\s]*)([^\s"\']+)',
            r'([Ss]ecret["\s]*[:=]["\s]*)([a-zA-Z0-9-_]{20,})',
        ]
    
    masked_data = data
    for pattern in patterns:
        masked_data = re.sub(pattern, r'\1***MASKED***', masked_data)
    
    return masked_data

def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        import time
        
        start_time = time.time()
        function_name = f"{func.__module__}.{func.__name__}"
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            structured_logger.log_system_event(
                "function_execution",
                {
                    "function": function_name,
                    "execution_time": execution_time,
                    "success": True,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            structured_logger.log_error(
                "function_error",
                str(e),
                {
                    "function": function_name,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
            )
            
            raise
    
    return wrapper
