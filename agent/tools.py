"""
Tools for LangGraph Agent
"""
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime, timedelta
import requests
import os

logger = logging.getLogger(__name__)

class WebSearchTool:
    """Web search tool using SerpAPI or similar"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information"""
        if not self.api_key:
            logger.warning("Web search API key not available")
            return []
        
        try:
            # Simulate web search results for now
            # In production, you'd use actual SerpAPI or Bing Search
            mock_results = [
                {
                    "title": f"検索結果 {i+1}: {query}",
                    "snippet": f"これは{query}に関する検索結果のスニペットです。詳細な情報が含まれています。",
                    "url": f"https://example.com/result-{i+1}",
                    "source": "web_search",
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(min(num_results, 3))
            ]
            
            logger.info(f"Web search for '{query}' returned {len(mock_results)} results")
            return mock_results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

class CalculatorTool:
    """Calculator tool for mathematical operations"""
    
    def calculate(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions"""
        try:
            # Sanitize expression - only allow safe characters
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Invalid characters in expression")
            
            # Evaluate the expression
            result = eval(expression)
            
            return {
                "expression": expression,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return {
                "expression": expression,
                "result": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

class DataAnalysisTool:
    """Tool for data analysis and processing"""
    
    def analyze_data(self, data: List[Dict[str, Any]], operation: str) -> Dict[str, Any]:
        """Analyze data using pandas operations"""
        try:
            if not data:
                return {"error": "No data provided", "success": False}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            result = {"success": True, "operation": operation}
            
            if operation == "describe":
                # Statistical description
                description = df.describe()
                result["description"] = description.to_dict()
                
            elif operation == "count":
                # Count rows and columns
                result["row_count"] = len(df)
                result["column_count"] = len(df.columns)
                result["columns"] = list(df.columns)
                
            elif operation == "summary":
                # General summary
                result["summary"] = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "null_counts": df.isnull().sum().to_dict()
                }
                
            elif operation == "sample":
                # Show sample data
                result["sample"] = df.head().to_dict('records')
                
            else:
                result["error"] = f"Unknown operation: {operation}"
                result["success"] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Data analysis error: {e}")
            return {
                "error": str(e),
                "success": False,
                "operation": operation
            }

class DateTimeTool:
    """Tool for date and time operations"""
    
    def get_current_datetime(self) -> Dict[str, Any]:
        """Get current date and time"""
        now = datetime.now()
        return {
            "current_datetime": now.isoformat(),
            "formatted": now.strftime("%Y年%m月%d日 %H:%M:%S"),
            "timestamp": now.timestamp(),
            "weekday": now.strftime("%A"),
            "weekday_jp": ["月", "火", "水", "木", "金", "土", "日"][now.weekday()]
        }
    
    def calculate_date_difference(self, date1: str, date2: str) -> Dict[str, Any]:
        """Calculate difference between two dates"""
        try:
            # Parse dates
            dt1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            
            # Calculate difference
            diff = abs(dt2 - dt1)
            
            return {
                "date1": date1,
                "date2": date2,
                "difference_days": diff.days,
                "difference_seconds": diff.total_seconds(),
                "difference_hours": diff.total_seconds() / 3600,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Date calculation error: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def format_relative_time(self, timestamp: str) -> Dict[str, Any]:
        """Format timestamp as relative time"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                relative = f"{diff.days}日前"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                relative = f"{hours}時間前"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                relative = f"{minutes}分前"
            else:
                relative = "たった今"
            
            return {
                "timestamp": timestamp,
                "relative_time": relative,
                "exact_diff": str(diff),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Relative time error: {e}")
            return {
                "error": str(e),
                "success": False
            }

class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools = {
            "web_search": WebSearchTool(),
            "calculator": CalculatorTool(),
            "data_analysis": DataAnalysisTool(),
            "datetime": DateTimeTool(),
        }
    
    def get_tool(self, tool_name: str):
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List available tool names"""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, method: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool method with given arguments"""
        try:
            tool = self.get_tool(tool_name)
            if not tool:
                return {"error": f"Tool {tool_name} not found", "success": False}
            
            if not hasattr(tool, method):
                return {"error": f"Method {method} not found in tool {tool_name}", "success": False}
            
            method_func = getattr(tool, method)
            result = method_func(**kwargs)
            
            # Add execution metadata
            if isinstance(result, dict):
                result["tool_name"] = tool_name
                result["method"] = method
                result["execution_time"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "error": str(e),
                "success": False,
                "tool_name": tool_name,
                "method": method
            }

def get_tool_descriptions() -> Dict[str, Dict[str, Any]]:
    """Get descriptions of all available tools"""
    return {
        "web_search": {
            "description": "ウェブ検索を実行して最新情報を取得",
            "methods": {
                "search": {
                    "description": "クエリでウェブ検索",
                    "parameters": {
                        "query": "検索クエリ文字列",
                        "num_results": "取得する結果数（デフォルト: 5）"
                    }
                }
            }
        },
        "calculator": {
            "description": "数学的計算を実行",
            "methods": {
                "calculate": {
                    "description": "数式を評価",
                    "parameters": {
                        "expression": "計算する数式"
                    }
                }
            }
        },
        "data_analysis": {
            "description": "データ分析とパンダス操作",
            "methods": {
                "analyze_data": {
                    "description": "データを分析",
                    "parameters": {
                        "data": "分析するデータのリスト",
                        "operation": "実行する操作（describe, count, summary, sample）"
                    }
                }
            }
        },
        "datetime": {
            "description": "日付・時刻操作",
            "methods": {
                "get_current_datetime": {
                    "description": "現在の日時を取得",
                    "parameters": {}
                },
                "calculate_date_difference": {
                    "description": "日付の差を計算",
                    "parameters": {
                        "date1": "最初の日付（ISO形式）",
                        "date2": "2番目の日付（ISO形式）"
                    }
                },
                "format_relative_time": {
                    "description": "相対時刻でフォーマット",
                    "parameters": {
                        "timestamp": "フォーマットするタイムスタンプ"
                    }
                }
            }
        }
    }

# Global tool registry instance
tool_registry = ToolRegistry()
