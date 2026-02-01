"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field


class NewMessageRequest(BaseModel):
    """Request model for inserting text into the database."""
    
    chat_id: int = Field(..., description="chat ID in the database")
    text: str = Field(..., description="Text from user")


class NewMessageResponse(BaseModel):
    """Response model for successful insert operation."""
    
    status: str = Field(default="success", description="Operation status")
    gigachat_response: str = Field(..., description="Response of gigachat")


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    status: str = Field(default="error", description="Operation status")
    detail: str = Field(..., description="Error description")