"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class NewMessageRequest(BaseModel):
    """Request model for inserting text into the database."""
    
    chat_id: int = Field(..., description="chat ID in the database")
    user_id: int = Field(..., description="user ID for ownership verification")
    text: str = Field(..., description="Text from user")


class NewMessageResponse(BaseModel):
    """Response model for successful insert operation."""
    
    status: str = Field(default="success", description="Operation status")
    gigachat_response: str = Field(..., description="Response of gigachat")


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    status: str = Field(default="error", description="Operation status")
    detail: str = Field(..., description="Error description")


# ==================== Auth models ====================

class RegisterRequest(BaseModel):
    """Request model for user registration."""
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=1, max_length=100, description="Password")


class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class AuthResponse(BaseModel):
    """Response model for authentication."""
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")


# ==================== Chat models ====================

class CreateChatRequest(BaseModel):
    """Request model for creating a new chat."""
    user_id: int = Field(..., description="User ID")
    title: str = Field(default="Новый чат", description="Chat title")


class ChatInfo(BaseModel):
    """Chat information model."""
    id: int = Field(..., description="Chat ID")
    title: str = Field(..., description="Chat title")
    created_at: str = Field(..., description="Creation timestamp")


class ChatListResponse(BaseModel):
    """Response model for chat list."""
    chats: list[ChatInfo] = Field(..., description="List of user chats")


class DeleteChatRequest(BaseModel):
    """Request model for deleting a chat."""
    user_id: int = Field(..., description="User ID")
    chat_id: int = Field(..., description="Chat ID to delete")