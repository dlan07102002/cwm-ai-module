"""
Production-ready FastAPI service with:
- Caching for performance
- Comprehensive error handling
- Request validation
- Monitoring and metrics
- Health checks
- Batch prediction support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import lightgbm as lgb
from .feature_builder import FeatureBuilder
from pathlib import Path
import logging
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# App Initialization
# ============================================================
app = FastAPI(
    title="🚗 Vehicle Recommender API",
    version="2.0",
    description="Production-ready vehicle ranking and recommendation system"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_PATH = Path(__file__).parent.parent / "models" / "vehicle_ranker.txt"
METADATA_PATH = Path(__file__).parent.parent / "models" / "model_metadata.json"

model = None
model_metadata = None
feature_builder = None
request_count = 0
error_count = 0

# ============================================================
# Request/Response Models
# ============================================================

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="User UUID")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of recommendations")
    max_candidates: int = Field(default=100, ge=10, le=500, description="Max candidates to consider")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid user_id format')
        return v


class RecommendBuyerRequest(BaseModel):
    vehicle_id: str = Field(..., description="Vehicle UUID")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of recommendations")
    max_candidates: int = Field(default=100, ge=10, le=500, description="Max candidates to consider")
    
    @validator('vehicle_id')
    def validate_vehicle_id(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Invalid vehicle_id format')
        return v


class BatchRecommendRequest(BaseModel):
    user_ids: List[str] = Field(..., description="List of user UUIDs", max_items=100)
    top_k: int = Field(default=5, ge=1, le=20)
    max_candidates: int = Field(default=50, ge=10, le=200)


class RecommendationResponse(BaseModel):
    vehicle_id: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationResponse]
    total_candidates: int
    processing_time_ms: float
    timestamp: str


class BuyerRecommendationResponse(BaseModel):
    user_id: str
    score: float
    rank: int


class BuyerRecommendResponse(BaseModel):
    vehicle_id: str
    recommendations: List[BuyerRecommendationResponse]
    total_candidates: int
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    total_requests: int
    error_count: int


# ============================================================
# Helper Functions
# ============================================================

def load_model():
    """Load the trained model and metadata."""
    global model, model_metadata, feature_builder
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train it first using train_ranker.py."
        )
    
    model = lgb.Booster(model_file=str(MODEL_PATH))
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    
    # Load metadata if available
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            model_metadata = json.load(f)
        logger.info(f"✅ Model metadata loaded: {model_metadata.get('timestamp')}")
    
    # Initialize feature builder with caching
    feature_builder = FeatureBuilder(cache_ttl_minutes=5)
    logger.info("✅ Feature builder initialized")


def track_request():
    """Track request metrics."""
    global request_count
    request_count += 1


def track_error():
    """Track error metrics."""
    global error_count
    error_count += 1


# ============================================================
# Startup/Shutdown Events
# ============================================================

start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("🚀 Starting Vehicle Recommender API...")
    try:
        load_model()
        logger.info("✅ API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("👋 Shutting down Vehicle Recommender API...")
    if feature_builder:
        feature_builder.close()
    logger.info("✅ Cleanup completed")


# ============================================================
# API Routes
# ============================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🚗 Vehicle Recommender API is running!",
        "version": "2.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get('timestamp') if model_metadata else None,
        uptime_seconds=uptime,
        total_requests=request_count,
        error_count=error_count
    )


@app.post("/recommend/vehicles", response_model=RecommendResponse)
async def recommend_vehicles(req: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Recommend top-K vehicles for a specific user.
    
    This endpoint uses real-time feature generation to provide
    personalized vehicle recommendations based on user preferences
    and current vehicle inventory.
    """
    start_time_ms = time.time() * 1000
    
    background_tasks.add_task(track_request)
    
    if model is None or feature_builder is None:
        track_error()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Build features for this user
        logger.info(f"Processing recommendation request for user: {req.user_id}")
        
        df = feature_builder.build_features(
            include_interactions=False,
            max_candidates_per_user=req.max_candidates,
            user_id=req.user_id
        )
        
        if df.empty:
            logger.warning(f"No candidates found for user: {req.user_id}")
            return RecommendResponse(
                user_id=req.user_id,
                recommendations=[],
                total_candidates=0,
                processing_time_ms=time.time() * 1000 - start_time_ms,
                timestamp=datetime.now().isoformat()
            )
        
        # Prepare features
        feature_cols = model.feature_name()
        X = df[feature_cols]
        
        # Predict scores
        preds = model.predict(X)
        df['score'] = preds
        
        # Rank and select top-K
        topk = (
            df.groupby("vehicle_id")["score"]
            .max()
            .reset_index()
            .sort_values("score", ascending=False)
            .head(req.top_k)
            .reset_index(drop=True)
        )
        
        # Format response
        recommendations = [
            RecommendationResponse(
                vehicle_id=row['vehicle_id'],
                score=float(row['score']),
                rank=idx + 1
            )
            for idx, row in topk.iterrows()
        ]
        
        processing_time = time.time() * 1000 - start_time_ms
        
        logger.info(
            f"Recommendation completed for user {req.user_id}: "
            f"{len(recommendations)} vehicles in {processing_time:.2f}ms"
        )
        
        return RecommendResponse(
            user_id=req.user_id,
            recommendations=recommendations,
            total_candidates=len(df),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        track_error()
        logger.error(f"Error processing recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post("/recommend/buyers", response_model=BuyerRecommendResponse)
async def recommend_buyers(req: RecommendBuyerRequest, background_tasks: BackgroundTasks):
    """
    Recommend top-K potential buyers for a specific vehicle.
    
    This endpoint identifies users whose preferences and budgets
    align with the given vehicle's characteristics.
    """
    start_time_ms = time.time() * 1000
    
    background_tasks.add_task(track_request)
    
    if model is None or feature_builder is None:
        track_error()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Build features for this vehicle
        logger.info(f"Processing buyer recommendation for vehicle: {req.vehicle_id}")
        
        df = feature_builder.build_features(
            include_interactions=False,
            max_candidates_per_user=req.max_candidates,
            vehicle_id=req.vehicle_id
        )
        
        if df.empty:
            logger.warning(f"No buyers found for vehicle: {req.vehicle_id}")
            return BuyerRecommendResponse(
                vehicle_id=req.vehicle_id,
                recommendations=[],
                total_candidates=0,
                processing_time_ms=time.time() * 1000 - start_time_ms,
                timestamp=datetime.now().isoformat()
            )
        
        # Prepare features
        feature_cols = model.feature_name()
        X = df[feature_cols]
        
        # Predict scores
        preds = model.predict(X)
        df['score'] = preds
        
        # Rank and select top-K buyers
        topk = (
            df.groupby("user_id")["score"]
            .max()
            .reset_index()
            .sort_values("score", ascending=False)
            .head(req.top_k)
            .reset_index(drop=True)
        )
        
        # Format response
        recommendations = [
            BuyerRecommendationResponse(
                user_id=row['user_id'],
                score=float(row['score']),
                rank=idx + 1
            )
            for idx, row in topk.iterrows()
        ]
        
        processing_time = time.time() * 1000 - start_time_ms
        
        logger.info(
            f"Buyer recommendation completed for vehicle {req.vehicle_id}: "
            f"{len(recommendations)} buyers in {processing_time:.2f}ms"
        )
        
        return BuyerRecommendResponse(
            vehicle_id=req.vehicle_id,
            recommendations=recommendations,
            total_candidates=len(df),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        track_error()
        logger.error(f"Error processing buyer recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate buyer recommendations: {str(e)}"
        )


@app.post("/recommend/batch")
async def batch_recommend(req: BatchRecommendRequest, background_tasks: BackgroundTasks):
    """
    Batch recommendation endpoint for multiple users.
    
    Returns recommendations for up to 100 users in a single request.
    Useful for batch processing or email campaigns.
    """
    start_time_ms = time.time() * 1000
    
    background_tasks.add_task(track_request)
    
    if model is None or feature_builder is None:
        track_error()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        results = []
        
        for user_id in req.user_ids:
            try:
                df = feature_builder.build_features(
                    include_interactions=False,
                    max_candidates_per_user=req.max_candidates,
                    user_id=user_id
                )
                
                if df.empty:
                    results.append({
                        "user_id": user_id,
                        "recommendations": [],
                        "error": None
                    })
                    continue
                
                feature_cols = model.feature_name()
                X = df[feature_cols]
                preds = model.predict(X)
                df['score'] = preds
                
                topk = (
                    df.groupby("vehicle_id")["score"]
                    .max()
                    .reset_index()
                    .sort_values("score", ascending=False)
                    .head(req.top_k)
                )
                
                recommendations = [
                    {"vehicle_id": row['vehicle_id'], "score": float(row['score'])}
                    for _, row in topk.iterrows()
                ]
                
                results.append({
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "error": None
                })
                
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                results.append({
                    "user_id": user_id,
                    "recommendations": [],
                    "error": str(e)
                })
        
        processing_time = time.time() * 1000 - start_time_ms
        
        return {
            "results": results,
            "total_users": len(req.user_ids),
            "successful": sum(1 for r in results if r['error'] is None),
            "failed": sum(1 for r in results if r['error'] is not None),
            "processing_time_ms": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        track_error()
        logger.error(f"Error in batch recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch recommendation failed: {str(e)}"
        )


@app.post("/reload_model")
async def reload_model():
    """
    Reload the trained model manually without restarting the API.
    
    Useful after retraining the model to hot-reload without downtime.
    """
    try:
        logger.info("Reloading model...")
        load_model()
        return {
            "message": "✅ Model reloaded successfully",
            "timestamp": datetime.now().isoformat(),
            "version": model_metadata.get('timestamp') if model_metadata else None
        }
    except Exception as e:
        track_error()
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "features": model.feature_name() if model else [],
        "num_features": len(model.feature_name()) if model else 0,
        "metadata": model_metadata,
        "num_trees": model.num_trees() if model else 0
    }


@app.get("/metrics")
async def get_metrics():
    """Get API metrics and statistics."""
    uptime = time.time() - start_time
    
    return {
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "error_count": error_count,
        "error_rate": error_count / request_count if request_count > 0 else 0,
        "requests_per_second": request_count / uptime if uptime > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    track_error()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="Internal server error. Please check server logs."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")