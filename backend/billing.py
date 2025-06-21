import stripe
import os
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import User
from auth import get_current_user
from datetime import datetime
from typing import Dict, Any
import logging

# Configure Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "sk_test_...")

router = APIRouter(prefix="/api/billing", tags=["billing"])

# Configure logging
logger = logging.getLogger(__name__)

# Pricing plans aligned with our market strategy
PRICING_PLANS = {
    "free": {
        "name": "Free",
        "credits": 100,
        "price": 0,
        "features": ["100 free labels", "Basic image classification", "CSV export"],
        "target": "Testing & small projects"
    },
    "starter": {
        "name": "Starter",
        "credits": 10000,
        "price": 4900,  # $49.00 in cents
        "features": [
            "10,000 labels/month",
            "Advanced ML models", 
            "Batch processing",
            "Multiple export formats",
            "Email support"
        ],
        "target": "AI startups & small teams"
    },
    "professional": {
        "name": "Professional", 
        "credits": 100000,
        "price": 19900,  # $199.00 in cents
        "features": [
            "100,000 labels/month",
            "Custom model training",
            "API access",
            "Priority processing",
            "Human review interface",
            "Advanced analytics",
            "Priority support"
        ],
        "target": "Growing AI companies"
    },
    "enterprise": {
        "name": "Enterprise",
        "credits": 1000000,
        "price": 99900,  # $999.00 in cents
        "features": [
            "1M+ labels/month",
            "Custom deployment",
            "SLA guarantee",
            "Dedicated support",
            "Custom integrations",
            "Advanced security",
            "White-label options"
        ],
        "target": "Large organizations"
    }
}

@router.get("/plans")
async def get_pricing_plans():
    """Get all available pricing plans"""
    return {
        "plans": PRICING_PLANS,
        "current_promotion": {
            "message": "50% off first month for early adopters",
            "code": "EARLY50",
            "expires": "2024-12-31"
        }
    }

@router.post("/create-payment-intent")
async def create_payment_intent(
    plan: str,
    promotion_code: str = None,
    current_user: User = Depends(get_current_user)
):
    """Create Stripe payment intent for subscription"""
    
    if plan not in PRICING_PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    plan_info = PRICING_PLANS[plan]
    amount = plan_info["price"]
    
    # Apply promotion if valid
    discount = 0
    if promotion_code == "EARLY50" and plan != "free":
        discount = amount // 2
        amount = amount - discount
    
    # Skip payment for free plan
    if plan == "free":
        return {"message": "Free plan - no payment required"}
    
    try:
        # Create or get Stripe customer
        if not hasattr(current_user, 'stripe_customer_id') or not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': current_user.id}
            )
            current_user.stripe_customer_id = customer.id
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency='usd',
            customer=current_user.stripe_customer_id,
            metadata={
                'user_id': current_user.id,
                'plan': plan,
                'original_amount': plan_info["price"],
                'discount': discount,
                'promotion_code': promotion_code or ""
            }
        )
        
        return {
            "client_secret": intent.client_secret,
            "amount": amount,
            "original_amount": plan_info["price"],
            "discount": discount,
            "plan": plan_info
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Payment error: {str(e)}")

@router.post("/confirm-payment")
async def confirm_payment(
    payment_intent_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Confirm payment and upgrade user plan"""
    
    try:
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        if intent.status == 'succeeded':
            plan = intent.metadata['plan']
            credits = PRICING_PLANS[plan]["credits"]
            
            # Update user credits and plan
            current_user.credits_remaining += credits
            current_user.subscription_tier = plan
            current_user.last_payment_date = datetime.utcnow()
            
            # Log the successful payment
            logger.info(f"Payment confirmed for user {current_user.id}: {plan} plan, {credits} credits")
            
            db.commit()
            
            return {
                "success": True,
                "plan": plan,
                "plan_info": PRICING_PLANS[plan],
                "credits_added": credits,
                "total_credits": current_user.credits_remaining,
                "message": f"Successfully upgraded to {PRICING_PLANS[plan]['name']} plan!"
            }
        else:
            raise HTTPException(status_code=400, detail="Payment not completed")
            
    except stripe.error.StripeError as e:
        logger.error(f"Payment confirmation error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Payment confirmation failed: {str(e)}")

@router.get("/usage")
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user usage statistics"""
    
    # Calculate usage stats
    from models import Job, Result
    
    total_jobs = db.query(Job).filter(Job.user_id == current_user.id).count()
    completed_jobs = db.query(Job).filter(
        Job.user_id == current_user.id,
        Job.status == "completed"
    ).count()
    
    total_labels = db.query(Result).join(Job).filter(
        Job.user_id == current_user.id,
        Result.status == "success"
    ).count()
    
    current_plan = PRICING_PLANS.get(current_user.subscription_tier, PRICING_PLANS["free"])
    
    return {
        "current_plan": {
            "name": current_plan["name"],
            "tier": current_user.subscription_tier,
            "features": current_plan["features"]
        },
        "credits": {
            "remaining": current_user.credits_remaining,
            "monthly_limit": current_plan["credits"],
            "usage_percentage": max(0, (current_plan["credits"] - current_user.credits_remaining) / current_plan["credits"] * 100)
        },
        "usage_stats": {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "total_labels": total_labels,
            "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        },
        "billing_info": {
            "last_payment": current_user.last_payment_date.isoformat() if hasattr(current_user, 'last_payment_date') and current_user.last_payment_date else None,
            "next_billing": "Monthly recurring" if current_user.subscription_tier != "free" else None
        }
    }

@router.post("/add-credits")
async def add_credits(
    credits: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add credits for testing (admin only in production)"""
    
    # TODO: Add admin check in production
    current_user.credits_remaining += credits
    db.commit()
    
    return {
        "success": True,
        "credits_added": credits,
        "total_credits": current_user.credits_remaining
    }

@router.post("/webhook")
async def stripe_webhook(
    request: dict,
    background_tasks: BackgroundTasks
):
    """Handle Stripe webhooks for subscription management"""
    
    # TODO: Implement webhook signature verification
    # TODO: Handle subscription events (created, updated, canceled)
    
    event_type = request.get('type')
    
    if event_type == 'payment_intent.succeeded':
        # Handle successful payment
        background_tasks.add_task(process_successful_payment, request['data']['object'])
    elif event_type == 'payment_intent.payment_failed':
        # Handle failed payment
        background_tasks.add_task(process_failed_payment, request['data']['object'])
    
    return {"status": "received"}

async def process_successful_payment(payment_intent: dict):
    """Process successful payment in background"""
    logger.info(f"Payment succeeded: {payment_intent['id']}")

async def process_failed_payment(payment_intent: dict):
    """Process failed payment in background"""
    logger.warning(f"Payment failed: {payment_intent['id']}")